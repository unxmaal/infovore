import json
import re
from pathlib import Path
from typing import Any

import pytest

from infovore.claims.extract import (
    CLAIM_MAX_CHARS,
    SCHEMA,
    ClaimsReplyError,
    RawClaim,
    build_request,
    extract_exchange,
    fetch_model_id,
    parse_reply,
    prompt_hash,
    render_window,
    validate,
    windows,
)
from infovore.claims.redact import RenderedLine, pseudonym, redact_conversation
from tests.claims.seed import SALT, conversation, db, messages_of

OLD_HASHES = {"a82adb7b1e83"}
U1, U2 = pseudonym(1, SALT), pseudonym(2, SALT)


def lines(*texts: str) -> list[RenderedLine]:
    return [RenderedLine(i + 1, 100 + i, U1, t) for i, t in enumerate(texts)]


def reply(claims: list[list[Any]], tin: int = 10, tout: int = 3) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": json.dumps({"c": claims})}}],
        "usage": {"prompt_tokens": tin, "completion_tokens": tout},
    }


def test_the_request_pins_a_strict_schema_and_names_the_scope() -> None:
    request = build_request("eval-4b", "[1] user-aaaa: hi")
    system = request["messages"][0]["content"]

    assert request["model"] == "eval-4b" and request["temperature"] == 0
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"] == SCHEMA
    for phrase in ("standalone", "IRIX", "first-hand", "zero", "pronouns", "questions", "jokes"):
        assert phrase in system
    assert system.count("Bad:") == 3 and system.count("Good:") == 3
    assert request["max_tokens"] == 400 and build_request("m", "w", 77)["max_tokens"] == 77
    assert len(system) < 1700
    assert request["messages"][1]["content"].endswith("[1] user-aaaa: hi")
    assert prompt_hash() == prompt_hash() and len(prompt_hash()) == 12
    assert prompt_hash() not in OLD_HASHES


def test_the_schema_is_compact_and_caps_claim_length() -> None:
    item = SCHEMA["properties"]["c"]["items"]
    s, t, r = item["prefixItems"]

    assert list(SCHEMA["properties"]) == ["c"] and SCHEMA["required"] == ["c"]
    assert item["minItems"] == item["maxItems"] == 3 and item["type"] == "array"
    assert s["type"] == "string" and r["type"] == "array" and r["minItems"] == 1
    assert t["maxLength"] == CLAIM_MAX_CHARS


def test_a_reply_cut_off_at_the_token_cap_is_a_failure() -> None:
    cut = reply([[U1, "s", [1]]])
    cut["choices"][0]["finish_reason"] = "length"

    with pytest.raises(ClaimsReplyError, match="truncated"):
        parse_reply(cut, 0.0)
    cut["choices"][0]["finish_reason"] = "stop"
    assert len(parse_reply(cut, 0.0).claims) == 1


def test_the_token_cap_reaches_every_request(redacted: Any) -> None:
    sent: list[dict[str, Any]] = []

    def post(payload: Any) -> tuple[dict[str, Any], float]:
        sent.append(dict(payload))
        return reply([]), 0.1

    extract_exchange(post, "m", redacted, 1000, 55)
    extract_exchange(post, "m", redacted, 1000)

    assert [p["max_tokens"] for p in sent] == [55, 400]


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", text))


FIXTURE = (
    "[1] user-1a2b: my Indy runs IRIX 6.5.3 off an external SCSI disk\n"
    "[2] user-3c4d: try the PROM monitor to boot from it"
)
OLD_EXAMPLE = {
    "claims": [
        {
            "speaker": "user-1a2b",
            "statement": "user-1a2b says their SGI Indy runs IRIX 6.5.3 from an external disk.",
            "refs": [1],
        },
        {
            "speaker": "user-3c4d",
            "statement": "user-3c4d says the PROM monitor can be used to boot an Indy from an"
            " external disk.",
            "refs": [2],
        },
    ]
}
NEW_EXAMPLE = {
    "c": [
        ["user-1a2b", "Indy runs IRIX 6.5.3 from external SCSI disk.", [1]],
        ["user-3c4d", "PROM monitor boots an Indy from external disk.", [2]],
    ]
}


def test_the_new_example_output_is_at_least_35_percent_shorter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    old = estimate_tokens(json.dumps(OLD_EXAMPLE, separators=(",", ":")))
    new = estimate_tokens(json.dumps(NEW_EXAMPLE, separators=(",", ":")))
    with capsys.disabled():
        print(f"\nexample output tokens (estimate): old={old} new={new} saved={1 - new / old:.0%}")

    assert new <= old * 0.65
    assert FIXTURE.count("\n") == 1
    for claim in NEW_EXAMPLE["c"]:
        assert len(claim[1]) <= CLAIM_MAX_CHARS
    parsed = parse_reply({"choices": [{"message": {"content": json.dumps(NEW_EXAMPLE)}}]}, 0.0)
    assert [c.refs for c in parsed.claims] == [(1,), (2,)]


def test_windows_never_split_a_message_and_keep_global_refs() -> None:
    parts = windows(lines("aaaa", "bbbb", "cccc"), 400)
    assert len(parts) == 1

    parts = windows(lines("aaaa", "bbbb", "cccc"), len(render_window(lines("aaaa"))) + 5)
    assert [[line.ref for line in part] for part in parts] == [[1], [2], [3]]

    long = windows(lines("z" * 500), 60)
    assert len(long) == 1 and len(render_window(long[0])) <= 60
    assert windows([], 60) == []


def test_a_window_renders_one_ref_and_speaker_per_line() -> None:
    assert render_window(lines("hi", "yo")) == f"[1] {U1}: hi\n[2] {U1}: yo"


def test_replies_parse_into_claims_and_usage() -> None:
    parsed = parse_reply(reply([[U1, "s", [1, 2]]], 7, 2), 1.5)

    assert parsed.claims == [RawClaim(U1, "s", (1, 2))]
    assert (parsed.input_tokens, parsed.output_tokens, parsed.seconds) == (7, 2, 1.5)
    bare = parse_reply({"choices": [{"message": {"content": '{"c": []}'}}]}, 0)
    assert (bare.claims, bare.input_tokens) == ([], 0)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"choices": [{"message": {"content": "nope"}}]},
        {"choices": [{"message": {"content": '{"c": 3}'}}]},
        {"choices": [{"message": {"content": '{"c": [["u"]]}'}}]},
        {"choices": [{"message": {"content": '{"c": [1]}'}}]},
    ],
)
def test_unusable_replies_raise(bad: dict[str, Any]) -> None:
    with pytest.raises(ClaimsReplyError):
        parse_reply(bad, 0.0)


@pytest.fixture
def redacted(tmp_path: Path) -> Any:
    conn = db(tmp_path)
    eid, _ = conversation(conn, [(1, "alice", "my Indy boots"), (2, "bobby", "mine too")], 1)
    return redact_conversation(messages_of(conn, eid), SALT)


def test_validation_accepts_a_grounded_attributed_claim(redacted: Any) -> None:
    good = RawClaim(U1, f"{U1} said the Indy boots", (1, 1, 2))

    accepted, rejected = validate([good], redacted)

    assert [(c.speaker, c.message_ids) for c in accepted] == [
        (U1, (redacted.lines[0].message_id, redacted.lines[1].message_id))
    ]
    assert rejected == []


@pytest.mark.parametrize(
    "statement",
    [
        "The SGI O2 can take an R12000 CPU; one user's runs at 400 MHz.",
        "sa, which contains sash, is 15.7 MB in the copy on IRIXnet's nonfree.",
        "A socketed O2 board may allow a CPU swap.",
    ],
)
def test_attribution_is_the_speaker_field_not_the_statement_text(
    redacted: Any, statement: str
) -> None:
    accepted, rejected = validate([RawClaim(U1, statement, (1,))], redacted)

    assert [c.statement for c in accepted] == [statement] and rejected == []


@pytest.mark.parametrize(
    ("claim", "reason"),
    [
        (RawClaim(U1, f"{U1} said x", (1, 9)), "unknown ref 9"),
        (RawClaim(U1, f"{U1} said x", ()), "no refs"),
        (RawClaim("user-zzzz", "user-zzzz said x", (1,)), "unknown speaker"),
        (RawClaim(U1, "  ", (1,)), "empty statement"),
        (RawClaim(U1, f"{U1} said bobby's Indy boots", (1,)), "name leak"),
    ],
)
def test_validation_rejects_and_records_why(redacted: Any, claim: RawClaim, reason: str) -> None:
    accepted, rejected = validate([claim], redacted)

    assert accepted == []
    assert [r.reason for r in rejected] == [reason]
    assert rejected[0].statement == claim.statement
    assert rejected[0].cited == json.dumps(list(claim.refs))


def test_an_exchange_is_windowed_summed_and_validated(redacted: Any) -> None:
    sent: list[dict[str, Any]] = []
    answers = iter(
        [
            reply([[U1, f"{U1} said a", [1]]], 10, 1),
            reply([[U2, f"{U2} said b", [2]]], 20, 2),
        ]
    )

    def post(payload: Any) -> tuple[dict[str, Any], float]:
        sent.append(dict(payload))
        return next(answers), 0.5

    one_line = len(render_window(redacted.lines[:1])) + 3
    result = extract_exchange(post, "eval-4b", redacted, one_line)

    assert len(sent) == 2
    assert [c.speaker for c in result.claims] == [U1, U2]
    assert (result.windows, result.input_tokens, result.output_tokens) == (2, 30, 3)
    assert result.seconds == 1.0
    assert result.rejected == []


def test_a_window_citing_a_ref_outside_itself_is_rejected(redacted: Any) -> None:
    answers = iter([[[U1, f"{U1} said a", [2]]], []])

    def post(payload: Any) -> tuple[dict[str, Any], float]:
        return reply(next(answers)), 0.1

    one_line = len(render_window(redacted.lines[:1])) + 3
    result = extract_exchange(post, "m", redacted, one_line)

    assert result.claims == []
    assert {r.reason for r in result.rejected} == {"ref 2 not in window"}


def test_a_conversation_with_no_text_sends_nothing(tmp_path: Path) -> None:
    conn = db(tmp_path)
    eid, _ = conversation(conn, [(1, "ann", " ")], 1)
    empty = redact_conversation(messages_of(conn, eid), SALT)

    def post(payload: Any) -> tuple[dict[str, Any], float]:
        raise AssertionError("must not call the model")

    result = extract_exchange(post, "m", empty, 1000)

    assert (result.windows, result.claims) == (0, [])


def test_model_id_comes_from_model_info_else_the_alias() -> None:
    seen: list[str] = []

    def get(url: str) -> Any:
        seen.append(url)
        return {
            "data": [
                {"model_name": "other", "litellm_params": {"model": "x/other"}},
                {"model_name": "eval-4b", "litellm_params": {"model": "openai/real-4b"}},
            ]
        }

    assert fetch_model_id("http://h:4000/v1/", "eval-4b", get) == ("openai/real-4b", "model_info")
    assert seen == ["http://h:4000/model/info"]
    assert fetch_model_id("http://h:4000", "zzz", get) == ("zzz", "alias")

    def boom(url: str) -> Any:
        raise OSError("down")

    assert fetch_model_id("http://h/v1", "eval-4b", boom) == ("eval-4b", "alias")
    assert fetch_model_id("http://h/v1", "a", lambda url: {"data": "no"}) == ("a", "alias")


@pytest.mark.parametrize("bad", ["22f9", "[12]", "I always wondered why", "user-", "user-XYZ1"])
def test_the_speaker_slot_rejects_anything_but_a_pseudonym(bad: str) -> None:
    pattern = SCHEMA["properties"]["c"]["items"]["prefixItems"][0]["pattern"]

    assert re.fullmatch(pattern.strip("^$"), bad) is None


def test_the_speaker_slot_accepts_every_pseudonym_width() -> None:
    pattern = SCHEMA["properties"]["c"]["items"]["prefixItems"][0]["pattern"]

    assert all(re.fullmatch(pattern.strip("^$"), pseudonym(7, "salt", w)) for w in (4, 5, 9))
