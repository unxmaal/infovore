import json
from pathlib import Path
from typing import Any

import pytest

from infovore.claims.extract import (
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

U1, U2 = pseudonym(1, SALT), pseudonym(2, SALT)


def lines(*texts: str) -> list[RenderedLine]:
    return [RenderedLine(i + 1, 100 + i, U1, t) for i, t in enumerate(texts)]


def reply(claims: list[dict[str, Any]], tin: int = 10, tout: int = 3) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": json.dumps({"claims": claims})}}],
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
    assert request["messages"][1]["content"].endswith("[1] user-aaaa: hi")
    assert prompt_hash() == prompt_hash() and len(prompt_hash()) == 12
    assert prompt_hash() != "a82adb7b1e83"


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
    parsed = parse_reply(reply([{"speaker": U1, "statement": "s", "refs": [1, 2]}], 7, 2), 1.5)

    assert parsed.claims == [RawClaim(U1, "s", (1, 2))]
    assert (parsed.input_tokens, parsed.output_tokens, parsed.seconds) == (7, 2, 1.5)
    bare = parse_reply({"choices": [{"message": {"content": '{"claims": []}'}}]}, 0)
    assert (bare.claims, bare.input_tokens) == ([], 0)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"choices": [{"message": {"content": "nope"}}]},
        {"choices": [{"message": {"content": '{"claims": 3}'}}]},
        {"choices": [{"message": {"content": '{"claims": [{"speaker": "u"}]}'}}]},
        {"choices": [{"message": {"content": '{"claims": [1]}'}}]},
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
            reply([{"speaker": U1, "statement": f"{U1} said a", "refs": [1]}], 10, 1),
            reply([{"speaker": U2, "statement": f"{U2} said b", "refs": [2]}], 20, 2),
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
    answers = iter([[{"speaker": U1, "statement": f"{U1} said a", "refs": [2]}], []])

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
