import io
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from infovore.claims.httpd import payload
from infovore.claims.value import NearDuplicateIndex, sample_ids, tokens, wilson
from infovore.cli import ExitCode, main
from infovore.db.claim_checks import CheckRow, record_checks
from infovore.db.claims_v2 import ClaimIn, ExchangeOutcome, record_exchange, record_review
from tests.claims.seed import SALT, conversation, db, environment
from tests.claims.test_store import AT, make_run

OK = ExchangeOutcome("ok", None, 1, 1, 1, 1.0)
FAILED = ExchangeOutcome("failed", "boom", 0, 0, 0, 0.0)


def run_value(tmp_path: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["claims", "value", *argv],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


def claim(text: str) -> list[ClaimIn]:
    return [ClaimIn("u", text, ())]


def populate(tmp_path: Path) -> sqlite3.Connection:
    conn = db(tmp_path)
    conn.execute("INSERT INTO channels (id, guild_id, name, kind) VALUES (2, 9, 'd', 'text')")
    prior = make_run(conn)
    p, _ = conversation(conn, [(1, "ann", "x")], 9)
    record_exchange(conn, prior, p, OK, claim("Indy has a slow boot disk"), [])
    run = make_run(conn)
    a, _ = conversation(conn, [(1, "ann", "x")], 1)
    b, _ = conversation(conn, [(1, "ann", "x")], 2)
    c, _ = conversation(conn, [(1, "ann", "x")], 3)
    d, _ = conversation(conn, [(1, "ann", "x")], 4)
    e, _ = conversation(conn, [(1, "ann", "x")], 5)
    conn.execute("UPDATE exchanges SET channel_id = 2 WHERE id = ?", (c,))
    both = [
        ClaimIn("u", "O2 has a quiet fan unit", ()),
        ClaimIn("u", "Octane uses a loud power supply", ()),
    ]
    record_exchange(conn, run, a, OK, both, [])
    record_exchange(conn, run, b, OK, claim("O2 has a quiet fan unit"), [])
    record_exchange(conn, run, c, OK, claim("Indy has a slow boot disk"), [])
    record_exchange(conn, run, d, OK, [], [])
    record_exchange(conn, run, e, FAILED, [], [])
    return conn


def test_tokens_drop_stopwords_and_case() -> None:
    assert tokens("The O2 is a Quiet fan") == frozenset({"o2", "quiet", "fan"})


def test_near_duplicates_are_found_exact_and_close_but_not_distinct_or_empty() -> None:
    index = NearDuplicateIndex()
    base = tokens("indy has slow boot disk noisy fan today")
    assert not index.seen(base)
    index.add(base)
    index.add(frozenset())
    assert index.seen(base)
    assert index.seen(tokens("indy has slow boot disk noisy fan"))
    assert not index.seen(tokens("octane uses loud power supply"))
    assert not index.seen(frozenset())
    assert not index.seen(tokens("indy has slow boot disk entirely different words here now"))


def test_wilson_interval() -> None:
    interval = wilson(5, 10)
    assert interval is not None
    assert (round(interval[0], 4), round(interval[1], 4)) == (0.2366, 0.7634)
    assert wilson(0, 0) is None


def test_sample_ids_are_seeded_sorted_and_bounded() -> None:
    ids = list(range(1, 101))
    first = sample_ids(ids, 10, 7)
    assert first == sample_ids(ids, 10, 7) and first == sorted(first) and len(set(first)) == 10
    assert first != sample_ids(ids, 10, 8)
    assert sample_ids(ids, 500, 7) == ids


def test_value_reports_every_section(tmp_path: Path) -> None:
    conn = populate(tmp_path)
    record_checks(conn, [CheckRow(2, "supported", 1.0, [])], {"r": 1}, AT)
    for claim_id, verdict, interface in (
        (1, "good", "conversation"),
        (2, "good", "conversation"),
        (3, "good", "conversation"),
        (4, "wrong", "conversation"),
        (5, "good", "cited-only"),
    ):
        record_review(conn, claim_id, verdict, AT, interface)
    conn.close()

    code, out, _ = run_value(tmp_path, "--runs", "2", "--min-claims", "2")

    assert code == ExitCode.OK
    assert "runs 2: conversations 4 (failed 1), claims 4 (1.00 per conversation), rejected 0" in out
    assert "first half: 1 of 3 claims (33.3%) over 2 conversations" in out
    assert "second half: 1 of 1 claims (100.0%) over 2 conversations" in out
    assert "new subjects: 2 (500.0 per 1000 conversations)" in out
    assert "reaching 2 claims: 2 (500.0 per 1000 conversations)" in out
    assert "conversations 1-4: new subjects 2, reaching 2" in out
    assert "c: conversations 3, claims per conversation 1.00, new subjects 2" in out
    assert "d: conversations 1, claims per conversation 1.00, new subjects 0" in out
    assert "claim checks: supported 1, unchecked 3" in out
    assert "good 2 of 3 (66.7%), Wilson 95% 20.8% to 93.9%" in out


def test_value_over_two_runs_and_nothing_checked_or_reviewed(tmp_path: Path) -> None:
    populate(tmp_path).close()

    code, out, _ = run_value(tmp_path, "--runs", "1,2")

    assert code == ExitCode.OK
    assert "runs 1,2: conversations 5 (failed 1), claims 5" in out
    assert "claim checks: none" in out
    assert "reviewed sample: none" in out
    assert "reaching 3 claims: 0" in out


def test_value_windows_split_every_thousand_conversations(tmp_path: Path) -> None:
    conn = db(tmp_path)
    run = make_run(conn)
    for n in range(1, 1002):
        eid, _ = conversation(conn, [(1, "ann", "x")], n)
        record_exchange(conn, run, eid, OK, claim("Indy"), [])
    conn.close()

    _, out, _ = run_value(tmp_path, "--runs", "1", "--min-claims", "1")

    assert "conversations 1-1000: new subjects 1, reaching 1" in out
    assert "conversations 1001-1001: new subjects 0, reaching 0" in out


def test_value_with_no_conversations_has_nothing_to_divide(tmp_path: Path) -> None:
    conn = db(tmp_path)
    make_run(conn)
    conn.close()

    code, out, _ = run_value(tmp_path, "--runs", "1")

    assert code == ExitCode.OK and "claims 0 (n/a per conversation)" in out
    assert "first half: 0 of 0 claims (n/a)" in out


@pytest.mark.parametrize(
    "argv", [["--runs", "9"], ["--runs", "x"], ["--runs", "1", "--min-claims", "0"]]
)
def test_value_refuses_bad_arguments(tmp_path: Path, argv: list[str]) -> None:
    populate(tmp_path).close()

    code, _, err = run_value(tmp_path, *argv)

    assert code == ExitCode.CONFIG and err


def seeded_run(tmp_path: Path) -> tuple[sqlite3.Connection, int]:
    conn = db(tmp_path)
    run = make_run(conn)
    for n in range(1, 4):
        eid, ids = conversation(conn, [(1, "ann", "hi")], n)
        record_exchange(conn, run, eid, OK, [ClaimIn("u", f"u said {n}", (ids[0],))] * 2, [])
    return conn, run


def test_payload_can_be_limited_to_sampled_claims(tmp_path: Path) -> None:
    conn, run = seeded_run(tmp_path)

    data: dict[str, Any] = payload(conn, run, SALT, frozenset({2, 5}))

    assert [[c["id"] for c in conv["claims"]] for conv in data["conversations"]] == [[2], [5]]
    assert len(payload(conn, run, SALT)["conversations"]) == 3


def serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *argv: str) -> tuple[int, str, str]:
    monkeypatch.setattr("infovore.sift.httpd.block_until_interrupted", lambda event: None)
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["claims", "serve", "--run", "1", "--port", "0", *argv],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


def test_serve_sample_reports_the_sample_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded_run(tmp_path)[0].close()

    code, out, _ = serve(tmp_path, monkeypatch, "--sample", "4", "--seed", "3")

    assert code == ExitCode.OK and "sample 4 of 6 claims, 0 reviewed" in out


def test_serve_sample_must_be_positive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seeded_run(tmp_path)[0].close()

    code, _, err = serve(tmp_path, monkeypatch, "--sample", "0")

    assert code == ExitCode.CONFIG and "--sample must be positive" in err
