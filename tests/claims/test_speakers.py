import io
import json
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from infovore.claims.httpd import listening_url, shutdown_all
from infovore.claims.redact import pseudonym, redact_conversation
from infovore.claims.speakers import (
    dropped_pairs,
    exclude_dropped,
    kept_exchange_ids,
    speaker_groups,
    speaker_stats,
)
from infovore.claims.speakers_httpd import start_all
from infovore.cli import ExitCode, main
from infovore.config import ConfigError
from infovore.db.claim_checks import CheckRow, record_checks
from infovore.db.claims_v2 import ClaimIn, ExchangeOutcome, record_exchange, record_review
from infovore.db.speaker_drops import decisions, dropped_authors, record_decision
from infovore.timing import SystemClock
from infovore.wiki.build import load_claims
from tests.claims.seed import SALT, conversation, db, environment, messages_of
from tests.claims.test_serve import fetch
from tests.claims.test_store import AT, make_run

OK = ExchangeOutcome("ok", None, 1, 1, 1, 1.0)
ANN, BOB, CAT = pseudonym(1, SALT), pseudonym(2, SALT), pseudonym(3, SALT)


def seeded(tmp_path: Path) -> tuple[sqlite3.Connection, int, list[int]]:
    conn = db(tmp_path)
    e1, _ = conversation(conn, [(1, "ann", "alpha indy"), (2, "bob", "beta octane")], 1)
    e2, _ = conversation(conn, [(1, "ann", "gamma fuel"), (3, "cat", "delta o2")], 2)
    e3, _ = conversation(conn, [(2, "bob", "epsilon only")], 3)
    run = make_run(conn)
    first = [
        ClaimIn(ANN, "The Indy one fact", ()),
        ClaimIn(ANN, "The Indy two fact", ()),
        ClaimIn(BOB, "The Octane fact", ()),
        ClaimIn("user-zzzz", "A model invented speaker", ()),
    ]
    record_exchange(conn, run, e1, OK, first, [])
    second = [ClaimIn(ANN, "The Fuel fact", ()), ClaimIn(CAT, "The O2 fact", ())]
    record_exchange(conn, run, e2, OK, second, [])
    record_exchange(conn, run, e3, OK, [ClaimIn(BOB, "The Onyx fact", ())], [])
    return conn, run, [e1, e2, e3]


def test_stats_group_claims_by_author_and_ignore_unresolvable_speakers(tmp_path: Path) -> None:
    conn, run, _ = seeded(tmp_path)
    record_review(conn, 1, "good", AT, "conversation")
    record_review(conn, 2, "good", AT, "cited-only")
    record_review(conn, 3, "wrong", AT, "conversation")

    stats = speaker_stats(conn, SALT, [run])

    assert [(s.author_id, s.label, len(s.claims)) for s in stats] == [
        (1, ANN, 3),
        (2, BOB, 2),
        (3, CAT, 1),
    ]
    assert [(s.reviewed, s.good) for s in stats] == [(1, 1), (1, 0), (0, 0)]
    assert [c.verdict for c in stats[0].claims] == ["good", "good", None]


def test_stats_can_leave_out_authors(tmp_path: Path) -> None:
    conn, run, _ = seeded(tmp_path)

    stats = speaker_stats(conn, SALT, [run], frozenset({1}))

    assert [s.author_id for s in stats] == [2, 3]


def test_dropped_pairs_need_a_salt_only_when_someone_is_dropped(tmp_path: Path) -> None:
    conn, _, (e1, e2, _) = seeded(tmp_path)
    assert dropped_pairs(conn, None) == frozenset()

    record_decision(conn, 1, "drop", AT)

    with pytest.raises(ConfigError, match="INFOVORE_PSEUDONYM_SALT"):
        dropped_pairs(conn, None)
    assert dropped_pairs(conn, SALT) == {(e1, ANN), (e2, ANN)}
    record_decision(conn, 1, "keep", AT)
    assert dropped_pairs(conn, SALT) == frozenset()


def test_sql_can_exclude_dropped_claims(tmp_path: Path) -> None:
    conn, _, (e1, _, _) = seeded(tmp_path)
    exclude_dropped(conn, frozenset({(e1, ANN)}))

    row = conn.execute(
        "SELECT COUNT(*) FROM claims_v2 c WHERE NOT is_dropped(c.exchange_id, c.speaker)"
    ).fetchone()

    assert row[0] == 5


def test_groups_are_biggest_first_sampled_and_seeded(tmp_path: Path) -> None:
    conn, run, _ = seeded(tmp_path)
    record_decision(conn, 2, "drop", AT)

    groups = speaker_groups(conn, SALT, run, top=2, per=2, seed=5)

    assert [(g.label, g.total, g.decision) for g in groups] == [
        (ANN, 3, None),
        (BOB, 2, "drop"),
    ]
    assert [len(g.sample) for g in groups] == [2, 2]
    assert groups == speaker_groups(conn, SALT, run, top=2, per=2, seed=5)
    assert len(speaker_groups(conn, SALT, run, top=9, per=1, seed=5)) == 3


def test_redaction_can_skip_dropped_authors_without_renaming_anyone(tmp_path: Path) -> None:
    conn, _, (e1, _, _) = seeded(tmp_path)
    messages = messages_of(conn, e1)

    kept = redact_conversation(messages, SALT, frozenset({1}))

    assert [(ln.ref, ln.speaker) for ln in kept.lines] == [(1, BOB)]
    assert kept.speakers == {ANN, BOB}


def test_exchanges_with_nothing_left_are_dropped_in_order(tmp_path: Path) -> None:
    conn, _, (e1, e2, e3) = seeded(tmp_path)

    assert kept_exchange_ids(conn, [e3, e1, e2], frozenset()) == [e3, e1, e2]
    assert kept_exchange_ids(conn, [e3, e1, e2], frozenset({2})) == [e1, e2]
    assert kept_exchange_ids(conn, [e3, e1, e2], frozenset({1, 3, 2})) == []


def run_cli(tmp_path: Path, *argv: str, salt: str | None = SALT) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        list(argv),
        environ=environment(tmp_path, salt),
        dotenv_path=None,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


def extract_dry(tmp_path: Path) -> tuple[int, str]:
    code, out, _ = run_cli(
        tmp_path,
        *("claims", "extract", "--endpoint", "http://127.0.0.1:9/v1", "--model", "m"),
        *("--ids", "1,2,3", "--limit", "9", "--dry-run"),
    )
    return code, out


def test_extract_leaves_dropped_messages_out_of_the_prompt(tmp_path: Path) -> None:
    conn, _, _ = seeded(tmp_path)
    record_decision(conn, 1, "drop", AT)
    conn.close()

    code, out = extract_dry(tmp_path)

    assert code == ExitCode.OK
    assert "alpha" not in out and "gamma" not in out and "beta octane" in out
    assert "dry-run: 3 requests for 3 conversations, none sent" in out
    assert "skipped" not in out


def test_extract_skips_conversations_with_nothing_left(tmp_path: Path) -> None:
    conn, _, _ = seeded(tmp_path)
    record_decision(conn, 1, "drop", AT)
    record_decision(conn, 3, "drop", AT)
    conn.close()

    code, out = extract_dry(tmp_path)

    assert code == ExitCode.OK
    assert "skipped 1 conversations whose speakers are all dropped" in out
    assert "delta" not in out and "epsilon" in out
    assert "dry-run: 2 requests for 2 conversations, none sent" in out


def test_value_ignores_dropped_claims_and_reports_per_speaker_good_rates(tmp_path: Path) -> None:
    conn, run, _ = seeded(tmp_path)
    for claim_id, verdict in ((1, "good"), (2, "wrong"), (3, "good"), (6, "good")):
        record_review(conn, claim_id, verdict, AT, "conversation")
    record_review(conn, 5, "good", AT, "cited-only")
    record_decision(conn, 2, "drop", AT)
    conn.close()

    code, out, _ = run_cli(tmp_path, "claims", "value", "--runs", str(run))

    assert code == ExitCode.OK
    assert "claims 5 (1.67 per conversation)" in out
    assert "dropped speakers: 1 (2 claims ignored)" in out
    assert f"  {ANN}: good 1 of 2 (50.0%), claims 3" in out
    assert BOB not in out.split("per speaker")[1]
    assert "good 2 of 3 (66.7%)" in out


def test_value_without_a_salt_skips_the_per_speaker_section(tmp_path: Path) -> None:
    _, run, _ = seeded(tmp_path)

    code, out, _ = run_cli(tmp_path, "claims", "value", "--runs", str(run), salt=None)

    assert code == ExitCode.OK and "per speaker: needs INFOVORE_PSEUDONYM_SALT" in out


def test_value_refuses_when_speakers_are_dropped_and_there_is_no_salt(tmp_path: Path) -> None:
    conn, run, _ = seeded(tmp_path)
    record_decision(conn, 2, "drop", AT)
    conn.close()

    code, _, err = run_cli(tmp_path, "claims", "value", "--runs", str(run), salt=None)

    assert code == ExitCode.CONFIG and "INFOVORE_PSEUDONYM_SALT" in err


def test_wiki_ignores_dropped_speakers(tmp_path: Path) -> None:
    conn, _, _ = seeded(tmp_path)
    record_checks(conn, [CheckRow(i, "supported", 1.0, []) for i in range(1, 8)], {}, AT)
    before, _ = load_claims(conn, salt=SALT)
    assert len(before) == 7
    record_decision(conn, 1, "drop", AT)

    claims, excluded = load_claims(conn, salt=SALT)

    assert len(claims) == 4 and excluded.total == 3
    assert all(c.speaker != ANN for c in claims)
    with pytest.raises(ConfigError):
        load_claims(conn, salt=None)


def test_wiki_command_uses_the_configured_salt(tmp_path: Path) -> None:
    conn, _, _ = seeded(tmp_path)
    record_decision(conn, 1, "drop", AT)
    conn.close()

    code, out, _ = run_cli(tmp_path, "wiki", "stats")
    assert code == ExitCode.OK and "excluded: " in out
    code, _, err = run_cli(tmp_path, "wiki", "stats", salt=None)
    assert code == ExitCode.CONFIG and "INFOVORE_PSEUDONYM_SALT" in err


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[str, sqlite3.Connection]]:
    conn, run, _ = seeded(tmp_path)
    groups = speaker_groups(conn, SALT, run, top=3, per=2, seed=1)
    servers = start_all(["127.0.0.1"], 0, conn, SystemClock(), run, groups)
    yield listening_url(servers[0]), conn
    shutdown_all(servers)


def test_the_page_lists_speakers_without_author_ids(
    served: tuple[str, sqlite3.Connection],
) -> None:
    url, _ = served

    code, page = fetch(url)
    code_api, body = fetch(url + "api/speakers")
    data = json.loads(body)

    assert code == 200 and b"drop" in page and code_api == 200
    assert [s["label"] for s in data["speakers"]] == [ANN, BOB, CAT]
    assert data["speakers"][0]["total"] == 3 and len(data["speakers"][0]["sample"]) == 2
    assert data["speakers"][0]["decision"] is None and data["run"] == 1
    assert "author" not in body.decode()
    assert b'name="viewport"' in page and b'id="keepall"' in page


def test_decisions_are_appended_by_rank(served: tuple[str, sqlite3.Connection]) -> None:
    url, conn = served

    code, _ = fetch(url + "api/speaker", {"rank": 1, "decision": "drop"})
    assert code == 200
    fetch(url + "api/speaker", {"rank": 1, "decision": "keep"})
    fetch(url + "api/speaker", {"rank": 0, "decision": "drop"})

    assert decisions(conn) == {1: "drop", 2: "keep"}
    shown = json.loads(fetch(url + "api/speakers")[1])["speakers"]
    assert [s["decision"] for s in shown] == ["drop", "keep", None]
    assert dropped_authors(conn) == frozenset({1})


@pytest.mark.parametrize(
    "body",
    [
        {"rank": 0, "decision": "maybe"},
        {"rank": 9, "decision": "drop"},
        {"rank": -1, "decision": "drop"},
        {"decision": "drop"},
        [],
    ],
)
def test_bad_decisions_are_refused(served: tuple[str, sqlite3.Connection], body: Any) -> None:
    url, conn = served

    code, _ = fetch(url + "api/speaker", body)

    assert code == 400 and decisions(conn) == {}


def test_unknown_paths_404(served: tuple[str, sqlite3.Connection]) -> None:
    url, _ = served

    assert fetch(url + "nope")[0] == 404
    assert fetch(url + "nope", {})[0] == 404


def test_speakers_command_listens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, run, _ = seeded(tmp_path)
    seen: list[str] = []
    monkeypatch.setattr("infovore.sift.httpd.block_until_interrupted", lambda e: seen.append("b"))

    code, out, _ = run_cli(
        tmp_path, "claims", "speakers", "--run", str(run), "--top", "2", "--per", "1", "--port", "0"
    )

    assert code == ExitCode.OK and seen == ["b"]
    assert "2 of 3 speakers, 0 dropped" in out
    assert "listening on http://127.0.0.1:" in out


@pytest.mark.parametrize(
    ("extra", "salt", "message"),
    [
        (["--run", "7"], SALT, "unknown run 7"),
        (["--run", "1"], None, "INFOVORE_PSEUDONYM_SALT"),
        (["--run", "1", "--top", "0"], SALT, "--top must be positive"),
        (["--run", "1", "--per", "0"], SALT, "--per must be positive"),
    ],
)
def test_speakers_command_refuses_bad_input(
    tmp_path: Path, extra: list[str], salt: str | None, message: str
) -> None:
    seeded(tmp_path)

    code, _, err = run_cli(tmp_path, "claims", "speakers", *extra, salt=salt)

    assert code == ExitCode.CONFIG and message in err


def claim_count(out: str) -> int:
    match = re.search(r"conversations \d+ \(failed \d+\), claims (\d+)", out)
    assert match
    return int(match.group(1))


def test_wiki_applies_the_claim_gate_and_the_drop_list_together(tmp_path: Path) -> None:
    conn, _, _ = seeded(tmp_path)
    record_checks(conn, [CheckRow(i, "supported", 1.0, []) for i in range(1, 8)], {}, AT)
    everything = {c.claim_id for c in load_claims(conn)[0]}
    gated = {c.claim_id for c in load_claims(conn, tech_only=True)[0]}
    record_decision(conn, 1, "drop", AT)
    dropped = {c.claim_id for c in load_claims(conn, salt=SALT)[0]}

    both = {c.claim_id for c in load_claims(conn, tech_only=True, salt=SALT)[0]}

    assert gated < everything and dropped < everything
    assert both == gated & dropped and both < gated and both < dropped


def test_value_applies_the_claim_gate_and_the_drop_list_together(tmp_path: Path) -> None:
    conn, run, _ = seeded(tmp_path)
    conn.close()
    base = ("claims", "value", "--runs", str(run))
    everything = claim_count(run_cli(tmp_path, *base)[1])
    gated = claim_count(run_cli(tmp_path, *base, "--claim-gate")[1])
    conn = db(tmp_path)
    record_decision(conn, 1, "drop", AT)
    conn.close()
    dropped = claim_count(run_cli(tmp_path, *base)[1])

    both = claim_count(run_cli(tmp_path, *base, "--claim-gate")[1])

    assert gated < everything and dropped < everything and both < gated and both < dropped
