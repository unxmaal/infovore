import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.claims_v2 import (
    ClaimIn,
    ExchangeOutcome,
    Rejection,
    create_run,
    latest_run_id,
    record_exchange,
    record_review,
    report_rows,
    review_rows,
    run_ids,
)
from infovore.db.connection import load_migrations, migrate, open_database
from tests.claims.seed import conversation, db

AT = datetime(2026, 2, 1, tzinfo=UTC)


def make_run(conn: sqlite3.Connection) -> int:
    return create_run(
        conn,
        endpoint="http://x/v1",
        model_alias="eval-4b",
        model_id="real/4b",
        model_id_source="model_info",
        prompt_hash="abc",
        selection="gold",
        recipe={"k": 1},
        now=AT,
    )


def outcome(**kw: object) -> ExchangeOutcome:
    base: dict[str, object] = {
        "outcome": "ok",
        "error": None,
        "windows": 1,
        "input_tokens": 100,
        "output_tokens": 10,
        "seconds": 2.0,
    }
    return ExchangeOutcome(**{**base, **kw})  # type: ignore[arg-type]


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return db(tmp_path)


def test_a_run_and_its_claims_round_trip(conn: sqlite3.Connection) -> None:
    eid, ids = conversation(conn, [(1, "ann", "hi"), (2, "bob", "yo")], 1)
    run = make_run(conn)

    record_exchange(
        conn,
        run,
        eid,
        outcome(),
        [ClaimIn("user-aaaa", "user-aaaa said hi", (ids[0], ids[1]))],
        [Rejection("user-bbbb", "bad", "[9]", "unknown ref 9")],
    )

    rows = review_rows(conn, run)
    assert [(r.statement, r.sources, r.verdict) for r in rows] == [
        ("user-aaaa said hi", [(ids[0], "ann", "hi"), (ids[1], "bob", "yo")], None)
    ]
    assert conn.execute("SELECT reason FROM claim_rejections").fetchone()[0] == "unknown ref 9"
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
    model = conn.execute("SELECT model_id, model_id_source FROM claim_runs").fetchone()
    assert tuple(model) == ("real/4b", "model_info")


def test_a_failed_exchange_stores_no_claims_but_is_recorded(conn: sqlite3.Connection) -> None:
    eid, _ = conversation(conn, [(1, "ann", "hi")], 1)
    run = make_run(conn)

    record_exchange(conn, run, eid, outcome(outcome="failed", error="boom"), [], [])

    assert review_rows(conn, run) == []
    row = conn.execute("SELECT outcome, error FROM claim_run_exchanges").fetchone()
    assert tuple(row) == ("failed", "boom")


def test_reviews_are_append_only_and_the_latest_wins(conn: sqlite3.Connection) -> None:
    eid, ids = conversation(conn, [(1, "ann", "hi")], 1)
    run = make_run(conn)
    record_exchange(conn, run, eid, outcome(), [ClaimIn("u", "u said x", (ids[0],))], [])
    claim = review_rows(conn, run)[0].claim_id

    record_review(conn, claim, "wrong", AT)
    record_review(conn, claim, "good", AT)

    assert review_rows(conn, run)[0].verdict == "good"
    assert conn.execute("SELECT COUNT(*) FROM claim_reviews").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError):
        record_review(conn, claim, "bogus", AT)


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE claim_reviews SET verdict = 'good'",
        "DELETE FROM claim_reviews",
        "UPDATE claims_v2 SET statement = 'x'",
        "DELETE FROM claims_v2",
        "UPDATE claim_runs SET model_id = 'x'",
        "DELETE FROM claim_runs",
        "UPDATE claim_run_exchanges SET seconds = 0",
        "DELETE FROM claim_run_exchanges",
    ],
)
def test_every_trial_table_refuses_edits(conn: sqlite3.Connection, sql: str) -> None:
    eid, ids = conversation(conn, [(1, "ann", "hi")], 1)
    run = make_run(conn)
    record_exchange(conn, run, eid, outcome(), [ClaimIn("u", "u said x", (ids[0],))], [])
    record_review(conn, review_rows(conn, run)[0].claim_id, "good", AT)

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(sql)


def test_run_listing_and_latest(conn: sqlite3.Connection) -> None:
    assert latest_run_id(conn) is None
    first, second = make_run(conn), make_run(conn)

    assert run_ids(conn) == [first, second]
    assert latest_run_id(conn) == second


def test_report_counts_everything_per_run(conn: sqlite3.Connection) -> None:
    run = make_run(conn)
    e1, i1 = conversation(conn, [(1, "ann", "a")], 1)
    e2, _ = conversation(conn, [(1, "ann", "b")], 2)
    e3, _ = conversation(conn, [(1, "ann", "c")], 3)
    record_exchange(
        conn,
        run,
        e1,
        outcome(),
        [ClaimIn("u", "u said 1", (i1[0],)), ClaimIn("u", "u said 2", (i1[0],))],
        [Rejection("u", "s", "[]", "no refs")],
    )
    record_exchange(conn, run, e2, outcome(input_tokens=50, output_tokens=5, seconds=4.0), [], [])
    record_exchange(conn, run, e3, outcome(outcome="failed", error="x"), [], [])
    claims = [r.claim_id for r in review_rows(conn, run)]
    record_review(conn, claims[0], "made_up", AT)
    record_review(conn, claims[0], "wrong", AT)
    record_review(conn, claims[1], "good", AT)

    (report,) = report_rows(conn, run)

    assert report.run_id == run
    assert (report.model_id, report.model_id_source) == ("real/4b", "model_info")
    assert (report.conversations, report.failed, report.claims, report.rejected) == (2, 1, 2, 1)
    assert report.zero_claim_conversations == 1
    assert report.verdicts == {"good": 1, "wrong": 1, "made_up": 0, "not_useful": 0}
    assert report.unreviewed == 0
    assert (report.input_tokens, report.output_tokens) == (250, 25)
    assert report.seconds == 8.0
    assert report_rows(conn, None)[0].run_id == run
    assert report_rows(conn, run + 5) == []


def test_reviews_record_their_interface_and_default_to_conversation(
    conn: sqlite3.Connection,
) -> None:
    eid, ids = conversation(conn, [(1, "ann", "hi")], 1)
    run = make_run(conn)
    record_exchange(conn, run, eid, outcome(), [ClaimIn("u", "u said", (ids[0],))], [])
    claim = review_rows(conn, run)[0].claim_id

    record_review(conn, claim, "good", AT)
    record_review(conn, claim, "wrong", AT, "cited-only")

    rows = conn.execute("SELECT interface FROM claim_reviews ORDER BY id").fetchall()
    assert [r[0] for r in rows] == ["conversation", "cited-only"]
    assert conn.execute("SELECT interface FROM current_claim_reviews").fetchone()[0] == "cited-only"
    with pytest.raises(sqlite3.IntegrityError):
        record_review(conn, claim, "good", AT, "bogus")


def test_the_report_splits_current_reviews_by_interface(conn: sqlite3.Connection) -> None:
    run = make_run(conn)
    e1, i1 = conversation(conn, [(1, "ann", "a")], 1)
    claims = [ClaimIn("u", f"u said {n}", (i1[0],)) for n in range(3)]
    record_exchange(conn, run, e1, outcome(), claims, [])
    ids = [r.claim_id for r in review_rows(conn, run)]
    record_review(conn, ids[0], "good", AT, "cited-only")
    record_review(conn, ids[0], "made_up", AT, "conversation")
    record_review(conn, ids[1], "good", AT, "cited-only")

    (report,) = report_rows(conn, run)

    assert report.interfaces["cited-only"] == {"good": 1, "wrong": 0, "made_up": 0, "not_useful": 0}
    assert report.interfaces["conversation"]["made_up"] == 1
    assert report.verdicts["good"] == 1 and report.verdicts["made_up"] == 1


def test_the_migration_backfills_existing_reviews_as_cited_only(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    older = [m for m in load_migrations() if m.version < 29]
    migrate(conn, older)
    conn.execute("INSERT INTO claim_runs VALUES (1, 't', 'e', 'a', 'm', 'alias', 'h', 's', '{}')")
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json) VALUES (1, 1, 9, 1, 'a', 'n', 'x', 'n', '{}')"
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (1, 1, 1, 1, 'n', 'n', 1, 'quiet_gap', 'h1')"
    )
    conn.execute(
        "INSERT INTO claims_v2 (id, run_id, exchange_id, speaker, statement)"
        " VALUES (1, 1, 1, 'u', 's')"
    )
    conn.execute(
        "INSERT INTO claim_reviews (claim_id, verdict, reviewed_at) VALUES (1, 'good', 't')"
    )

    migrate(conn)
    record_review(conn, 1, "wrong", AT)

    rows = conn.execute("SELECT verdict, interface FROM claim_reviews ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [("good", "cited-only"), ("wrong", "conversation")]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE claim_reviews SET interface = 'conversation'")
