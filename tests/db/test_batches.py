import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.batches import record_extraction_batch
from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database
from infovore.rows import RunMode

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def test_record_extraction_batch_writes_a_full_trial_row(tmp_path: Path) -> None:
    conn = db(tmp_path)

    record_extraction_batch(conn, "b1", RunMode.TRIAL, "uncertain", 5, 20, NOW)

    row = conn.execute("SELECT * FROM extraction_batches WHERE batch_id = 'b1'").fetchone()
    assert row["mode"] == "trial"
    assert row["strategy"] == "uncertain"
    assert row["seed"] == 5
    assert row["sample"] == 20
    assert row["created_at"] == to_db_time(NOW)


def test_record_extraction_batch_allows_null_strategy_and_sample_for_live_mode(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)

    record_extraction_batch(conn, "b2", RunMode.LIVE, None, 0, None, NOW)

    row = conn.execute("SELECT * FROM extraction_batches WHERE batch_id = 'b2'").fetchone()
    assert row["mode"] == "live"
    assert row["strategy"] is None
    assert row["sample"] is None


def test_record_extraction_batch_is_idempotent_for_the_same_batch_id(tmp_path: Path) -> None:
    conn = db(tmp_path)

    record_extraction_batch(conn, "b1", RunMode.TRIAL, "random", 1, 10, NOW)
    record_extraction_batch(conn, "b1", RunMode.TRIAL, "random", 1, 10, NOW)

    count = conn.execute("SELECT COUNT(*) AS n FROM extraction_batches").fetchone()["n"]
    assert count == 1
