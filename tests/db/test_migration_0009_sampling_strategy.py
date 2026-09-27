import sqlite3
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database


def test_extraction_batches_table_has_the_expected_columns(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(extraction_batches)")}
    assert columns == {"batch_id", "mode", "strategy", "seed", "sample", "created_at"}


def test_extraction_batches_batch_id_is_the_primary_key(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO extraction_batches (batch_id, mode, strategy, seed, sample, created_at)"
        " VALUES ('b1', 'trial', 'stratified', 0, 10, 'now')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO extraction_batches (batch_id, mode, strategy, seed, sample, created_at)"
            " VALUES ('b1', 'live', NULL, 0, NULL, 'now')"
        )


def test_extraction_batches_strategy_and_sample_may_be_null_for_live_mode(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO extraction_batches (batch_id, mode, strategy, seed, sample, created_at)"
        " VALUES ('b2', 'live', NULL, 0, NULL, 'now')"
    )
    row = conn.execute(
        "SELECT strategy, sample FROM extraction_batches WHERE batch_id = 'b2'"
    ).fetchone()
    assert row["strategy"] is None
    assert row["sample"] is None


def test_extraction_batches_rejects_an_unknown_mode(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO extraction_batches (batch_id, mode, strategy, seed, sample, created_at)"
            " VALUES ('b3', 'bogus', NULL, 0, NULL, 'now')"
        )


def test_extraction_batches_rejects_an_unknown_strategy(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO extraction_batches (batch_id, mode, strategy, seed, sample, created_at)"
            " VALUES ('b4', 'trial', 'bogus', 0, 10, 'now')"
        )


def test_extraction_runs_gains_a_nullable_sampled_by_column(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(extraction_runs)")}
    assert "sampled_by" in columns


def test_existing_extraction_runs_rows_stay_null_after_the_migration(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO prompt_versions (version, text_sha256, created_at) VALUES ('v1', 'sha', 'now')"
    )
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (1, 1, 9, 1, 'a', 'now', 'x', 'now', '{}')"
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (1, 1, 1, 1, 'now', 'now', 1, 'quiet_gap', 'h1')"
    )
    conn.execute(
        "INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at, mode,"
        " outcome) VALUES (1, 'm', 'v1', 'now', 'trial', 'ok')"
    )
    row = conn.execute("SELECT sampled_by FROM extraction_runs WHERE exchange_id = 1").fetchone()
    assert row["sampled_by"] is None
