from pathlib import Path

from infovore.db.connection import migrate, open_database


def test_extraction_runs_gains_a_nullable_batch_id_column(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(extraction_runs)")}
    assert "batch_id" in columns


def test_existing_rows_stay_null_after_the_migration(tmp_path: Path) -> None:
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
    row = conn.execute("SELECT batch_id FROM extraction_runs WHERE exchange_id = 1").fetchone()
    assert row["batch_id"] is None


def test_batch_id_can_be_set_and_indexed(tmp_path: Path) -> None:
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
        " outcome, batch_id) VALUES (1, 'm', 'v1', 'now', 'trial', 'ok', 'batch-1')"
    )
    row = conn.execute("SELECT batch_id FROM extraction_runs WHERE exchange_id = 1").fetchone()
    assert row["batch_id"] == "batch-1"
