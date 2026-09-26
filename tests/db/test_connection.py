import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from infovore.db.codec import from_db_time, to_db_time
from infovore.db.connection import (
    Migration,
    MigrationError,
    applied_versions,
    load_migrations,
    migrate,
    open_database,
    transaction,
)

EXPECTED_TABLES = {
    "schema_migrations",
    "channels",
    "messages",
    "message_revisions",
    "attachments",
    "reactions",
    "opt_outs",
    "exchanges",
    "exchange_messages",
    "prompt_versions",
    "extraction_runs",
    "claims",
    "claim_sources",
    "claims_fts",
}


def tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def test_open_database_uses_wal_and_foreign_keys(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.row_factory is sqlite3.Row


def test_fresh_database_migrates_to_latest(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    newly_applied = migrate(conn)
    latest = max(m.version for m in load_migrations())
    assert newly_applied == [m.version for m in load_migrations()]
    assert applied_versions(conn) == newly_applied
    assert conn.execute("PRAGMA user_version").fetchone()[0] == latest
    assert EXPECTED_TABLES <= tables(conn)


def test_remigrating_is_a_noop(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    before = applied_versions(conn)
    assert migrate(conn) == []
    assert applied_versions(conn) == before


def write(directory: Path, name: str, sql: str) -> None:
    (directory / name).write_text(sql)


def test_failed_migration_rolls_back_and_leaves_version_untouched(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    write(migrations, "0001_good.sql", "CREATE TABLE good (id INTEGER PRIMARY KEY);")
    write(
        migrations,
        "0002_bad.sql",
        "CREATE TABLE partial (id INTEGER PRIMARY KEY);\nTHIS IS NOT SQL;",
    )
    conn = open_database(tmp_path / "x.db")
    with pytest.raises(MigrationError, match="0002_bad"):
        migrate(conn, load_migrations(migrations))
    assert applied_versions(conn) == [1]
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "good" in tables(conn)
    assert "partial" not in tables(conn)


def test_load_migrations_orders_numerically_and_ignores_other_files(tmp_path: Path) -> None:
    write(tmp_path, "0010_ten.sql", "SELECT 1;")
    write(tmp_path, "0002_two.sql", "SELECT 1;")
    write(tmp_path, "notes.txt", "ignored")
    write(tmp_path, "draft.sql", "ignored: no numeric prefix")
    assert load_migrations(tmp_path) == [
        Migration(2, "0002_two", "SELECT 1;"),
        Migration(10, "0010_ten", "SELECT 1;"),
    ]


def test_duplicate_migration_versions_are_rejected(tmp_path: Path) -> None:
    write(tmp_path, "0001_a.sql", "SELECT 1;")
    write(tmp_path, "0001_b.sql", "SELECT 1;")
    with pytest.raises(MigrationError, match="duplicate"):
        load_migrations(tmp_path)


def test_transaction_commits_on_success_and_rolls_back_on_error(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    conn.execute("CREATE TABLE t (v INTEGER)")
    with transaction(conn):
        conn.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(RuntimeError), transaction(conn):
        conn.execute("INSERT INTO t VALUES (2)")
        raise RuntimeError("boom")
    assert [row[0] for row in conn.execute("SELECT v FROM t")] == [1]


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO reactions (message_id, emoji, count) VALUES (999, 'x', 1)"
        )


def test_a_message_belongs_to_at_most_one_exchange(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    now = to_db_time(datetime(2026, 1, 1, tzinfo=UTC))
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json) VALUES (1, 1, 1, 1, 'a', ?, 'x', ?, '{}')",
        (now, now),
    )
    for content_hash in ("h1", "h2"):
        conn.execute(
            "INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,"
            " ended_at, message_count, grouping_rule, content_hash)"
            " VALUES (1, 1, 1, ?, ?, 1, 'quiet_gap', ?)",
            (now, now, content_hash),
        )
    conn.execute("INSERT INTO exchange_messages VALUES (1, 1, 0)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO exchange_messages VALUES (2, 1, 0)")


def test_claims_fts_tracks_inserts_updates_and_deletes(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    now = to_db_time(datetime(2026, 1, 1, tzinfo=UTC))
    conn.executescript(
        f"""
        INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,
          created_at, content, ingested_at, raw_json) VALUES (1, 1, 1, 1, 'a', '{now}', 'x', '{now}', '{{}}');
        INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,
          ended_at, message_count, grouping_rule, content_hash)
          VALUES (1, 1, 1, '{now}', '{now}', 1, 'quiet_gap', 'h');
        INSERT INTO prompt_versions VALUES ('v1', 'sha', '{now}', NULL);
        INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at, mode, outcome)
          VALUES (1, 'm', 'v1', '{now}', 'trial', 'ok');
        INSERT INTO claims (exchange_id, extraction_run_id, statement, subject, kind, confidence,
          probe_question, permalink) VALUES (1, 1, 'Octane2 needs PROM 6.5', 'IP30', 'fact', 0.9,
          'q?', 'https://discord.com/channels/1/1/1');
        """
    )

    def match(term: str) -> list[int]:
        return [r[0] for r in conn.execute("SELECT rowid FROM claims_fts WHERE claims_fts MATCH ?", (term,))]

    assert match("IP30") == [1]
    assert match('"6.5"') == [1]
    conn.execute("UPDATE claims SET subject = 'Octane2' WHERE id = 1")
    assert match("IP30") == []
    assert match("Octane2") == [1]
    conn.execute("DELETE FROM claims WHERE id = 1")
    assert match("Octane2") == []


def test_time_codec_round_trips_and_requires_timezone() -> None:
    moment = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)
    assert to_db_time(moment) == "2026-01-01T12:30:00+00:00"
    assert from_db_time(to_db_time(moment)) == moment
    shifted = datetime(2026, 1, 1, 7, 30, tzinfo=timezone(timedelta(hours=-5)))
    assert to_db_time(shifted) == "2026-01-01T12:30:00+00:00"
    assert to_db_time(None) is None
    assert from_db_time(None) is None
    with pytest.raises(ValueError, match="timezone"):
        to_db_time(datetime(2026, 1, 1))
