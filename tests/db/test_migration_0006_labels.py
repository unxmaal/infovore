import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database

NOW = to_db_time(datetime(2026, 1, 1, tzinfo=UTC))


def seeded(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json) VALUES (1, 1, 1, 1, 'a', ?, 'x', ?, '{}')",
        (NOW, NOW),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (1, 1, 1, 1, ?, ?, 1, 'quiet_gap', 'h')",
        (NOW, NOW),
    )
    return conn


def test_exchange_labels_table_exists(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "exchange_labels" in names


def test_one_label_per_exchange_per_source(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    conn.execute(
        "INSERT INTO exchange_labels (exchange_id, label, source, labeled_at)"
        " VALUES (1, 'lore', 'llm', ?)",
        (NOW,),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO exchange_labels (exchange_id, label, source, labeled_at)"
            " VALUES (1, 'noise', 'llm', ?)",
            (NOW,),
        )


def test_label_is_constrained_to_lore_or_noise(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO exchange_labels (exchange_id, label, source, labeled_at)"
            " VALUES (1, 'bogus', 'llm', ?)",
            (NOW,),
        )


def test_source_is_constrained_to_llm_or_human(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO exchange_labels (exchange_id, label, source, labeled_at)"
            " VALUES (1, 'lore', 'robot', ?)",
            (NOW,),
        )


def test_exchange_id_is_a_foreign_key(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO exchange_labels (exchange_id, label, source, labeled_at)"
            " VALUES (999, 'lore', 'llm', ?)",
            (NOW,),
        )


def test_source_ref_is_optional(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    conn.execute(
        "INSERT INTO exchange_labels (exchange_id, label, source, source_ref, labeled_at)"
        " VALUES (1, 'lore', 'human', NULL, ?)",
        (NOW,),
    )
    row = conn.execute("SELECT source_ref FROM exchange_labels WHERE source = 'human'").fetchone()
    assert row["source_ref"] is None
