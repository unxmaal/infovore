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
    return conn


def test_message_labels_table_exists(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "message_labels" in names


def test_one_label_per_message_per_source(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    conn.execute(
        "INSERT INTO message_labels (message_id, label, source, labeled_at)"
        " VALUES (1, 'trash', 'human', ?)",
        (NOW,),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO message_labels (message_id, label, source, labeled_at)"
            " VALUES (1, 'keep', 'human', ?)",
            (NOW,),
        )


def test_label_is_constrained_to_trash_or_keep(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO message_labels (message_id, label, source, labeled_at)"
            " VALUES (1, 'bogus', 'human', ?)",
            (NOW,),
        )


def test_source_is_constrained_to_human_citation_or_rule(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO message_labels (message_id, label, source, labeled_at)"
            " VALUES (1, 'trash', 'robot', ?)",
            (NOW,),
        )


def test_message_id_is_a_foreign_key(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO message_labels (message_id, label, source, labeled_at)"
            " VALUES (999, 'trash', 'human', ?)",
            (NOW,),
        )


def test_source_ref_is_optional(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    conn.execute(
        "INSERT INTO message_labels (message_id, label, source, source_ref, labeled_at)"
        " VALUES (1, 'trash', 'human', NULL, ?)",
        (NOW,),
    )
    row = conn.execute("SELECT source_ref FROM message_labels WHERE source = 'human'").fetchone()
    assert row["source_ref"] is None


def test_messages_carry_p_trash_columns(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
    assert {"p_trash", "p_trash_model"} <= columns
