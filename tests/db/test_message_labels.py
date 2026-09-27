import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database
from infovore.db.message_labels import (
    MessageLabelCounts,
    effective_message_labels,
    human_labeled_message_ids,
    message_label_counts,
    set_message_label,
)
from infovore.rows import MessageLabel, MessageLabelSource

NOW = datetime(2026, 1, 1, tzinfo=UTC)
NOW_TEXT = to_db_time(NOW)
LATER = datetime(2026, 1, 2, tzinfo=UTC)


def seeded(tmp_path: Path, message_ids: tuple[int, ...] = (1,)) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    for message_id in message_ids:
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 1, 1, 'a', ?, 'x', ?, '{}')",
            (message_id, NOW_TEXT, NOW_TEXT),
        )
    return conn


def test_set_message_label_records_a_new_label(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    set_message_label(conn, 1, MessageLabel.TRASH, MessageLabelSource.HUMAN, "sift:x:a", NOW)
    row = conn.execute("SELECT * FROM message_labels WHERE message_id = 1").fetchone()
    assert row["label"] == "trash"
    assert row["source"] == "human"
    assert row["source_ref"] == "sift:x:a"
    assert row["labeled_at"] == NOW_TEXT


def test_set_message_label_is_idempotent_and_replaces_the_same_source(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    set_message_label(conn, 1, MessageLabel.TRASH, MessageLabelSource.HUMAN, "a", NOW)
    set_message_label(conn, 1, MessageLabel.KEEP, MessageLabelSource.HUMAN, "b", LATER)
    rows = conn.execute("SELECT * FROM message_labels WHERE message_id = 1").fetchall()
    assert len(rows) == 1
    assert rows[0]["label"] == "keep"
    assert rows[0]["source_ref"] == "b"
    assert rows[0]["labeled_at"] == to_db_time(LATER)


def test_set_message_label_keeps_separate_rows_per_source(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    set_message_label(conn, 1, MessageLabel.TRASH, MessageLabelSource.RULE, None, NOW)
    set_message_label(conn, 1, MessageLabel.KEEP, MessageLabelSource.HUMAN, None, LATER)
    rows = conn.execute(
        "SELECT source, label FROM message_labels WHERE message_id = 1 ORDER BY source"
    ).fetchall()
    assert [(r["source"], r["label"]) for r in rows] == [("human", "keep"), ("rule", "trash")]


def test_set_message_label_rejects_unknown_message(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        set_message_label(conn, 999, MessageLabel.TRASH, MessageLabelSource.HUMAN, None, NOW)


def test_effective_message_labels_is_empty_on_a_fresh_database(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    assert effective_message_labels(conn) == {}


def test_effective_message_labels_uses_weak_source_when_no_human_label_exists(
    tmp_path: Path,
) -> None:
    conn = seeded(tmp_path)
    set_message_label(conn, 1, MessageLabel.TRASH, MessageLabelSource.CITATION, None, NOW)
    assert effective_message_labels(conn) == {1: MessageLabel.TRASH}


def test_effective_message_labels_prefers_human_over_citation_or_rule(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    set_message_label(conn, 1, MessageLabel.TRASH, MessageLabelSource.RULE, None, NOW)
    set_message_label(conn, 1, MessageLabel.KEEP, MessageLabelSource.HUMAN, None, LATER)
    assert effective_message_labels(conn) == {1: MessageLabel.KEEP}


def test_effective_message_labels_prefers_human_regardless_of_insert_order(
    tmp_path: Path,
) -> None:
    conn = seeded(tmp_path)
    set_message_label(conn, 1, MessageLabel.KEEP, MessageLabelSource.HUMAN, None, NOW)
    set_message_label(conn, 1, MessageLabel.TRASH, MessageLabelSource.CITATION, None, LATER)
    assert effective_message_labels(conn) == {1: MessageLabel.KEEP}


def test_effective_message_labels_covers_every_labeled_message(tmp_path: Path) -> None:
    conn = seeded(tmp_path, message_ids=(1, 2, 3))
    set_message_label(conn, 1, MessageLabel.TRASH, MessageLabelSource.CITATION, None, NOW)
    set_message_label(conn, 2, MessageLabel.KEEP, MessageLabelSource.HUMAN, None, NOW)
    assert effective_message_labels(conn) == {1: MessageLabel.TRASH, 2: MessageLabel.KEEP}


def test_human_labeled_message_ids_is_empty_on_a_fresh_database(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    assert human_labeled_message_ids(conn) == frozenset()


def test_human_labeled_message_ids_only_includes_human_source(tmp_path: Path) -> None:
    conn = seeded(tmp_path, message_ids=(1, 2))
    set_message_label(conn, 1, MessageLabel.TRASH, MessageLabelSource.HUMAN, None, NOW)
    set_message_label(conn, 2, MessageLabel.KEEP, MessageLabelSource.RULE, None, NOW)
    assert human_labeled_message_ids(conn) == frozenset({1})


def test_message_label_counts_on_a_fresh_database(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    assert message_label_counts(conn) == MessageLabelCounts(by_source={}, effective={})


def test_message_label_counts_by_source_and_effective(tmp_path: Path) -> None:
    conn = seeded(tmp_path, message_ids=(1, 2, 3))
    set_message_label(conn, 1, MessageLabel.TRASH, MessageLabelSource.CITATION, None, NOW)
    set_message_label(conn, 2, MessageLabel.KEEP, MessageLabelSource.CITATION, None, NOW)
    set_message_label(conn, 2, MessageLabel.TRASH, MessageLabelSource.HUMAN, None, LATER)
    set_message_label(conn, 3, MessageLabel.KEEP, MessageLabelSource.HUMAN, None, LATER)
    counts = message_label_counts(conn)
    assert counts.by_source == {
        "citation": {"trash": 1, "keep": 1},
        "human": {"trash": 1, "keep": 1},
    }
    assert counts.effective == {"keep": 1, "trash": 2}
