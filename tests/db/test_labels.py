import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database
from infovore.db.labels import LabelCounts, effective_labels, label_counts, set_label
from infovore.rows import Label, LabelSource

NOW = datetime(2026, 1, 1, tzinfo=UTC)
NOW_TEXT = to_db_time(NOW)
LATER = datetime(2026, 1, 2, tzinfo=UTC)


def seeded(tmp_path: Path, exchange_ids: tuple[int, ...] = (1,)) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    for exchange_id in exchange_ids:
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 1, 1, 'a', ?, 'x', ?, '{}')",
            (exchange_id, NOW_TEXT, NOW_TEXT),
        )
        conn.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
            " started_at, ended_at, message_count, grouping_rule, content_hash)"
            " VALUES (?, 1, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
            (exchange_id, exchange_id, exchange_id, NOW_TEXT, NOW_TEXT, f"h{exchange_id}"),
        )
    return conn


def test_set_label_records_a_new_label(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    set_label(conn, 1, Label.LORE, LabelSource.LLM, "run:1 model:m", NOW)
    row = conn.execute("SELECT * FROM exchange_labels WHERE exchange_id = 1").fetchone()
    assert row["label"] == "lore"
    assert row["source"] == "llm"
    assert row["source_ref"] == "run:1 model:m"
    assert row["labeled_at"] == NOW_TEXT


def test_set_label_is_idempotent_and_replaces_the_same_source(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    set_label(conn, 1, Label.LORE, LabelSource.LLM, "run:1 model:m", NOW)
    set_label(conn, 1, Label.NOISE, LabelSource.LLM, "run:2 model:m", LATER)
    rows = conn.execute("SELECT * FROM exchange_labels WHERE exchange_id = 1").fetchall()
    assert len(rows) == 1
    assert rows[0]["label"] == "noise"
    assert rows[0]["source_ref"] == "run:2 model:m"
    assert rows[0]["labeled_at"] == to_db_time(LATER)


def test_set_label_keeps_separate_rows_per_source(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    set_label(conn, 1, Label.NOISE, LabelSource.LLM, None, NOW)
    set_label(conn, 1, Label.LORE, LabelSource.HUMAN, None, LATER)
    rows = conn.execute(
        "SELECT source, label FROM exchange_labels WHERE exchange_id = 1 ORDER BY source"
    ).fetchall()
    assert [(r["source"], r["label"]) for r in rows] == [("human", "lore"), ("llm", "noise")]


def test_set_label_rejects_unknown_exchange(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        set_label(conn, 999, Label.LORE, LabelSource.HUMAN, None, NOW)


def test_effective_labels_is_empty_on_a_fresh_database(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    assert effective_labels(conn) == {}


def test_effective_labels_uses_llm_when_no_human_label_exists(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    set_label(conn, 1, Label.LORE, LabelSource.LLM, None, NOW)
    assert effective_labels(conn) == {1: Label.LORE}


def test_effective_labels_prefers_human_over_llm(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    set_label(conn, 1, Label.NOISE, LabelSource.LLM, None, NOW)
    set_label(conn, 1, Label.LORE, LabelSource.HUMAN, None, LATER)
    assert effective_labels(conn) == {1: Label.LORE}


def test_effective_labels_prefers_human_regardless_of_insert_order(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    set_label(conn, 1, Label.LORE, LabelSource.HUMAN, None, NOW)
    set_label(conn, 1, Label.NOISE, LabelSource.LLM, None, LATER)
    assert effective_labels(conn) == {1: Label.LORE}


def test_effective_labels_covers_every_labeled_exchange(tmp_path: Path) -> None:
    conn = seeded(tmp_path, exchange_ids=(1, 2, 3))
    set_label(conn, 1, Label.LORE, LabelSource.LLM, None, NOW)
    set_label(conn, 2, Label.NOISE, LabelSource.HUMAN, None, NOW)
    assert effective_labels(conn) == {1: Label.LORE, 2: Label.NOISE}


def test_label_counts_on_a_fresh_database(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    assert label_counts(conn) == LabelCounts(by_source={}, effective={})


def test_label_counts_by_source_and_effective(tmp_path: Path) -> None:
    conn = seeded(tmp_path, exchange_ids=(1, 2, 3))
    set_label(conn, 1, Label.LORE, LabelSource.LLM, None, NOW)
    set_label(conn, 2, Label.NOISE, LabelSource.LLM, None, NOW)
    set_label(conn, 2, Label.LORE, LabelSource.HUMAN, None, LATER)
    set_label(conn, 3, Label.NOISE, LabelSource.HUMAN, None, LATER)
    counts = label_counts(conn)
    assert counts.by_source == {
        "llm": {"lore": 1, "noise": 1},
        "human": {"lore": 1, "noise": 1},
    }
    assert counts.effective == {"lore": 2, "noise": 1}
