"""Tests for the shared batch loader (issue #104).

`exchange_inputs_for_ids` must return exactly what the per-exchange loaders
(`exchange_message_ids` + `messages_by_ids` + `reactions_for_messages` +
`attachments_for_messages`) would have returned, for many exchanges in a few
set-based queries instead of 4 queries per exchange -- including chunking
long IN (...) lists to stay under SQLite's bound-parameter limit.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.batch import SQLITE_MAX_VARIABLES, exchange_inputs_for_ids
from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import exchange_message_ids, insert_exchange
from infovore.db.raw import (
    attachments_for_messages,
    messages_by_ids,
    reactions_for_messages,
    set_reaction_count,
    upsert_attachment,
)
from infovore.rows import AttachmentRow, ExchangeRow, ExtractionStatus, GroupingRule, MessageRow

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def make_message(message_id: int, channel_id: int = 1, content: str = "hi") -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=channel_id,
        guild_id=9,
        author_id=1,
        author_name_at_time="alice",
        author_is_bot=False,
        created_at=NOW,
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        deleted_at=None,
        ingested_at=NOW,
        raw_json="{}",
    )


def seed_exchange(conn: sqlite3.Connection, messages: list[MessageRow], content_hash: str) -> int:
    for message in messages:
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " author_is_bot, created_at, content, ingested_at, raw_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                message.id,
                message.channel_id,
                message.guild_id,
                message.author_id,
                message.author_name_at_time,
                int(message.author_is_bot),
                message.created_at.isoformat(),
                message.content,
                message.ingested_at.isoformat(),
                message.raw_json,
            ),
        )
    row = ExchangeRow(
        id=None,
        channel_id=messages[0].channel_id,
        thread_id=None,
        first_message_id=messages[0].id,
        last_message_id=messages[-1].id,
        started_at=NOW,
        ended_at=NOW,
        message_count=len(messages),
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=content_hash,
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    return insert_exchange(conn, row, [m.id for m in messages])


def test_exchange_inputs_for_ids_returns_empty_dict_for_empty_ids(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert exchange_inputs_for_ids(conn, []) == {}


def test_exchange_inputs_for_ids_matches_the_per_exchange_loaders(tmp_path: Path) -> None:
    conn = db(tmp_path)
    e1 = seed_exchange(
        conn,
        [make_message(1, content="alpha octane"), make_message(2, content="beta")],
        "h1",
    )
    e2 = seed_exchange(conn, [make_message(3, channel_id=2, content="gamma")], "h2")
    set_reaction_count(conn, 1, "\U0001f44d", 2)
    set_reaction_count(conn, 2, "\U0001f525", 1)
    upsert_attachment(conn, AttachmentRow(1, 1, "a.pdf", "application/pdf", 10, "u", None, None))
    upsert_attachment(conn, AttachmentRow(2, 3, "b.png", "image/png", 20, "u", None, None))

    result = exchange_inputs_for_ids(conn, [e1, e2])

    assert set(result) == {e1, e2}
    for exchange_id in (e1, e2):
        message_ids = exchange_message_ids(conn, exchange_id)
        assert result[exchange_id].messages == messages_by_ids(conn, message_ids)
        assert result[exchange_id].reactions == reactions_for_messages(conn, message_ids)
        assert result[exchange_id].attachments == attachments_for_messages(conn, message_ids)


def test_exchange_inputs_for_ids_handles_exchanges_sharing_nothing(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ids = []
    for i in range(5):
        first = i * 10 + 1
        exchange_id = seed_exchange(
            conn,
            [make_message(first, channel_id=i, content=f"content-{i}")],
            f"hash-{i}",
        )
        set_reaction_count(conn, first, "✅", i + 1)
        ids.append(exchange_id)

    result = exchange_inputs_for_ids(conn, ids)

    for i, exchange_id in enumerate(ids):
        assert [m.content for m in result[exchange_id].messages] == [f"content-{i}"]
        assert [r.count for r in result[exchange_id].reactions] == [i + 1]
        assert result[exchange_id].attachments == []


def test_exchange_inputs_for_ids_missing_exchange_has_no_inputs(tmp_path: Path) -> None:
    conn = db(tmp_path)
    result = exchange_inputs_for_ids(conn, [12345])
    assert result[12345].messages == []
    assert result[12345].reactions == []
    assert result[12345].attachments == []


def test_exchange_inputs_for_ids_skips_message_ids_that_do_not_exist(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, [make_message(1)], "h1")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, ?)",
        (exchange_id, 999, 1),
    )
    conn.execute("PRAGMA foreign_keys = ON")

    result = exchange_inputs_for_ids(conn, [exchange_id])

    assert [m.id for m in result[exchange_id].messages] == [1]


def test_exchange_inputs_for_ids_chunks_beyond_the_sqlite_variable_limit(tmp_path: Path) -> None:
    conn = db(tmp_path)
    conn.execute("BEGIN IMMEDIATE")
    total = SQLITE_MAX_VARIABLES + 50
    message_rows = []
    exchange_rows = []
    exchange_message_rows = []
    for i in range(total):
        message_id = i + 1
        message_rows.append(
            (
                message_id,
                1,
                9,
                1,
                "alice",
                0,
                NOW.isoformat(),
                f"m{message_id}",
                NOW.isoformat(),
                "{}",
            )
        )
        exchange_rows.append(
            (
                message_id,
                1,
                None,
                message_id,
                message_id,
                NOW.isoformat(),
                NOW.isoformat(),
                1,
                "quiet_gap",
                f"hash-{message_id}",
                None,
                "pending",
                0,
                None,
            )
        )
        exchange_message_rows.append((message_id, message_id, 0))
    conn.executemany(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        message_rows,
    )
    conn.executemany(
        "INSERT INTO exchanges (id, channel_id, thread_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash, parent_exchange_id,"
        " extraction_status, retry_count, last_error)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        exchange_rows,
    )
    conn.executemany(
        "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, ?)",
        exchange_message_rows,
    )
    conn.execute("COMMIT")
    ids = list(range(1, total + 1))

    result = exchange_inputs_for_ids(conn, ids)

    assert len(result) == total
    for exchange_id in ids:
        assert [m.id for m in result[exchange_id].messages] == [exchange_id]
