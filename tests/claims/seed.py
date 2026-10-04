import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.batch import exchange_inputs_for_ids
from infovore.db.connection import migrate, open_database
from infovore.rows import MessageRow

NOW = datetime(2026, 1, 1, tzinfo=UTC)
SALT = "test-salt"

Line = tuple[int, str, str]


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "infovore.db")
    migrate(conn)
    return conn


def environment(tmp_path: Path, salt: str | None = SALT) -> dict[str, str]:
    env = {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
    }
    if salt is not None:
        env["INFOVORE_PSEUDONYM_SALT"] = salt
    return env


def conversation(
    conn: sqlite3.Connection, lines: Sequence[Line], base: int, slice_name: str | None = "gold"
) -> tuple[int, list[int]]:
    ids = []
    for offset, (author_id, name, text) in enumerate(lines):
        message_id = base * 100 + offset + 1
        ids.append(message_id)
        conn.execute(
            "INSERT OR IGNORE INTO channels (id, guild_id, parent_id, name, kind)"
            " VALUES (1, 9, NULL, 'c', 'text')"
        )
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " author_is_bot, created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 9, ?, ?, 0, ?, ?, ?, '{}')",
            (message_id, author_id, name, NOW.isoformat(), text, NOW.isoformat()),
        )
    cursor = conn.execute(
        "INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (1, ?, ?, ?, ?, ?, 'quiet_gap', ?)",
        (ids[0], ids[-1], NOW.isoformat(), NOW.isoformat(), len(ids), f"h{base}"),
    )
    exchange_id = int(cursor.lastrowid or 0)
    for position, message_id in enumerate(ids, start=1):
        conn.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, ?)",
            (exchange_id, message_id, position),
        )
    if slice_name is not None:
        conn.execute(
            "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
            " VALUES (?, ?, ?, 'p', 1, ?)",
            (slice_name, exchange_id, base, NOW.isoformat()),
        )
    return exchange_id, ids


def messages_of(conn: sqlite3.Connection, exchange_id: int) -> list[MessageRow]:
    return exchange_inputs_for_ids(conn, [exchange_id])[exchange_id].messages
