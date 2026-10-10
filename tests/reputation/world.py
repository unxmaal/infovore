import sqlite3
from itertools import count
from pathlib import Path

from infovore.db.annotations import Annotation, record_annotation
from infovore.db.connection import migrate, open_database
from infovore.triage.human import HUMAN_SCORER
from tests.claims.seed import NOW, SALT, Line, conversation

_outside = count(900_000)


def world(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "infovore.db")
    migrate(conn)
    return conn


def exchange(
    conn: sqlite3.Connection, lines: list[Line], base: int, held_out: bool = False
) -> tuple[int, list[int]]:
    return conversation(conn, lines, base, slice_name="gold" if held_out else None)


def reply(
    conn: sqlite3.Connection,
    to: int,
    author: int,
    text: str = "ok then",
    *,
    bot: bool = False,
    deleted: bool = False,
) -> int:
    message_id = next(_outside)
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, content, reply_to_id, deleted_at, ingested_at, raw_json)"
        " VALUES (?, 1, 9, ?, 'n', ?, ?, ?, ?, ?, ?, '{}')",
        (
            message_id,
            author,
            int(bot),
            NOW.isoformat(),
            text,
            to,
            NOW.isoformat() if deleted else None,
            NOW.isoformat(),
        ),
    )
    return message_id


def react(conn: sqlite3.Connection, message_id: int, total: int) -> None:
    conn.execute(
        "INSERT INTO reactions (message_id, emoji, count) VALUES (?, 'x', ?)", (message_id, total)
    )


def label_message(
    conn: sqlite3.Connection,
    message_id: int,
    keep: bool,
    regime: str = "context",
    source: str = "human",
) -> None:
    conn.execute(
        "INSERT INTO message_labels (message_id, label, source, source_ref, labeled_at, regime)"
        " VALUES (?, ?, ?, 'r', 't', ?)",
        (message_id, "keep" if keep else "trash", source, regime),
    )


def label_exchange(conn: sqlite3.Connection, exchange_id: int, relevant: bool) -> None:
    record_annotation(
        conn,
        Annotation(
            "exchange",
            exchange_id,
            HUMAN_SCORER,
            1,
            "recorded",
            label="relevant" if relevant else "irrelevant",
        ),
        NOW,
    )


def mark_stage(
    conn: sqlite3.Connection, exchange_id: int, stage: str, label: str | None, version: int = 1
) -> None:
    record_annotation(
        conn,
        Annotation(
            "exchange",
            exchange_id,
            f"relevance_{stage}",
            version,
            "derived",
            score=0.5 if label is None else None,
            label=label,
            recipe={"x": 1},
        ),
        NOW,
    )


__all__ = ["SALT", "exchange", "label_exchange", "label_message", "mark_stage", "react", "reply"]
