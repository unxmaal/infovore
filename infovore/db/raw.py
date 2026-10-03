import sqlite3
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum

from infovore.db.codec import from_db_time, to_db_time
from infovore.db.connection import transaction
from infovore.rows import (
    AttachmentRow,
    ChannelKind,
    ChannelRow,
    MessageRevisionRow,
    MessageRow,
    ReactionRow,
)


class UpsertOutcome(StrEnum):
    INSERTED = "inserted"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


def upsert_channel(conn: sqlite3.Connection, channel: ChannelRow) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT INTO channels (id, guild_id, parent_id, name, kind, archived,"
            " last_backfilled_message_id) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (id) DO UPDATE SET guild_id = excluded.guild_id,"
            " parent_id = excluded.parent_id, name = excluded.name, kind = excluded.kind,"
            " archived = excluded.archived",
            (
                channel.id,
                channel.guild_id,
                channel.parent_id,
                channel.name,
                channel.kind.value,
                int(channel.archived),
                channel.last_backfilled_message_id,
            ),
        )


def get_channel(conn: sqlite3.Connection, channel_id: int) -> ChannelRow | None:
    row = conn.execute("SELECT * FROM channels WHERE id = ?", (channel_id,)).fetchone()
    if row is None:
        return None
    return ChannelRow(
        id=row["id"],
        guild_id=row["guild_id"],
        parent_id=row["parent_id"],
        name=row["name"],
        kind=ChannelKind(row["kind"]),
        archived=bool(row["archived"]),
        last_backfilled_message_id=row["last_backfilled_message_id"],
    )


def get_backfill_checkpoint(conn: sqlite3.Connection, channel_id: int) -> int | None:
    row = conn.execute(
        "SELECT last_backfilled_message_id FROM channels WHERE id = ?", (channel_id,)
    ).fetchone()
    if row is None:
        return None
    checkpoint: int | None = row[0]
    return checkpoint


def _set_backfill_checkpoint(conn: sqlite3.Connection, channel_id: int, message_id: int) -> None:
    conn.execute(
        "UPDATE channels SET last_backfilled_message_id = ? WHERE id = ?",
        (message_id, channel_id),
    )


def set_backfill_checkpoint(conn: sqlite3.Connection, channel_id: int, message_id: int) -> None:
    with transaction(conn):
        _set_backfill_checkpoint(conn, channel_id, message_id)


def _message_core(conn: sqlite3.Connection, message_id: int) -> sqlite3.Row | None:
    row: sqlite3.Row | None = conn.execute(
        "SELECT content, edited_at, raw_json FROM messages WHERE id = ?", (message_id,)
    ).fetchone()
    return row


def _next_revision(conn: sqlite3.Connection, message_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(revision), -1) + 1 FROM message_revisions WHERE message_id = ?",
        (message_id,),
    ).fetchone()
    revision: int = row[0]
    return revision


def _archive_revision(
    conn: sqlite3.Connection,
    message_id: int,
    content: str,
    edited_at: str | None,
    raw_json: str,
) -> None:
    conn.execute(
        "INSERT INTO message_revisions (message_id, revision, content, edited_at, raw_json)"
        " VALUES (?, ?, ?, ?, ?)",
        (message_id, _next_revision(conn, message_id), content, edited_at, raw_json),
    )


def _apply_edit(
    conn: sqlite3.Connection,
    message_id: int,
    existing: sqlite3.Row,
    content: str,
    edited_at: datetime | None,
    raw_json: str,
) -> None:
    _archive_revision(
        conn, message_id, existing["content"], existing["edited_at"], existing["raw_json"]
    )
    conn.execute(
        "UPDATE messages SET content = ?, edited_at = ?, raw_json = ? WHERE id = ?",
        (content, to_db_time(edited_at), raw_json, message_id),
    )


def _insert_message(conn: sqlite3.Connection, message: MessageRow) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, edited_at, content, reply_to_id, thread_id, deleted_at,"
        " ingested_at, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            message.id,
            message.channel_id,
            message.guild_id,
            message.author_id,
            message.author_name_at_time,
            int(message.author_is_bot),
            to_db_time(message.created_at),
            to_db_time(message.edited_at),
            message.content,
            message.reply_to_id,
            message.thread_id,
            to_db_time(message.deleted_at),
            to_db_time(message.ingested_at),
            message.raw_json,
        ),
    )


def _upsert_message(conn: sqlite3.Connection, message: MessageRow) -> UpsertOutcome:
    existing = _message_core(conn, message.id)
    if existing is None:
        _insert_message(conn, message)
        return UpsertOutcome.INSERTED
    if existing["content"] == message.content:
        return UpsertOutcome.UNCHANGED
    _apply_edit(conn, message.id, existing, message.content, message.edited_at, message.raw_json)
    return UpsertOutcome.UPDATED


def upsert_message(conn: sqlite3.Connection, message: MessageRow) -> UpsertOutcome:
    with transaction(conn):
        return _upsert_message(conn, message)


def mark_edited(
    conn: sqlite3.Connection,
    message_id: int,
    content: str,
    edited_at: datetime | None,
    raw_json: str,
) -> bool:
    with transaction(conn):
        existing = _message_core(conn, message_id)
        if existing is None:
            return False
        if existing["content"] == content:
            return True
        _apply_edit(conn, message_id, existing, content, edited_at, raw_json)
        return True


def mark_deleted(conn: sqlite3.Connection, message_id: int, at: datetime) -> bool:
    with transaction(conn):
        row = conn.execute("SELECT deleted_at FROM messages WHERE id = ?", (message_id,)).fetchone()
        if row is None:
            return False
        if row["deleted_at"] is not None:
            return True
        conn.execute(
            "UPDATE messages SET deleted_at = ? WHERE id = ?", (to_db_time(at), message_id)
        )
        return True


def message_from_row(row: sqlite3.Row) -> MessageRow:
    return MessageRow(
        id=row["id"],
        channel_id=row["channel_id"],
        guild_id=row["guild_id"],
        author_id=row["author_id"],
        author_name_at_time=row["author_name_at_time"],
        author_is_bot=bool(row["author_is_bot"]),
        created_at=from_db_time(row["created_at"]),
        edited_at=from_db_time(row["edited_at"]),
        content=row["content"],
        reply_to_id=row["reply_to_id"],
        thread_id=row["thread_id"],
        deleted_at=from_db_time(row["deleted_at"]),
        ingested_at=from_db_time(row["ingested_at"]),
        raw_json=row["raw_json"],
    )


def get_message(conn: sqlite3.Connection, message_id: int) -> MessageRow | None:
    row = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    if row is None:
        return None
    return message_from_row(row)


def message_revisions(conn: sqlite3.Connection, message_id: int) -> list[MessageRevisionRow]:
    rows = conn.execute(
        "SELECT message_id, revision, content, edited_at, raw_json FROM message_revisions"
        " WHERE message_id = ? ORDER BY revision",
        (message_id,),
    ).fetchall()
    return [
        MessageRevisionRow(
            message_id=row["message_id"],
            revision=row["revision"],
            content=row["content"],
            edited_at=from_db_time(row["edited_at"]),
            raw_json=row["raw_json"],
        )
        for row in rows
    ]


def _upsert_attachment(conn: sqlite3.Connection, attachment: AttachmentRow) -> None:
    conn.execute(
        "INSERT INTO attachments (id, message_id, filename, content_type, size, url,"
        " sha256, local_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (id) DO UPDATE SET message_id = excluded.message_id,"
        " filename = excluded.filename, content_type = excluded.content_type,"
        " size = excluded.size, url = excluded.url, sha256 = excluded.sha256,"
        " local_path = excluded.local_path",
        (
            attachment.id,
            attachment.message_id,
            attachment.filename,
            attachment.content_type,
            attachment.size,
            attachment.url,
            attachment.sha256,
            attachment.local_path,
        ),
    )


def upsert_attachment(conn: sqlite3.Connection, attachment: AttachmentRow) -> None:
    with transaction(conn):
        _upsert_attachment(conn, attachment)


def attachments_for_messages(
    conn: sqlite3.Connection, message_ids: Sequence[int]
) -> list[AttachmentRow]:
    ids = list(message_ids)
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        "SELECT id, message_id, filename, content_type, size, url, sha256, local_path"
        f" FROM attachments WHERE message_id IN ({placeholders}) ORDER BY message_id, id",
        ids,
    ).fetchall()
    return [
        AttachmentRow(
            id=row["id"],
            message_id=row["message_id"],
            filename=row["filename"],
            content_type=row["content_type"],
            size=row["size"],
            url=row["url"],
            sha256=row["sha256"],
            local_path=row["local_path"],
        )
        for row in rows
    ]


def _set_reaction_count(conn: sqlite3.Connection, message_id: int, emoji: str, count: int) -> None:
    if count <= 0:
        conn.execute(
            "DELETE FROM reactions WHERE message_id = ? AND emoji = ?", (message_id, emoji)
        )
        return
    conn.execute(
        "INSERT INTO reactions (message_id, emoji, count) VALUES (?, ?, ?)"
        " ON CONFLICT (message_id, emoji) DO UPDATE SET count = excluded.count",
        (message_id, emoji, count),
    )


def set_reaction_count(conn: sqlite3.Connection, message_id: int, emoji: str, count: int) -> None:
    with transaction(conn):
        _set_reaction_count(conn, message_id, emoji, count)


def reactions_for_messages(
    conn: sqlite3.Connection, message_ids: Sequence[int]
) -> list[ReactionRow]:
    ids = list(message_ids)
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        "SELECT message_id, emoji, count FROM reactions"
        f" WHERE message_id IN ({placeholders}) ORDER BY message_id, emoji",
        ids,
    ).fetchall()
    return [
        ReactionRow(message_id=row["message_id"], emoji=row["emoji"], count=row["count"])
        for row in rows
    ]


def ungrouped_channel_ids(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        "SELECT DISTINCT channel_id FROM messages WHERE id NOT IN"
        " (SELECT message_id FROM exchange_messages) ORDER BY channel_id"
    ).fetchall()
    return [int(row["channel_id"]) for row in rows]


def ungrouped_messages_for_channel(conn: sqlite3.Connection, channel_id: int) -> list[MessageRow]:
    rows = conn.execute(
        "SELECT * FROM messages WHERE channel_id = ? AND id NOT IN"
        " (SELECT message_id FROM exchange_messages) ORDER BY created_at, id",
        (channel_id,),
    ).fetchall()
    return [message_from_row(row) for row in rows]


def light_channel_messages(conn: sqlite3.Connection, channel_id: int) -> list[MessageRow]:
    rows = conn.execute(
        "SELECT id, guild_id, author_id, author_is_bot, created_at, reply_to_id, thread_id,"
        " deleted_at FROM messages WHERE channel_id = ? ORDER BY created_at, id",
        (channel_id,),
    )
    return [
        MessageRow(
            id=row["id"],
            channel_id=channel_id,
            guild_id=row["guild_id"],
            author_id=row["author_id"],
            author_name_at_time="",
            author_is_bot=bool(row["author_is_bot"]),
            created_at=from_db_time(row["created_at"]),
            edited_at=None,
            content="",
            reply_to_id=row["reply_to_id"],
            thread_id=row["thread_id"],
            deleted_at=from_db_time(row["deleted_at"]),
            ingested_at=from_db_time(row["created_at"]),
            raw_json="",
        )
        for row in rows
    ]


def latest_exchange_for_thread(conn: sqlite3.Connection, thread_id: int) -> int | None:
    row = conn.execute(
        "SELECT id FROM exchanges WHERE thread_id = ? AND superseded_by_recipe IS NULL"
        " ORDER BY started_at DESC, id DESC LIMIT 1",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row["id"])


def messages_by_ids(conn: sqlite3.Connection, message_ids: Sequence[int]) -> list[MessageRow]:
    ids = list(message_ids)
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(f"SELECT * FROM messages WHERE id IN ({placeholders})", ids).fetchall()
    by_id = {row["id"]: message_from_row(row) for row in rows}
    return [by_id[message_id] for message_id in ids if message_id in by_id]
