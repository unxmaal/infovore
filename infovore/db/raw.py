import sqlite3

from infovore.db.connection import transaction
from infovore.rows import ChannelKind, ChannelRow


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


def set_backfill_checkpoint(conn: sqlite3.Connection, channel_id: int, message_id: int) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE channels SET last_backfilled_message_id = ? WHERE id = ?",
            (message_id, channel_id),
        )
