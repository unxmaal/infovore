import logging
import sqlite3
from dataclasses import dataclass, replace

from infovore.db import claims
from infovore.db.codec import to_db_time
from infovore.db.connection import transaction
from infovore.ingest.normalize import NormalizedMessage
from infovore.source.protocol import DiscordSource
from infovore.timing import Clock

REDACTED_CONTENT = "[redacted]"
REDACTED_AUTHOR = "[redacted]"

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncReport:
    added: frozenset[int]
    removed: frozenset[int]
    redacted_messages: int
    retracted_claims: tuple[int, ...]


def opted_out_user_ids(conn: sqlite3.Connection) -> frozenset[int]:
    rows = conn.execute("SELECT user_id FROM opt_outs").fetchall()
    return frozenset(row["user_id"] for row in rows)


def redact_normalized(
    normalized: NormalizedMessage, opted_out: frozenset[int]
) -> NormalizedMessage:
    if normalized.message.author_id not in opted_out:
        return normalized
    message = replace(
        normalized.message,
        content=REDACTED_CONTENT,
        author_name_at_time=REDACTED_AUTHOR,
        raw_json="{}",
    )
    return NormalizedMessage(message=message, attachments=(), reactions=normalized.reactions)


def redact_stored(conn: sqlite3.Connection, user_ids: frozenset[int]) -> int:
    if not user_ids:
        return 0
    placeholders = ",".join("?" for _ in user_ids)
    params = tuple(user_ids)
    authored = f"SELECT id FROM messages WHERE author_id IN ({placeholders})"
    with transaction(conn):
        redacted = conn.execute(
            "UPDATE messages SET content = ?, author_name_at_time = ?, raw_json = ?"
            f" WHERE author_id IN ({placeholders})",
            (REDACTED_CONTENT, REDACTED_AUTHOR, "{}", *params),
        ).rowcount
        conn.execute(
            "UPDATE message_revisions SET content = ?, raw_json = ?"
            f" WHERE message_id IN ({authored})",
            (REDACTED_CONTENT, "{}", *params),
        )
        conn.execute(f"DELETE FROM attachments WHERE message_id IN ({authored})", params)
    return redacted


async def sync_opt_outs(
    conn: sqlite3.Connection,
    source: DiscordSource,
    guild_id: int,
    role_name: str,
    clock: Clock,
) -> SyncReport:
    role_members = await source.role_member_ids(guild_id, role_name)
    existing = opted_out_user_ids(conn)
    added = frozenset(role_members - existing)
    removed = frozenset(existing - role_members)
    now = clock.now()
    if added or removed:
        with transaction(conn):
            for user_id in added:
                conn.execute(
                    "INSERT INTO opt_outs (user_id, since) VALUES (?, ?)",
                    (user_id, to_db_time(now)),
                )
            for user_id in removed:
                conn.execute("DELETE FROM opt_outs WHERE user_id = ?", (user_id,))
    redacted_messages = 0
    retracted_claims: tuple[int, ...] = ()
    if added:
        redacted_messages = redact_stored(conn, added)
        retracted_claims = tuple(claims.retract_claims_with_all_sources_opted_out(conn, now))
    for user_id in added:
        _LOGGER.info("opt-out added user_id=%s", user_id)
    for user_id in removed:
        _LOGGER.info("opt-out removed user_id=%s", user_id)
    return SyncReport(
        added=added,
        removed=removed,
        redacted_messages=redacted_messages,
        retracted_claims=retracted_claims,
    )
