import sqlite3
from dataclasses import dataclass

from infovore.db.fts import match_terms, quote

DEFAULT_SEARCH_LIMIT = 50
DEFAULT_CONTEXT = 0


def as_fts_query(query: str) -> str:
    """Terms are ANDed, which is what a reader searching the archive means:
    every word should appear. `related_claims` ORs instead because it is
    fishing for recall. That difference is the only thing the two query
    builders should disagree about (issue #179)."""
    return " ".join(quote(term) for term in match_terms(query.split()))


@dataclass(frozen=True)
class MessageHit:
    message_id: int
    channel_name: str
    created_at: str
    author_name: str
    content: str
    exchange_id: int | None


def search_messages(
    conn: sqlite3.Connection, query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[MessageHit]:
    """Full-text search over the raw corpus, which is what makes a claim the
    extractor missed recoverable rather than lost: `claims_fts` only covers
    what extraction already produced (issue #176). Returns the exchange so a
    hit can be read back in its conversation, and `None` for the 54% of
    messages belonging to no exchange the gate ever admitted."""
    match = as_fts_query(query)
    if not match:
        return []
    rows = conn.execute(
        "SELECT m.id AS message_id, c.name AS channel_name, m.created_at AS created_at,"
        " m.author_name_at_time AS author_name, m.content AS content,"
        " em.exchange_id AS exchange_id"
        " FROM messages_fts f"
        " JOIN messages m ON m.id = f.rowid"
        " JOIN channels c ON c.id = m.channel_id"
        " LEFT JOIN exchange_messages em ON em.message_id = m.id"
        " WHERE messages_fts MATCH ?"
        " ORDER BY bm25(messages_fts), m.id"
        " LIMIT ?",
        (match, limit),
    ).fetchall()
    return [
        MessageHit(
            message_id=row["message_id"],
            channel_name=row["channel_name"],
            created_at=row["created_at"],
            author_name=row["author_name"],
            content=row["content"],
            exchange_id=row["exchange_id"],
        )
        for row in rows
    ]


@dataclass(frozen=True)
class ContextMessage:
    message_id: int
    created_at: str
    author_name: str
    content: str
    is_hit: bool


def message_context(conn: sqlite3.Connection, hit: MessageHit, size: int) -> list[ContextMessage]:
    """The messages either side of a hit in its CHANNEL, not its exchange.
    54% of messages belong to no exchange the gate ever admitted, and those are
    precisely the ones this index exists to reach, so exchange-scoped context
    would miss the majority case (issue #182).

    Ordered by id, which is a Discord snowflake and so monotonic in creation
    time, and bounded to the channel: a window that spilled into a neighbouring
    channel would read as one conversation."""
    rows = conn.execute(
        "SELECT * FROM ("
        "  SELECT id, created_at, author_name_at_time AS author, content FROM messages"
        "   WHERE channel_id = (SELECT channel_id FROM messages WHERE id = ?)"
        "     AND id <= ? ORDER BY id DESC LIMIT ?"
        ") UNION SELECT * FROM ("
        "  SELECT id, created_at, author_name_at_time AS author, content FROM messages"
        "   WHERE channel_id = (SELECT channel_id FROM messages WHERE id = ?)"
        "     AND id > ? ORDER BY id ASC LIMIT ?"
        ") ORDER BY id",
        (hit.message_id, hit.message_id, size + 1, hit.message_id, hit.message_id, size),
    ).fetchall()
    return [
        ContextMessage(
            message_id=row["id"],
            created_at=row["created_at"],
            author_name=row["author"],
            content=row["content"],
            is_hit=row["id"] == hit.message_id,
        )
        for row in rows
    ]
