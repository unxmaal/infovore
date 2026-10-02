import sqlite3
from dataclasses import dataclass

from infovore.db.fts import match_terms, quote

DEFAULT_SEARCH_LIMIT = 50


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
        " m.content AS content, em.exchange_id AS exchange_id"
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
            content=row["content"],
            exchange_id=row["exchange_id"],
        )
        for row in rows
    ]
