import sqlite3
from dataclasses import dataclass

from infovore.db.channel_filter import exclude_channels_clause
from infovore.db.messages_fts import DEFAULT_SEARCH_LIMIT, as_fts_query
from infovore.privacy.optout import REDACTED_CONTENT
from infovore.triage.gate import gate_sql

SNIPPETS_PER_EXCHANGE = 2
SNIPPET_CHARS = 200
_NOT_OPTED_OUT = "m.author_id NOT IN (SELECT user_id FROM opt_outs)"
SHAREABLE_MESSAGE = (
    f"{_NOT_OPTED_OUT} AND m.deleted_at IS NULL AND m.content != '{REDACTED_CONTENT}'"
)


@dataclass(frozen=True)
class ExchangeHit:
    exchange_id: int
    channel_name: str
    started_at: str
    ended_at: str
    participants: int
    hit_count: int
    passes_gate: bool
    snippets: tuple[str, ...]
    jump_url: str


def snippet(content: str) -> str:
    flat = " ".join(content.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 3] + "..."


def search_exchanges(
    conn: sqlite3.Connection,
    query: str,
    min_score: float,
    min_p_lore: float,
    include_rejected: bool = False,
    exclude_channels: frozenset[str] = frozenset(),
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[ExchangeHit]:
    """Ranks whole conversations by the summed bm25 of their matching messages
    (bm25 is negative, so the lowest sum is best), ties broken by recency, so a
    thread where the term recurs outranks one stray mention (issue #191). The
    opt-out filter is applied here as well as by redaction, so a hit never
    depends on `sync-opt-outs` having already run."""
    match = as_fts_query(query)
    if not match:
        return []
    gate_clause, gate_params = gate_sql(min_score, min_p_lore)
    excl_clause, excl_params = exclude_channels_clause("e.channel_id", exclude_channels)
    rows = conn.execute(
        "SELECT em.exchange_id AS exchange_id, f.score AS score, m.content AS content,"
        f" ({gate_clause}) AS passes_gate, e.ended_at AS ended_at"
        " FROM (SELECT rowid, bm25(messages_fts) AS score FROM messages_fts"
        "        WHERE messages_fts MATCH ?) f"
        " JOIN messages m ON m.id = f.rowid"
        " JOIN exchange_messages em ON em.message_id = m.id"
        " JOIN current_exchanges e ON e.id = em.exchange_id"
        f" WHERE {SHAREABLE_MESSAGE}{excl_clause}"
        " ORDER BY f.score, m.id",
        (*gate_params, match, *excl_params),
    ).fetchall()
    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    snippets: dict[int, list[str]] = {}
    gated: dict[int, bool] = {}
    ended: dict[int, str] = {}
    for row in rows:
        exchange_id = row["exchange_id"]
        gated[exchange_id] = bool(row["passes_gate"])
        ended[exchange_id] = row["ended_at"]
        scores[exchange_id] = scores.get(exchange_id, 0.0) + row["score"]
        counts[exchange_id] = counts.get(exchange_id, 0) + 1
        best = snippets.setdefault(exchange_id, [])
        if len(best) < SNIPPETS_PER_EXCHANGE:
            best.append(snippet(row["content"]))
    visible = [i for i in scores if include_rejected or gated[i]]
    by_recency = sorted(visible, key=lambda i: ended[i], reverse=True)
    ranked = sorted(by_recency, key=lambda i: scores[i])[:limit]
    return [
        _describe(conn, exchange_id, counts[exchange_id], gated[exchange_id], snippets[exchange_id])
        for exchange_id in ranked
    ]


def _describe(
    conn: sqlite3.Connection,
    exchange_id: int,
    hit_count: int,
    passes_gate: bool,
    snippets: list[str],
) -> ExchangeHit:
    row = conn.execute(
        "SELECT c.name AS channel_name, e.started_at AS started_at, e.ended_at AS ended_at,"
        " fm.guild_id AS guild_id, fm.channel_id AS channel_id, fm.id AS first_id,"
        " (SELECT COUNT(DISTINCT m.author_id) FROM exchange_messages em"
        "   JOIN messages m ON m.id = em.message_id"
        f"  WHERE em.exchange_id = e.id AND {_NOT_OPTED_OUT}) AS participants"
        " FROM current_exchanges e"
        " JOIN messages fm ON fm.id = e.first_message_id"
        " JOIN channels c ON c.id = fm.channel_id"
        " WHERE e.id = ?",
        (exchange_id,),
    ).fetchone()
    url = f"https://discord.com/channels/{row['guild_id']}/{row['channel_id']}/{row['first_id']}"
    return ExchangeHit(
        exchange_id=exchange_id,
        channel_name=row["channel_name"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        participants=row["participants"],
        hit_count=hit_count,
        passes_gate=passes_gate,
        snippets=tuple(snippets),
        jump_url=url,
    )
