import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from infovore.db.author_ratings import RATINGS
from infovore.db.channel_filter import excluded_exchange_ids
from infovore.reputation.score import Reputation
from infovore.reputation.stats import permutation_p, wilson
from infovore.triage.lexicon import Lexicon, message_hits

SHORT_CHARS: Final = 80
TOP_SHARE: Final = 0.10
PERMUTATIONS: Final = 10000

_LABELLED: Final = (
    "SELECT m.id AS id, em.exchange_id AS exchange_id, m.author_id AS author_id,"
    " m.content AS content, l.label AS label"
    " FROM message_labels l JOIN messages m ON m.id = l.message_id"
    " JOIN exchange_messages em ON em.message_id = m.id"
    " JOIN current_exchanges e ON e.id = em.exchange_id"
    " WHERE l.source = 'human' AND l.regime IN ('context', 'value')"
    " AND m.deleted_at IS NULL AND m.author_is_bot = 0"
    " AND length(trim(m.content)) BETWEEN 1 AND ? ORDER BY m.id"
)


@dataclass(frozen=True)
class ShortMessage:
    message_id: int
    exchange_id: int
    author_id: int
    keep: bool


@dataclass(frozen=True)
class ShortTest:
    n: int
    top_n: int
    base_rate: float
    top_rate: float
    low: float
    high: float
    p: float


@dataclass(frozen=True)
class RatingLevel:
    level: str
    n: int
    kept: int
    rate: float
    low: float
    high: float


def by_rating(reputation: Reputation, messages: Sequence[ShortMessage]) -> list[RatingLevel]:
    if reputation.ratings is None:
        return []
    buckets: dict[str, list[bool]] = {str(r): [] for r in RATINGS}
    buckets["unrated"] = []
    for m in messages:
        rating = reputation.ratings.get(reputation.people.person(m.author_id))
        buckets["unrated" if rating is None else str(rating)].append(m.keep)
    out = []
    for level, keeps in buckets.items():
        kept = sum(keeps)
        low, high = wilson(kept, len(keeps))
        rate = kept / len(keeps) if keeps else 0.0
        out.append(RatingLevel(level, len(keeps), kept, rate, low, high))
    return out


def short_messages(
    conn: sqlite3.Connection, lexicon: Lexicon, exclude_channels: frozenset[str]
) -> list[ShortMessage]:
    skip = excluded_exchange_ids(conn, exclude_channels)
    return [
        ShortMessage(row["id"], row["exchange_id"], row["author_id"], row["label"] == "keep")
        for row in conn.execute(_LABELLED, (SHORT_CHARS,))
        if row["exchange_id"] not in skip and not message_hits(lexicon, row["content"])
    ]


def short_test(
    reputation: Reputation,
    messages: Sequence[ShortMessage],
    seed: int,
    permutations: int = PERMUTATIONS,
) -> ShortTest | None:
    if not messages:
        return None
    scored = [
        (
            reputation.of(
                reputation.people.person(m.author_id),
                reputation.evidence.contribution(m.exchange_id),
            ),
            m.keep,
        )
        for m in messages
    ]
    ranked = [keep for _, keep in sorted(scored, key=lambda pair: -pair[0])]
    top = max(1, math.ceil(len(ranked) * TOP_SHARE))
    kept = sum(ranked[:top])
    low, high = wilson(kept, top)
    return ShortTest(
        len(ranked),
        top,
        sum(ranked) / len(ranked),
        kept / top,
        low,
        high,
        permutation_p(ranked, top, seed, permutations),
    )
