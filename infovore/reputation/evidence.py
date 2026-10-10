import sqlite3
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from itertools import groupby
from typing import Final

from infovore.claims.redact import MENTION
from infovore.claims.speakers import _resolve
from infovore.db.channel_filter import excluded_exchange_ids
from infovore.reputation.people import People
from infovore.rows import Label

RESPONSE: Final = ("replies", "reactions", "mentions", "answered")
LABEL: Final = ("messages", "exchanges", "claims")
SIGNALS: Final = RESPONSE + LABEL
QUESTION_CHARS: Final = 15

Cell = tuple[float, float]
Ledger = dict[str, dict[str, Cell]]

_SCAN: Final = (
    "SELECT em.exchange_id AS exchange_id, m.id AS id, m.author_id AS author_id,"
    " m.content AS content,"
    " (SELECT group_concat(DISTINCT r.author_id) FROM messages r WHERE r.reply_to_id = m.id"
    "  AND r.deleted_at IS NULL AND r.author_is_bot = 0) AS repliers,"
    " (SELECT COALESCE(SUM(x.count), 0) FROM reactions x WHERE x.message_id = m.id) AS reacted,"
    " (SELECT l.label FROM message_labels l WHERE l.message_id = m.id AND l.source = 'human'"
    "  AND l.regime IN ('context', 'value')) AS label"
    " FROM exchange_messages em JOIN current_exchanges e ON e.id = em.exchange_id"
    " JOIN messages m ON m.id = em.message_id"
    " WHERE m.deleted_at IS NULL AND m.author_is_bot = 0"
    " ORDER BY em.exchange_id, em.position"
)
_CLAIMS: Final = (
    "SELECT c.exchange_id AS exchange_id, c.speaker AS speaker, r.verdict AS verdict"
    " FROM claims_v2 c JOIN current_claim_reviews r ON r.claim_id = c.id"
    " WHERE r.interface = 'conversation'"
)


@dataclass(frozen=True)
class Evidence:
    totals: Ledger
    priors: dict[str, float]
    contributions: dict[int, Ledger]

    def contribution(self, exchange_id: int) -> Ledger:
        return self.contributions.get(exchange_id, {})

    def volume(self, person: str) -> float:
        return self.totals.get(person, {}).get("replies", (0.0, 0.0))[1]


def add_cell(ledger: Ledger, person: str, signal: str, x: float, n: float) -> None:
    old = ledger.setdefault(person, {}).get(signal, (0.0, 0.0))
    ledger[person][signal] = (old[0] + x, old[1] + n)


class _Books:
    def __init__(self, keep: Collection[int]) -> None:
        self.totals: Ledger = {}
        self.contributions: dict[int, Ledger] = {}
        self._keep = frozenset(keep)

    def credit(self, exchange_id: int, person: str, signal: str, x: float, n: float) -> None:
        add_cell(self.totals, person, signal, x, n)
        if exchange_id in self._keep:
            add_cell(self.contributions.setdefault(exchange_id, {}), person, signal, x, n)


def _priors(totals: Ledger) -> dict[str, float]:
    sums: dict[str, list[float]] = {signal: [0.0, 0.0] for signal in SIGNALS}
    for cells in totals.values():
        for signal, (x, n) in cells.items():
            sums[signal][0] += x
            sums[signal][1] += n
    return {signal: x / n if n else 0.0 for signal, (x, n) in sums.items()}


def build_evidence(
    conn: sqlite3.Connection,
    people: People,
    salt: str,
    labels: Mapping[int, Label],
    exclude_channels: frozenset[str],
    held_out: frozenset[int],
    keep: Collection[int],
) -> Evidence:
    skip = held_out | excluded_exchange_ids(conn, exclude_channels)
    books = _Books(keep)
    seen: set[int] = set()
    for exchange_id, rows in groupby(conn.execute(_SCAN), key=lambda row: row["exchange_id"]):
        messages = list(rows)
        if exchange_id in skip:
            continue
        seen.add(exchange_id)
        for row in messages:
            _credit_message(books, people, exchange_id, row)
        if exchange_id in labels:
            shares = Counter(people.person(row["author_id"]) for row in messages)
            hit = 1.0 if labels[exchange_id] is Label.LORE else 0.0
            for person, count in shares.items():
                share = count / len(messages)
                books.credit(exchange_id, person, "exchanges", hit * share, share)
    resolved = _resolve(conn, salt, "SELECT exchange_id FROM claims_v2", [])
    for row in conn.execute(_CLAIMS):
        author = resolved.get((row["exchange_id"], row["speaker"]))
        if author is not None and row["exchange_id"] in seen:
            good = 1.0 if row["verdict"] == "good" else 0.0
            books.credit(row["exchange_id"], people.person(author), "claims", good, 1.0)
    return Evidence(books.totals, _priors(books.totals), books.contributions)


def _credit_message(books: _Books, people: People, exchange_id: int, row: sqlite3.Row) -> None:
    person = people.person(row["author_id"])
    others = {people.person(int(a)) for a in (row["repliers"] or "").split(",") if a} - {person}
    books.credit(exchange_id, person, "replies", len(others), 1.0)
    books.credit(exchange_id, person, "reactions", row["reacted"], 1.0)
    books.credit(exchange_id, person, "mentions", 0.0, 1.0)
    for mentioned in {people.person(int(m)) for m in MENTION.findall(row["content"])} - {person}:
        books.credit(exchange_id, mentioned, "mentions", 1.0, 0.0)
    content = row["content"]
    if "?" in content and len(content) >= QUESTION_CHARS:
        books.credit(exchange_id, person, "answered", 1.0 if others else 0.0, 1.0)
    if row["label"] is not None:
        books.credit(exchange_id, person, "messages", 1.0 if row["label"] == "keep" else 0.0, 1.0)
