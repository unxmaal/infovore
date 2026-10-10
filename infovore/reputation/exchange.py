import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence

from infovore.db.batch import SQLITE_MAX_VARIABLES
from infovore.reputation.evidence import Ledger
from infovore.reputation.people import People
from infovore.reputation.score import Reputation


def exchange_shares(
    conn: sqlite3.Connection, people: People, exchange_ids: Sequence[int]
) -> dict[int, dict[str, float]]:
    wanted = sorted(set(exchange_ids))
    counts: dict[int, Counter[str]] = {eid: Counter() for eid in wanted}
    for start in range(0, len(wanted), SQLITE_MAX_VARIABLES):
        chunk = wanted[start : start + SQLITE_MAX_VARIABLES]
        marks = ",".join("?" for _ in chunk)
        for row in conn.execute(
            "SELECT em.exchange_id AS eid, m.author_id AS author"
            " FROM exchange_messages em JOIN messages m ON m.id = em.message_id"
            f" WHERE em.exchange_id IN ({marks}) AND m.deleted_at IS NULL AND m.author_is_bot = 0",
            chunk,
        ):
            counts[row["eid"]][people.person(row["author"])] += 1
    return {
        eid: {person: n / sum(c.values()) for person, n in c.items()} for eid, c in counts.items()
    }


def exchange_score(
    reputation: Reputation, shares: Mapping[str, float], removed: Ledger | None = None
) -> float:
    return sum(share * reputation.of(person, removed) for person, share in shares.items())


def score_exchanges(
    conn: sqlite3.Connection, reputation: Reputation, exchange_ids: Sequence[int]
) -> dict[int, float]:
    shares = exchange_shares(conn, reputation.people, exchange_ids)
    return {
        eid: exchange_score(reputation, share, reputation.evidence.contribution(eid))
        for eid, share in shares.items()
    }
