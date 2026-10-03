import sqlite3
from collections.abc import Sequence

from infovore.db.batch import _chunked
from infovore.db.exchange_search import SHAREABLE_MESSAGE

_HAS_TEXT = f"TRIM(m.content, ' ' || char(9) || char(10) || char(13)) != '' AND {SHAREABLE_MESSAGE}"
_FROM = "FROM exchange_messages em JOIN messages m ON m.id = em.message_id"


def enough_text_clause(exchange_id_column: str) -> str:
    return (
        f"(SELECT COUNT(*) {_FROM} WHERE em.exchange_id = {exchange_id_column}"
        f" AND {_HAS_TEXT}) >= ?"
    )


def text_message_counts(conn: sqlite3.Connection, exchange_ids: Sequence[int]) -> dict[int, int]:
    counts = {eid: 0 for eid in exchange_ids}
    for chunk in _chunked(list(exchange_ids)):
        marks = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT em.exchange_id AS id, COUNT(*) AS n {_FROM}"
            f" WHERE em.exchange_id IN ({marks}) AND {_HAS_TEXT} GROUP BY em.exchange_id",
            chunk,
        ):
            counts[row["id"]] = row["n"]
    return counts
