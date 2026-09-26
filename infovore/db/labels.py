import sqlite3
from dataclasses import dataclass
from datetime import datetime

from infovore.db.codec import to_db_time
from infovore.rows import Label, LabelSource


@dataclass(frozen=True)
class LabelCounts:
    by_source: dict[str, dict[str, int]]
    effective: dict[str, int]


def set_label(
    conn: sqlite3.Connection,
    exchange_id: int,
    label: Label,
    source: LabelSource,
    source_ref: str | None,
    at: datetime,
) -> None:
    conn.execute(
        "INSERT INTO exchange_labels (exchange_id, label, source, source_ref, labeled_at)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT (exchange_id, source) DO UPDATE SET"
        " label = excluded.label, source_ref = excluded.source_ref,"
        " labeled_at = excluded.labeled_at",
        (exchange_id, label.value, source.value, source_ref, to_db_time(at)),
    )


def effective_labels(conn: sqlite3.Connection) -> dict[int, Label]:
    result: dict[int, Label] = {}
    llm: dict[int, Label] = {}
    for row in conn.execute("SELECT exchange_id, label, source FROM exchange_labels"):
        if row["source"] == LabelSource.HUMAN.value:
            result[row["exchange_id"]] = Label(row["label"])
        else:
            llm[row["exchange_id"]] = Label(row["label"])
    for exchange_id, label in llm.items():
        result.setdefault(exchange_id, label)
    return result


def label_counts(conn: sqlite3.Connection) -> LabelCounts:
    by_source: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT source, label, COUNT(*) AS n FROM exchange_labels"
        " GROUP BY source, label ORDER BY source, label"
    ):
        by_source.setdefault(row["source"], {})[row["label"]] = row["n"]
    effective_counts: dict[str, int] = {}
    for label in effective_labels(conn).values():
        effective_counts[label.value] = effective_counts.get(label.value, 0) + 1
    return LabelCounts(by_source=by_source, effective=dict(sorted(effective_counts.items())))
