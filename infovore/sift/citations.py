"""Citation-derived message labels (issue #128 PR 3).

Weak, free ground truth for the message classifier: a message in an exchange
that has at least one successfully extracted (`mode='live'`, `outcome='ok'`)
extraction run is `keep` if any non-retracted claim from that run (or any
other `ok` live run of the same exchange) cites it via `claim_sources`, else
`trash`. Exchanges that were never successfully extracted live are left
alone entirely -- an unprocessed exchange's messages get no citation label,
since "nothing cited it" there just means "no extraction ever looked", not
"this is trash".

Human labels always win over these at read time
(`infovore.db.message_labels.effective_message_labels`/
`effective_message_labels_with_source`); this module only ever writes
`source='citation'` rows, so it never clobbers a `sift import` hand label.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from infovore.db.batch import SQLITE_MAX_VARIABLES
from infovore.db.codec import to_db_time
from infovore.db.connection import transaction
from infovore.rows import MessageLabel, MessageLabelSource


@dataclass(frozen=True)
class CitationLabelReport:
    exchanges_considered: int
    keep: int
    trash: int


def _chunked(ids: Sequence[int], size: int = SQLITE_MAX_VARIABLES) -> list[list[int]]:
    return [list(ids[start : start + size]) for start in range(0, len(ids), size)]


def derive_citation_labels(conn: sqlite3.Connection, at: datetime) -> CitationLabelReport:
    exchange_ids = [
        row["exchange_id"]
        for row in conn.execute(
            "SELECT DISTINCT exchange_id FROM extraction_runs"
            " WHERE mode = 'live' AND outcome = 'ok'"
        )
    ]
    at_text = to_db_time(at)
    keep_total = 0
    trash_total = 0

    for chunk in _chunked(exchange_ids):
        placeholders = ",".join("?" for _ in chunk)

        member_rows = conn.execute(
            f"SELECT exchange_id, message_id FROM exchange_messages"
            f" WHERE exchange_id IN ({placeholders})",
            chunk,
        ).fetchall()
        members_by_exchange: dict[int, list[int]] = {}
        for row in member_rows:
            members_by_exchange.setdefault(row["exchange_id"], []).append(row["message_id"])

        cited_rows = conn.execute(
            f"SELECT DISTINCT c.exchange_id AS exchange_id, cs.message_id AS message_id"
            f" FROM claims c"
            f" JOIN extraction_runs r ON r.id = c.extraction_run_id"
            f" JOIN claim_sources cs ON cs.claim_id = c.id"
            f" WHERE c.exchange_id IN ({placeholders}) AND r.mode = 'live' AND r.outcome = 'ok'"
            f" AND c.retracted_at IS NULL",
            chunk,
        ).fetchall()
        cited_by_exchange: dict[int, set[int]] = {}
        for row in cited_rows:
            cited_by_exchange.setdefault(row["exchange_id"], set()).add(row["message_id"])

        updates: list[tuple[int, str, str, str, str]] = []
        for exchange_id in chunk:
            cited = cited_by_exchange.get(exchange_id, set())
            for message_id in members_by_exchange.get(exchange_id, []):
                label = MessageLabel.KEEP if message_id in cited else MessageLabel.TRASH
                if label is MessageLabel.KEEP:
                    keep_total += 1
                else:
                    trash_total += 1
                updates.append(
                    (
                        message_id,
                        label.value,
                        MessageLabelSource.CITATION.value,
                        f"citation:{exchange_id}",
                        at_text,
                    )
                )

        with transaction(conn):
            conn.executemany(
                "INSERT INTO message_labels (message_id, label, source, source_ref, labeled_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT (message_id, source) DO UPDATE SET"
                " label = excluded.label, source_ref = excluded.source_ref,"
                " labeled_at = excluded.labeled_at",
                updates,
            )

    return CitationLabelReport(
        exchanges_considered=len(exchange_ids), keep=keep_total, trash=trash_total
    )
