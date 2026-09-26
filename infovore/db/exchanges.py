import sqlite3
from collections.abc import Sequence
from typing import cast

from infovore.db.codec import from_db_time, to_db_time
from infovore.db.connection import transaction
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule


class DuplicateExchangeError(Exception):
    def __init__(self, content_hash: str) -> None:
        super().__init__(f"exchange with content_hash {content_hash!r} already exists")
        self.content_hash = content_hash


class MessageAlreadyGroupedError(Exception):
    def __init__(self, message_id: int) -> None:
        super().__init__(f"message {message_id} is already part of another exchange")
        self.message_id = message_id


def _row_to_exchange(row: sqlite3.Row) -> ExchangeRow:
    return ExchangeRow(
        id=row["id"],
        channel_id=row["channel_id"],
        thread_id=row["thread_id"],
        first_message_id=row["first_message_id"],
        last_message_id=row["last_message_id"],
        started_at=from_db_time(row["started_at"]),
        ended_at=from_db_time(row["ended_at"]),
        message_count=row["message_count"],
        grouping_rule=GroupingRule(row["grouping_rule"]),
        content_hash=row["content_hash"],
        parent_exchange_id=row["parent_exchange_id"],
        extraction_status=ExtractionStatus(row["extraction_status"]),
        retry_count=row["retry_count"],
        last_error=row["last_error"],
        triage_score=row["triage_score"],
        triage_reasons=row["triage_reasons"],
        triage_version=row["triage_version"],
    )


def insert_exchange(
    conn: sqlite3.Connection, exchange: ExchangeRow, message_ids: Sequence[int]
) -> int:
    if not message_ids or len(message_ids) != exchange.message_count:
        raise ValueError("message_ids must be non-empty and match exchange.message_count")
    with transaction(conn):
        duplicate = conn.execute(
            "SELECT 1 FROM exchanges WHERE content_hash = ?", (exchange.content_hash,)
        ).fetchone()
        if duplicate is not None:
            raise DuplicateExchangeError(exchange.content_hash)
        placeholders = ", ".join("?" * len(message_ids))
        already_grouped = {
            row["message_id"]
            for row in conn.execute(
                f"SELECT message_id FROM exchange_messages WHERE message_id IN ({placeholders})",
                tuple(message_ids),
            ).fetchall()
        }
        if already_grouped:
            offender = next(
                message_id for message_id in message_ids if message_id in already_grouped
            )
            raise MessageAlreadyGroupedError(offender)
        cursor = conn.execute(
            "INSERT INTO exchanges (channel_id, thread_id, first_message_id,"
            " last_message_id, started_at, ended_at, message_count, grouping_rule,"
            " content_hash, parent_exchange_id, extraction_status, retry_count,"
            " last_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exchange.channel_id,
                exchange.thread_id,
                exchange.first_message_id,
                exchange.last_message_id,
                to_db_time(exchange.started_at),
                to_db_time(exchange.ended_at),
                exchange.message_count,
                exchange.grouping_rule.value,
                exchange.content_hash,
                exchange.parent_exchange_id,
                exchange.extraction_status.value,
                exchange.retry_count,
                exchange.last_error,
            ),
        )
        exchange_id = cast(int, cursor.lastrowid)
        for position, message_id in enumerate(message_ids):
            conn.execute(
                "INSERT INTO exchange_messages (exchange_id, message_id, position)"
                " VALUES (?, ?, ?)",
                (exchange_id, message_id, position),
            )
    return exchange_id


def get_exchange(conn: sqlite3.Connection, id: int) -> ExchangeRow | None:
    row = conn.execute("SELECT * FROM exchanges WHERE id = ?", (id,)).fetchone()
    if row is None:
        return None
    return _row_to_exchange(row)


def exchange_message_ids(conn: sqlite3.Connection, id: int) -> list[int]:
    rows = conn.execute(
        "SELECT message_id FROM exchange_messages WHERE exchange_id = ? ORDER BY position",
        (id,),
    ).fetchall()
    return [row["message_id"] for row in rows]


def exchange_for_message(conn: sqlite3.Connection, message_id: int) -> int | None:
    row = conn.execute(
        "SELECT exchange_id FROM exchange_messages WHERE message_id = ?", (message_id,)
    ).fetchone()
    if row is None:
        return None
    return int(row["exchange_id"])


def grouped_message_ids(conn: sqlite3.Connection) -> set[int]:
    rows = conn.execute("SELECT message_id FROM exchange_messages").fetchall()
    return {row["message_id"] for row in rows}


def claimable_exchanges(
    conn: sqlite3.Connection, limit: int, max_retries: int, min_score: float | None = None
) -> list[ExchangeRow]:
    condition = "extraction_status IN (?, ?) AND retry_count < ?"
    params: list[object] = [
        ExtractionStatus.PENDING.value,
        ExtractionStatus.STALE.value,
        max_retries,
    ]
    if min_score is not None:
        condition += " AND triage_score >= ?"
        params.append(min_score)
    rows = conn.execute(
        f"SELECT * FROM exchanges WHERE {condition} ORDER BY started_at, id LIMIT ?",
        (*params, limit),
    ).fetchall()
    return [_row_to_exchange(row) for row in rows]


def has_untriaged_claimable(
    conn: sqlite3.Connection, current_version: str, max_retries: int
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM exchanges WHERE extraction_status IN (?, ?) AND retry_count < ?"
        " AND (triage_version IS NULL OR triage_version != ?) LIMIT 1",
        (
            ExtractionStatus.PENDING.value,
            ExtractionStatus.STALE.value,
            max_retries,
            current_version,
        ),
    ).fetchone()
    return row is not None


def set_status(
    conn: sqlite3.Connection, id: int, status: ExtractionStatus, last_error: str | None = None
) -> None:
    conn.execute(
        "UPDATE exchanges SET extraction_status = ?, last_error = ? WHERE id = ?",
        (status.value, last_error, id),
    )


def record_failure(conn: sqlite3.Connection, id: int, error: str, max_retries: int) -> None:
    conn.execute(
        "UPDATE exchanges SET retry_count = retry_count + 1, last_error = ?,"
        " extraction_status = CASE WHEN retry_count + 1 >= ? THEN ? ELSE extraction_status END"
        " WHERE id = ?",
        (error, max_retries, ExtractionStatus.FAILED.value, id),
    )


def mark_stale_for_message(conn: sqlite3.Connection, message_id: int) -> int | None:
    row = conn.execute(
        "SELECT e.id AS id, e.extraction_status AS extraction_status FROM exchange_messages em"
        " JOIN exchanges e ON e.id = em.exchange_id WHERE em.message_id = ?",
        (message_id,),
    ).fetchone()
    if row is None or row["extraction_status"] != ExtractionStatus.DONE.value:
        return None
    exchange_id = int(row["id"])
    conn.execute(
        "UPDATE exchanges SET extraction_status = ? WHERE id = ?",
        (ExtractionStatus.STALE.value, exchange_id),
    )
    return exchange_id
