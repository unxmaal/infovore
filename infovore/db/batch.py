"""Batched, set-based loading of exchange inputs (issue #104).

`infovore/triage/runner.py` and `infovore/triage/train.py` both need, per
exchange: its messages (in position order), reactions, and attachments --
previously 4 queries per exchange (`exchange_message_ids`, `messages_by_ids`,
`reactions_for_messages`, `attachments_for_messages`), i.e. ~800k round trips
for a full rescore of the real DB. `exchange_inputs_for_ids` fetches the same
data for many exchanges at once, in a handful of set-based queries.

It returns exactly what calling the per-exchange loaders would have: message
order, missing-message filtering, and reaction/attachment ordering all match
(see the docstrings on each loader in `infovore/db/raw.py` and
`infovore/db/exchanges.py` for the exact semantics being preserved).
"""

import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from infovore.db.raw import message_from_row
from infovore.rows import AttachmentRow, MessageRow, ReactionRow

# SQLite's compiled default limit on bound parameters in one statement
# (SQLITE_MAX_VARIABLE_NUMBER, 999 on modern builds). An earlier bug in this
# repo blew past it with a long `IN (...)` list, so every batched query here
# chunks its id list to stay comfortably under the limit.
SQLITE_MAX_VARIABLES = 900

# How many exchanges to score/write per commit. Shared by runner.py and
# train.py so a full rescore commits O(exchanges / BATCH_SIZE) times instead
# of once per exchange, while keeping a crash mid-run leaving only fully
# committed batches behind (resume is still "rows whose triage_version, or
# p_lore_model, differs").
BATCH_SIZE = 2000


@dataclass(frozen=True)
class ExchangeInputs:
    """The scoring inputs for one exchange, in the same shape and order the
    per-exchange loaders (`exchange_message_ids` + `messages_by_ids` +
    `reactions_for_messages` + `attachments_for_messages`) produced."""

    messages: list[MessageRow]
    reactions: list[ReactionRow]
    attachments: list[AttachmentRow]


def _chunked(ids: Sequence[int], size: int = SQLITE_MAX_VARIABLES) -> Iterator[Sequence[int]]:
    for start in range(0, len(ids), size):
        yield ids[start : start + size]


def exchange_inputs_for_ids(
    conn: sqlite3.Connection, exchange_ids: Sequence[int]
) -> dict[int, ExchangeInputs]:
    ids = list(exchange_ids)
    if not ids:
        return {}

    message_ids_by_exchange: dict[int, list[int]] = {exchange_id: [] for exchange_id in ids}
    for chunk in _chunked(ids):
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT exchange_id, message_id FROM all_exchange_messages"
            f" WHERE exchange_id IN ({placeholders}) ORDER BY exchange_id, position",
            chunk,
        ).fetchall()
        for row in rows:
            message_ids_by_exchange[row["exchange_id"]].append(row["message_id"])

    all_message_ids = [
        message_id for exchange_id in ids for message_id in message_ids_by_exchange[exchange_id]
    ]

    messages_by_id: dict[int, MessageRow] = {}
    reactions_by_message: dict[int, list[ReactionRow]] = {}
    attachments_by_message: dict[int, list[AttachmentRow]] = {}
    for chunk in _chunked(all_message_ids):
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(f"SELECT * FROM messages WHERE id IN ({placeholders})", chunk):
            messages_by_id[row["id"]] = message_from_row(row)
        for row in conn.execute(
            f"SELECT message_id, emoji, count FROM reactions WHERE message_id IN ({placeholders})",
            chunk,
        ):
            reactions_by_message.setdefault(row["message_id"], []).append(
                ReactionRow(message_id=row["message_id"], emoji=row["emoji"], count=row["count"])
            )
        for row in conn.execute(
            "SELECT id, message_id, filename, content_type, size, url, sha256, local_path"
            f" FROM attachments WHERE message_id IN ({placeholders})",
            chunk,
        ):
            attachments_by_message.setdefault(row["message_id"], []).append(
                AttachmentRow(
                    id=row["id"],
                    message_id=row["message_id"],
                    filename=row["filename"],
                    content_type=row["content_type"],
                    size=row["size"],
                    url=row["url"],
                    sha256=row["sha256"],
                    local_path=row["local_path"],
                )
            )

    result: dict[int, ExchangeInputs] = {}
    for exchange_id in ids:
        message_ids = message_ids_by_exchange[exchange_id]
        messages = [
            messages_by_id[message_id] for message_id in message_ids if message_id in messages_by_id
        ]
        reactions = sorted(
            (
                reaction
                for message_id in message_ids
                for reaction in reactions_by_message.get(message_id, [])
            ),
            key=lambda reaction: (reaction.message_id, reaction.emoji),
        )
        attachments = sorted(
            (
                attachment
                for message_id in message_ids
                for attachment in attachments_by_message.get(message_id, [])
            ),
            key=lambda attachment: (attachment.message_id, attachment.id),
        )
        result[exchange_id] = ExchangeInputs(
            messages=messages, reactions=reactions, attachments=attachments
        )
    return result
