import sqlite3
from dataclasses import dataclass
from datetime import datetime

from infovore.db.codec import to_db_time
from infovore.rows import MessageLabel


@dataclass(frozen=True)
class LabelEvent:
    label: MessageLabel
    source_ref: str | None
    labeled_at: str


@dataclass(frozen=True)
class SelfConsistency:
    """The ceiling on any technique's agreement with Eric: how often he agrees
    with his own earlier judgment on the same message in a later round."""

    repeated: int
    agreed: int

    @property
    def rate(self) -> float | None:
        if self.repeated == 0:
            return None
        return self.agreed / self.repeated


def record_label_event(
    conn: sqlite3.Connection,
    message_id: int,
    label: MessageLabel,
    source_ref: str | None,
    at: datetime,
) -> None:
    conn.execute(
        "INSERT INTO label_events (message_id, label, source_ref, labeled_at) VALUES (?, ?, ?, ?)",
        (message_id, label.value, source_ref, to_db_time(at)),
    )


def drop_last_label_event(conn: sqlite3.Connection, message_id: int) -> None:
    """Undo retracts the judgment, so the event log must lose it too or a
    withdrawn call would still count toward the self-consistency ceiling."""
    conn.execute(
        "DELETE FROM label_events WHERE id = ("
        "   SELECT id FROM label_events WHERE message_id = ? ORDER BY id DESC LIMIT 1"
        " )",
        (message_id,),
    )


def human_label_events(conn: sqlite3.Connection, message_id: int) -> tuple[LabelEvent, ...]:
    rows = conn.execute(
        "SELECT label, source_ref, labeled_at FROM label_events WHERE message_id = ? ORDER BY id",
        (message_id,),
    ).fetchall()
    return tuple(
        LabelEvent(
            label=MessageLabel(row["label"]),
            source_ref=row["source_ref"],
            labeled_at=row["labeled_at"],
        )
        for row in rows
    )


def self_consistency(conn: sqlite3.Connection) -> SelfConsistency:
    rows = conn.execute(
        "SELECT message_id, label, source_ref FROM label_events ORDER BY message_id, id"
    ).fetchall()
    by_message: dict[int, list[tuple[str | None, str]]] = {}
    for row in rows:
        by_message.setdefault(row["message_id"], []).append((row["source_ref"], row["label"]))
    repeated = agreed = 0
    for events in by_message.values():
        first_ref, first_label = events[0]
        later = [(ref, label) for ref, label in events if ref != first_ref]
        if not later:
            continue
        repeated += 1
        if later[-1][1] == first_label:
            agreed += 1
    return SelfConsistency(repeated=repeated, agreed=agreed)
