import sqlite3
from dataclasses import dataclass
from datetime import datetime

from infovore.db.codec import to_db_time
from infovore.db.label_events import record_label_event
from infovore.db.label_regime import regime_for_source_ref
from infovore.rows import LabelRegime, MessageLabel, MessageLabelSource


@dataclass(frozen=True)
class MessageLabelCounts:
    by_source: dict[str, dict[str, int]]
    effective: dict[str, int]


def set_message_label(
    conn: sqlite3.Connection,
    message_id: int,
    label: MessageLabel,
    source: MessageLabelSource,
    source_ref: str | None,
    at: datetime,
    regime: LabelRegime | None = None,
) -> None:
    """Record one `message_labels` row (issue #128), one per `(message_id,
    source)`: `human` (this PR), `citation`/`rule` (PR 3, the message
    classifier). Re-labeling the same message from the same source replaces
    the earlier row rather than accumulating duplicates, mirroring
    `infovore.db.labels.set_label` for `exchange_labels`."""
    effective = regime if regime is not None else regime_for_source_ref(source_ref)
    conn.execute(
        "INSERT INTO message_labels (message_id, label, source, source_ref, labeled_at, regime)"
        " VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (message_id, source) DO UPDATE SET"
        " label = excluded.label, source_ref = excluded.source_ref,"
        " labeled_at = excluded.labeled_at, regime = excluded.regime",
        (message_id, label.value, source.value, source_ref, to_db_time(at), effective.value),
    )
    if source is MessageLabelSource.HUMAN:
        record_label_event(conn, message_id, label, source_ref, at, effective)


def effective_message_labels(conn: sqlite3.Connection) -> dict[int, MessageLabel]:
    """One label per message, a human label always winning over a weaker
    (`citation`/`rule`) one for the same message, regardless of insert
    order — mirrors `infovore.db.labels.effective_labels`."""
    result: dict[int, MessageLabel] = {}
    weak: dict[int, MessageLabel] = {}
    for row in conn.execute("SELECT message_id, label, source, regime FROM message_labels"):
        if row["source"] == MessageLabelSource.HUMAN.value:
            if row["regime"] == LabelRegime.VALUE.value:
                continue
            result[row["message_id"]] = MessageLabel(row["label"])
        else:
            weak[row["message_id"]] = MessageLabel(row["label"])
    for message_id, label in weak.items():
        result.setdefault(message_id, label)
    return result


def effective_message_labels_with_source(
    conn: sqlite3.Connection,
    regimes: frozenset[LabelRegime] = frozenset({LabelRegime.CONTEXT}),
) -> dict[int, tuple[MessageLabel, MessageLabelSource]]:
    """Like `effective_message_labels`, but also reports which source won.
    `infovore.sift.train` (issue #128 PR 3) needs this to weight a human
    example more heavily than a citation one when training, and to report
    holdout precision/recall/AUC separately per source (citation labels are
    noisy -- uncited does not mean trash -- while human labels are the
    ground truth that matters)."""
    result: dict[int, tuple[MessageLabel, MessageLabelSource]] = {}
    weak: dict[int, tuple[MessageLabel, MessageLabelSource]] = {}
    wanted = {regime.value for regime in regimes}
    for row in conn.execute("SELECT message_id, label, source, regime FROM message_labels"):
        source = MessageLabelSource(row["source"])
        pair = (MessageLabel(row["label"]), source)
        if source is MessageLabelSource.HUMAN:
            if row["regime"] not in wanted:
                continue
            result[row["message_id"]] = pair
        else:
            weak[row["message_id"]] = pair
    for message_id, pair in weak.items():
        result.setdefault(message_id, pair)
    return result


def human_labeled_message_ids(conn: sqlite3.Connection) -> frozenset[int]:
    """Every message with a `human` label — a sift batch never re-offers a
    message the maintainer has already hand-labeled."""
    rows = conn.execute(
        "SELECT message_id FROM message_labels WHERE source = ?",
        (MessageLabelSource.HUMAN.value,),
    ).fetchall()
    return frozenset(row["message_id"] for row in rows)


def message_label_counts(conn: sqlite3.Connection) -> MessageLabelCounts:
    by_source: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT source, label, COUNT(*) AS n FROM message_labels"
        " GROUP BY source, label ORDER BY source, label"
    ):
        by_source.setdefault(row["source"], {})[row["label"]] = row["n"]
    effective_counts: dict[str, int] = {}
    for label in effective_message_labels(conn).values():
        effective_counts[label.value] = effective_counts.get(label.value, 0) + 1
    return MessageLabelCounts(by_source=by_source, effective=dict(sorted(effective_counts.items())))
