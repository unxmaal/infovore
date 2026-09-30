import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.label_events import (
    drop_last_label_event,
    human_label_events,
    self_consistency,
)
from infovore.db.message_labels import set_message_label
from infovore.rows import MessageLabel, MessageLabelSource

ROUND_ONE = datetime(2026, 1, 1, tzinfo=UTC)
ROUND_TWO = datetime(2026, 2, 1, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "test.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    for message_id in (10, 11, 12):
        connection.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 1, 1, 'author', ?, 'hello', ?, '{}')",
            (message_id, ROUND_ONE.isoformat(), ROUND_ONE.isoformat()),
        )
    return connection


def _label(
    conn: sqlite3.Connection,
    message_id: int,
    label: MessageLabel,
    source_ref: str,
    at: datetime,
    source: MessageLabelSource = MessageLabelSource.HUMAN,
) -> None:
    set_message_label(conn, message_id, label, source, source_ref, at)


def test_labeling_appends_an_event(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, "round:1", ROUND_ONE)

    events = human_label_events(conn, 10)

    assert [(event.label, event.source_ref) for event in events] == [(MessageLabel.KEEP, "round:1")]


def test_relabeling_appends_rather_than_replacing(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, "round:1", ROUND_ONE)
    _label(conn, 10, MessageLabel.TRASH, "round:2", ROUND_TWO)

    events = human_label_events(conn, 10)

    assert [(event.label, event.source_ref) for event in events] == [
        (MessageLabel.KEEP, "round:1"),
        (MessageLabel.TRASH, "round:2"),
    ]


def test_message_labels_keeps_one_row_per_source(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, "round:1", ROUND_ONE)
    _label(conn, 10, MessageLabel.TRASH, "round:2", ROUND_TWO)

    rows = conn.execute(
        "SELECT label, source_ref FROM message_labels WHERE message_id = 10 AND source = 'human'"
    ).fetchall()

    assert [(row["label"], row["source_ref"]) for row in rows] == [("trash", "round:2")]


def test_machine_labels_are_not_evented(conn: sqlite3.Connection) -> None:
    _label(
        conn,
        10,
        MessageLabel.TRASH,
        "citation",
        ROUND_ONE,
        source=MessageLabelSource.CITATION,
    )

    assert human_label_events(conn, 10) == ()


def test_self_consistency_is_unmeasurable_without_repeats(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, "round:1", ROUND_ONE)
    _label(conn, 11, MessageLabel.TRASH, "round:1", ROUND_ONE)

    report = self_consistency(conn)

    assert report.repeated == 0
    assert report.agreed == 0
    assert report.rate is None


def test_self_consistency_counts_agreement_across_rounds(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, "round:1", ROUND_ONE)
    _label(conn, 10, MessageLabel.KEEP, "round:2", ROUND_TWO)
    _label(conn, 11, MessageLabel.KEEP, "round:1", ROUND_ONE)
    _label(conn, 11, MessageLabel.TRASH, "round:2", ROUND_TWO)

    report = self_consistency(conn)

    assert report.repeated == 2
    assert report.agreed == 1
    assert report.rate == pytest.approx(0.5)


def test_a_repeat_within_one_round_is_not_a_repeat(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, "round:1", ROUND_ONE)
    _label(conn, 10, MessageLabel.TRASH, "round:1", ROUND_TWO)

    report = self_consistency(conn)

    assert report.repeated == 0
    assert report.rate is None


def test_self_consistency_compares_first_and_last_round(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, "round:1", ROUND_ONE)
    _label(conn, 10, MessageLabel.TRASH, "round:2", ROUND_TWO)
    _label(
        conn,
        10,
        MessageLabel.KEEP,
        "round:3",
        datetime(2026, 3, 1, tzinfo=UTC),
    )

    report = self_consistency(conn)

    assert report.repeated == 1
    assert report.agreed == 1
    assert report.rate == pytest.approx(1.0)


def test_dropping_the_last_event_retracts_that_judgment(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, "round:1", ROUND_ONE)
    _label(conn, 10, MessageLabel.TRASH, "round:2", ROUND_TWO)

    drop_last_label_event(conn, 10)

    assert [event.source_ref for event in human_label_events(conn, 10)] == ["round:1"]
    assert self_consistency(conn).repeated == 0


def test_dropping_an_event_for_an_unlabeled_message_is_a_no_op(conn: sqlite3.Connection) -> None:
    drop_last_label_event(conn, 11)

    assert human_label_events(conn, 11) == ()


def test_dropping_events_leaves_other_messages_alone(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, "round:1", ROUND_ONE)
    _label(conn, 11, MessageLabel.KEEP, "round:1", ROUND_ONE)

    drop_last_label_event(conn, 10)

    assert human_label_events(conn, 10) == ()
    assert len(human_label_events(conn, 11)) == 1
