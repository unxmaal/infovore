import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import load_migrations, migrate, open_database
from infovore.db.label_regime import regime_for_source_ref
from infovore.db.message_labels import (
    effective_message_labels_with_source,
    human_labeled_message_ids,
    set_message_label,
)
from infovore.rows import LabelRegime, MessageLabel, MessageLabelSource

AT = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    for message_id in (10, 11, 12):
        connection.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 1, 1, 'a', ?, 'hello', ?, '{}')",
            (message_id, AT.isoformat(), AT.isoformat()),
        )
    return connection


def _label(
    conn: sqlite3.Connection,
    message_id: int,
    label: MessageLabel,
    regime: LabelRegime,
    ref: str = "round:1",
) -> None:
    set_message_label(conn, message_id, label, MessageLabelSource.HUMAN, ref, AT, regime=regime)


def test_the_regime_is_recorded_on_the_label(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, LabelRegime.CONTEXT)

    row = conn.execute("SELECT regime FROM message_labels WHERE message_id = 10").fetchone()

    assert row["regime"] == LabelRegime.CONTEXT.value


def test_the_regime_is_recorded_on_the_event(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, LabelRegime.ISOLATED)

    row = conn.execute("SELECT regime FROM label_events WHERE message_id = 10").fetchone()

    assert row["regime"] == LabelRegime.ISOLATED.value


def test_training_labels_default_to_the_context_regime_only(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, LabelRegime.CONTEXT)
    _label(conn, 11, MessageLabel.TRASH, LabelRegime.ISOLATED)

    labels = effective_message_labels_with_source(conn)

    assert set(labels) == {10}


def test_an_isolated_label_can_be_asked_for_explicitly(conn: sqlite3.Connection) -> None:
    _label(conn, 10, MessageLabel.KEEP, LabelRegime.CONTEXT)
    _label(conn, 11, MessageLabel.TRASH, LabelRegime.ISOLATED)

    labels = effective_message_labels_with_source(conn, regimes=frozenset(LabelRegime))

    assert set(labels) == {10, 11}


def test_machine_labels_survive_the_regime_filter(conn: sqlite3.Connection) -> None:
    set_message_label(conn, 12, MessageLabel.TRASH, MessageLabelSource.CITATION, "citations", AT)

    labels = effective_message_labels_with_source(conn)

    assert set(labels) == {12}


def test_a_retired_label_still_blocks_re_offering_the_message(conn: sqlite3.Connection) -> None:
    _label(conn, 11, MessageLabel.TRASH, LabelRegime.ISOLATED)

    assert 11 in human_labeled_message_ids(conn)


def test_the_pre_context_batches_map_to_isolated() -> None:
    for ref in (
        "sift-serve:batch-002",
        "sift-serve:batch-003",
        "sift:2026-09-27T22:21:27.583979+00:00:a",
    ):
        assert regime_for_source_ref(ref) is LabelRegime.ISOLATED


def test_the_context_batches_map_to_context() -> None:
    for ref in ("sift-serve:batch-004", "sift-serve:batch-005"):
        assert regime_for_source_ref(ref) is LabelRegime.CONTEXT


def test_an_unknown_batch_is_assumed_to_have_context() -> None:
    assert regime_for_source_ref("sift-serve:batch-099") is LabelRegime.CONTEXT


def test_the_live_database_is_backfilled_consistently(conn: sqlite3.Connection) -> None:
    for message_id, ref in ((10, "sift-serve:batch-003"), (11, "sift-serve:batch-005")):
        conn.execute(
            "INSERT INTO message_labels (message_id, label, source, source_ref, labeled_at,"
            " regime) VALUES (?, 'keep', 'human', ?, ?, ?)",
            (message_id, ref, AT.isoformat(), regime_for_source_ref(ref).value),
        )

    rows = dict(conn.execute("SELECT message_id, regime FROM message_labels").fetchall())

    assert rows[10] == LabelRegime.ISOLATED.value
    assert rows[11] == LabelRegime.CONTEXT.value


def test_a_repeats_earlier_event_keeps_its_own_regime(conn: sqlite3.Connection) -> None:
    """A message judged in an isolated round and again in a context round has
    one event of each. Tagging both from the current label would erase the
    boundary the regime exists to mark (issue #170)."""
    set_message_label(
        conn,
        10,
        MessageLabel.TRASH,
        MessageLabelSource.HUMAN,
        "sift-serve:batch-002",
        AT,
    )
    set_message_label(
        conn,
        10,
        MessageLabel.KEEP,
        MessageLabelSource.HUMAN,
        "sift-serve:batch-005",
        AT,
    )

    regimes = [
        row["regime"]
        for row in conn.execute("SELECT regime FROM label_events WHERE message_id = 10 ORDER BY id")
    ]

    assert regimes == [LabelRegime.ISOLATED.value, LabelRegime.CONTEXT.value]


def test_the_backfill_recovers_each_rounds_own_regime(tmp_path: Path) -> None:
    """The 0017 backfill must read each event's own `source_ref`. Deriving it
    from the message's current label retags a repeat's earlier judgment as
    whatever the latest round was, erasing the boundary (issue #170)."""
    all_migrations = load_migrations()
    connection = open_database(tmp_path / "z.db")
    migrate(connection, [m for m in all_migrations if m.version <= 16])
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    connection.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (30, 1, 1, 1, 'a', ?, 'hello', ?, '{}')",
        (AT.isoformat(), AT.isoformat()),
    )
    # judged in an isolated round, then again in a context round
    connection.execute(
        "INSERT INTO message_labels (message_id, label, source, source_ref, labeled_at)"
        " VALUES (30, 'keep', 'human', 'sift-serve:batch-005', ?)",
        (AT.isoformat(),),
    )
    for ref, label in (("sift-serve:batch-002", "trash"), ("sift-serve:batch-005", "keep")):
        connection.execute(
            "INSERT INTO label_events (message_id, label, source_ref, labeled_at)"
            " VALUES (30, ?, ?, ?)",
            (label, ref, AT.isoformat()),
        )

    migrate(connection, [m for m in all_migrations if m.version == 17])

    regimes = [
        row["regime"]
        for row in connection.execute(
            "SELECT regime FROM label_events WHERE message_id = 30 ORDER BY id"
        )
    ]
    assert regimes == [LabelRegime.ISOLATED.value, LabelRegime.CONTEXT.value]


def test_0018_repairs_an_event_regime_taken_from_the_wrong_round(tmp_path: Path) -> None:
    """0017 shipped deriving each event's regime from the message's CURRENT
    label, which mistagged 18 earlier judgments on the live database. 0018
    repairs them from each event's own source_ref and is a no-op where 0017
    already did the right thing."""
    all_migrations = load_migrations()
    connection = open_database(tmp_path / "r.db")
    migrate(connection, [m for m in all_migrations if m.version <= 17])
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    connection.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (40, 1, 1, 1, 'a', ?, 'hello', ?, '{}')",
        (AT.isoformat(), AT.isoformat()),
    )
    connection.execute(
        "INSERT INTO label_events (message_id, label, source_ref, labeled_at, regime)"
        " VALUES (40, 'trash', 'sift-serve:batch-002', ?, 'context')",
        (AT.isoformat(),),
    )
    connection.execute(
        "INSERT INTO label_events (message_id, label, source_ref, labeled_at, regime)"
        " VALUES (40, 'keep', 'sift-serve:batch-005', ?, 'context')",
        (AT.isoformat(),),
    )

    migrate(connection, [m for m in all_migrations if m.version == 18])

    regimes = [
        row["regime"]
        for row in connection.execute(
            "SELECT regime FROM label_events WHERE message_id = 40 ORDER BY id"
        )
    ]
    assert regimes == [LabelRegime.ISOLATED.value, LabelRegime.CONTEXT.value]
