import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.eval.judge import (
    FACT,
    JUDGE_INTERFACE_VERSION,
    JUDGE_SCORER,
    NO_FACT,
    NotInExchangeError,
    QueueItem,
    UnknownQueueItemError,
    exchange_view,
    fact_counts,
    judging_queue,
    progress,
    self_agreement,
    submit,
)
from infovore.eval.slices import GOLD, GOLD_REPEATS

AT = datetime(2026, 10, 2, tzinfo=UTC)


def _exchange(
    conn: sqlite3.Connection, exchange_id: int, message_ids: list[int], parent: int | None = None
) -> None:
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash, parent_exchange_id)"
        " VALUES (?, 1, ?, ?, ?, ?, ?, 'quiet_gap', ?, ?)",
        (
            exchange_id,
            message_ids[0],
            message_ids[-1],
            AT.isoformat(),
            AT.isoformat(),
            len(message_ids),
            f"h{exchange_id}",
            parent,
        ),
    )
    for position, message_id in enumerate(message_ids, start=1):
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 9, 5, 'hal', ?, ?, ?, '{}')",
            (
                message_id,
                (AT + timedelta(minutes=message_id)).isoformat(),
                f"msg {message_id}",
                AT.isoformat(),
            ),
        )
        conn.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, ?)",
            (exchange_id, message_id, position),
        )


def _slice(conn: sqlite3.Connection, name: str, ids: list[int]) -> None:
    for position, exchange_id in enumerate(ids, start=1):
        conn.execute(
            "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
            " VALUES (?, ?, ?, 'test', 0, ?)",
            (name, exchange_id, position, AT.isoformat()),
        )


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 9, NULL, 'hardware', 'text')"
    )
    _exchange(connection, 1, [11, 12, 13])
    _exchange(connection, 2, [21, 22], parent=1)
    _exchange(connection, 3, [31])
    _slice(connection, GOLD, [1, 2, 3])
    _slice(connection, GOLD_REPEATS, [1])
    return connection


def test_the_queue_is_the_gold_set_then_the_repeats(conn: sqlite3.Connection) -> None:
    assert judging_queue(conn) == [
        QueueItem(1, 1),
        QueueItem(2, 1),
        QueueItem(3, 1),
        QueueItem(1, 2),
    ]


def test_a_repeat_is_judged_independently_of_its_first_pass(conn: sqlite3.Connection) -> None:
    """The second showing must not arrive pre-marked, or it measures memory
    of the page rather than the judgment."""
    submit(conn, 0, [12], AT)

    assert exchange_view(conn, 0).marked == {12}
    assert exchange_view(conn, 3).marked == frozenset()
    assert not exchange_view(conn, 3).done


def test_submitting_records_every_message_explicitly(conn: sqlite3.Connection) -> None:
    written = submit(conn, 0, [12], AT)

    rows = conn.execute(
        "SELECT subject_id, label, scorer_version, reproducibility FROM annotations"
        " WHERE scorer = ? ORDER BY subject_id",
        (JUDGE_SCORER,),
    ).fetchall()
    assert written == 3
    assert [(r["subject_id"], r["label"]) for r in rows] == [
        (11, NO_FACT),
        (12, FACT),
        (13, NO_FACT),
    ]
    assert {r["scorer_version"] for r in rows} == {JUDGE_INTERFACE_VERSION}
    assert {r["reproducibility"] for r in rows} == {"recorded"}


def test_an_unopened_exchange_stays_unjudged(conn: sqlite3.Connection) -> None:
    """The sift import defect: a message nobody considered became 'trash' by
    default. Here nothing is recorded until the exchange is submitted."""
    submit(conn, 0, [12], AT)

    assert progress(conn) == (1, 4)
    assert not exchange_view(conn, 1).done


def test_changing_a_judgment_appends_and_the_newest_wins(conn: sqlite3.Connection) -> None:
    submit(conn, 0, [12], AT)
    submit(conn, 0, [11, 13], AT + timedelta(minutes=1))

    assert exchange_view(conn, 0).marked == {11, 13}
    assert (
        conn.execute(
            "SELECT COUNT(*) AS n FROM annotations WHERE scorer = ?", (JUDGE_SCORER,)
        ).fetchone()["n"]
        == 6
    )


def test_a_message_outside_the_exchange_is_refused(conn: sqlite3.Connection) -> None:
    with pytest.raises(NotInExchangeError):
        submit(conn, 0, [21], AT)
    assert progress(conn) == (0, 4)


def test_an_index_off_the_queue_is_refused(conn: sqlite3.Connection) -> None:
    with pytest.raises(UnknownQueueItemError):
        exchange_view(conn, 4)
    with pytest.raises(UnknownQueueItemError):
        submit(conn, -1, [], AT)


def test_context_from_the_parent_is_shown_but_marked_as_context(conn: sqlite3.Connection) -> None:
    view = exchange_view(conn, 1)

    assert [(m.id, m.is_context) for m in view.messages] == [
        (11, True),
        (12, True),
        (13, True),
        (21, False),
        (22, False),
    ]


def test_context_cannot_be_marked_as_a_fact(conn: sqlite3.Connection) -> None:
    """Context belongs to another exchange; marking it here would credit
    this exchange with a fact it does not contain."""
    with pytest.raises(NotInExchangeError):
        submit(conn, 1, [11], AT)


def test_the_view_carries_channel_link_and_position(conn: sqlite3.Connection) -> None:
    view = exchange_view(conn, 2)

    assert (view.index, view.total, view.exchange_id, view.channel) == (2, 4, 3, "hardware")
    assert view.messages[0].link == "https://discord.com/channels/9/1/31"
    assert view.messages[0].author == "hal"


def test_self_agreement_compares_the_two_passes_per_message(conn: sqlite3.Connection) -> None:
    submit(conn, 0, [12], AT)
    submit(conn, 3, [12, 13], AT)

    agreement = self_agreement(conn)

    assert (agreement.exchanges, agreement.messages, agreement.agreed) == (1, 3, 2)
    assert agreement.rate == pytest.approx(2 / 3)


def test_self_agreement_is_undefined_until_a_repeat_has_both_passes(
    conn: sqlite3.Connection,
) -> None:
    submit(conn, 0, [12], AT)

    assert self_agreement(conn).rate is None


def test_fact_counts_use_the_first_pass_only(conn: sqlite3.Connection) -> None:
    """Repeats measure consistency; counting them would double-weight the
    repeated exchanges."""
    submit(conn, 0, [12], AT)
    submit(conn, 3, [11, 12, 13], AT)

    assert fact_counts(conn) == (1, 2)


def test_a_missing_channel_falls_back_to_its_id(tmp_path: Path) -> None:
    connection = open_database(tmp_path / "y.db")
    migrate(connection)
    connection.execute("PRAGMA foreign_keys = OFF")
    _exchange(connection, 1, [11])
    _slice(connection, GOLD, [1])

    assert exchange_view(connection, 0).channel == "1"


def test_an_exchange_with_no_messages_is_refused_not_silently_skipped(tmp_path: Path) -> None:
    """Writing nothing would leave the item undone forever, so the page would
    keep offering it."""
    connection = open_database(tmp_path / "z.db")
    migrate(connection)
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (1, 1, 1, 1, ?, ?, 1, 'quiet_gap', 'h')",
        (AT.isoformat(), AT.isoformat()),
    )
    _slice(connection, GOLD, [1])

    with pytest.raises(NotInExchangeError, match="no messages"):
        submit(connection, 0, [], AT)
