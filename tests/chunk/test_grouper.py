import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from infovore.chunk.grouper import (
    ChannelGrouped,
    GroupingEvent,
    GroupingReport,
    GroupingStarted,
    _resolve_parent,
    group_pending,
)
from infovore.chunk.rules import Group, drop_ungroupable
from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import (
    DuplicateExchangeError,
    exchange_message_ids,
    get_exchange,
    grouped_message_ids,
)
from infovore.db.raw import upsert_message
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule, MessageRow
from infovore.timing import FixedClock

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def at(minutes: int) -> datetime:
    return BASE + timedelta(minutes=minutes)


def make_message(id: int, **overrides: object) -> MessageRow:
    fields: dict[str, object] = {
        "id": id,
        "channel_id": 10,
        "guild_id": 100,
        "author_id": 5,
        "author_name_at_time": "alice",
        "author_is_bot": False,
        "created_at": BASE,
        "edited_at": None,
        "content": "hello",
        "reply_to_id": None,
        "thread_id": None,
        "deleted_at": None,
        "ingested_at": BASE,
        "raw_json": "{}",
    }
    fields.update(overrides)
    return MessageRow(**fields)  # type: ignore[arg-type]


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = open_database(":memory:")
    migrate(connection)
    return connection


def seed(conn: sqlite3.Connection, *messages: MessageRow) -> None:
    for message in messages:
        upsert_message(conn, message)


def test_group_pending_persists_closed_group_and_returns_counts(
    conn: sqlite3.Connection,
) -> None:
    seed(conn, make_message(1, created_at=at(0)), make_message(2, created_at=at(1)))
    clock = FixedClock(at(1000))
    report = group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    assert report == GroupingReport(exchanges_created=1, messages_grouped=2, groups_deferred=0)
    assert grouped_message_ids(conn) == {1, 2}


def test_group_pending_defers_open_group(conn: sqlite3.Connection) -> None:
    seed(conn, make_message(1, created_at=at(0)), make_message(2, created_at=at(1)))
    clock = FixedClock(at(2))
    report = group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    assert report == GroupingReport(exchanges_created=0, messages_grouped=0, groups_deferred=1)
    assert grouped_message_ids(conn) == set()


def test_group_pending_persists_exchange_fields(conn: sqlite3.Connection) -> None:
    seed(conn, make_message(2, created_at=at(1)), make_message(1, created_at=at(0)))
    clock = FixedClock(at(1000))
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    ids = exchange_message_ids(conn, 1)
    assert ids == [1, 2]
    exchange = get_exchange(conn, 1)
    assert exchange is not None
    assert exchange.channel_id == 10
    assert exchange.thread_id is None
    assert exchange.first_message_id == 1
    assert exchange.last_message_id == 2
    assert exchange.started_at == at(0)
    assert exchange.ended_at == at(1)
    assert exchange.message_count == 2
    assert exchange.grouping_rule == GroupingRule.QUIET_GAP
    assert exchange.parent_exchange_id is None
    assert exchange.extraction_status == ExtractionStatus.PENDING
    assert exchange.retry_count == 0
    assert exchange.last_error is None
    expected_hash = hashlib.sha256(b"1,2").hexdigest()
    assert exchange.content_hash == expected_hash


def test_group_pending_split_parts_chain_parent_exchange_id(conn: sqlite3.Connection) -> None:
    messages = [make_message(i, created_at=at(i)) for i in range(5)]
    seed(conn, *messages)
    clock = FixedClock(at(100000))
    report = group_pending(conn, clock, quiet_gap=timedelta(minutes=30), max_messages=2)
    assert report.exchanges_created == 3
    assert report.messages_grouped == 5
    part0 = get_exchange(conn, 1)
    part1 = get_exchange(conn, 2)
    part2 = get_exchange(conn, 3)
    assert part0 is not None and part1 is not None and part2 is not None
    assert exchange_message_ids(conn, part0.id) == [0, 1]  # type: ignore[arg-type]
    assert exchange_message_ids(conn, part1.id) == [2, 3]  # type: ignore[arg-type]
    assert exchange_message_ids(conn, part2.id) == [4]  # type: ignore[arg-type]
    assert part0.parent_exchange_id is None
    assert part1.parent_exchange_id == part0.id
    assert part2.parent_exchange_id == part1.id


def test_group_pending_late_reply_sets_parent_to_closed_exchange(
    conn: sqlite3.Connection,
) -> None:
    seed(conn, make_message(1, created_at=at(0)), make_message(2, created_at=at(1)))
    clock = FixedClock(at(1000))
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    first_exchange = get_exchange(conn, 1)
    assert first_exchange is not None

    seed(conn, make_message(3, created_at=at(2000), reply_to_id=1))
    clock.advance(timedelta(minutes=2000))
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    reply_exchange = get_exchange(conn, 2)
    assert reply_exchange is not None
    assert reply_exchange.parent_exchange_id == first_exchange.id
    assert exchange_message_ids(conn, 2) == [3]


def test_group_pending_thread_revival_sets_parent_to_latest_thread_exchange(
    conn: sqlite3.Connection,
) -> None:
    seed(
        conn,
        make_message(1, channel_id=77, thread_id=77, created_at=at(0)),
        make_message(2, channel_id=77, thread_id=77, created_at=at(1)),
    )
    clock = FixedClock(at(1000))
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    first_thread_exchange = get_exchange(conn, 1)
    assert first_thread_exchange is not None
    assert first_thread_exchange.grouping_rule == GroupingRule.THREAD

    seed(
        conn,
        make_message(3, channel_id=77, thread_id=77, created_at=at(2000)),
        make_message(4, channel_id=77, thread_id=77, created_at=at(2001)),
    )
    clock.advance(timedelta(minutes=2000))
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    revived = get_exchange(conn, 2)
    assert revived is not None
    assert revived.grouping_rule == GroupingRule.THREAD
    assert revived.parent_exchange_id == first_thread_exchange.id


def test_group_pending_late_reply_parent_wins_over_thread_revival(
    conn: sqlite3.Connection,
) -> None:
    seed(conn, make_message(100, channel_id=1, created_at=at(-1000)))
    seed(
        conn,
        make_message(1, channel_id=77, thread_id=77, created_at=at(0)),
        make_message(2, channel_id=77, thread_id=77, created_at=at(1)),
    )
    clock = FixedClock(at(1000))
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    unrelated = get_exchange(conn, 1)
    thread_exchange = get_exchange(conn, 2)
    assert unrelated is not None and thread_exchange is not None

    seed(
        conn,
        make_message(3, channel_id=77, thread_id=77, created_at=at(2000), reply_to_id=100),
        make_message(4, channel_id=77, thread_id=77, created_at=at(2001)),
    )
    clock.advance(timedelta(minutes=2000))
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    revived = get_exchange(conn, 3)
    assert revived is not None
    assert revived.grouping_rule == GroupingRule.THREAD
    assert revived.parent_exchange_id == unrelated.id
    assert revived.parent_exchange_id != thread_exchange.id


def test_group_pending_duplicate_content_hash_is_tolerated(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(conn, make_message(1, created_at=at(0)), make_message(2, created_at=at(1)))
    clock = FixedClock(at(1000))

    def fake_insert(
        connection: sqlite3.Connection, exchange: ExchangeRow, message_ids: list[int]
    ) -> int:
        raise DuplicateExchangeError(exchange.content_hash)

    monkeypatch.setattr("infovore.chunk.grouper.insert_exchange", fake_insert)
    report = group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    assert report == GroupingReport(exchanges_created=0, messages_grouped=0, groups_deferred=0)
    assert grouped_message_ids(conn) == set()


def test_group_pending_multiple_channels_processed_independently(
    conn: sqlite3.Connection,
) -> None:
    seed(
        conn,
        make_message(1, channel_id=1, created_at=at(0)),
        make_message(2, channel_id=1, created_at=at(1)),
        make_message(3, channel_id=2, created_at=at(0)),
    )
    clock = FixedClock(at(1000))
    report = group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    assert report.exchanges_created == 2
    assert report.messages_grouped == 3
    exchanges = [get_exchange(conn, i) for i in (1, 2)]
    channel_ids = {exchange.channel_id for exchange in exchanges if exchange is not None}
    assert channel_ids == {1, 2}


def test_group_pending_second_run_is_a_noop(conn: sqlite3.Connection) -> None:
    seed(conn, make_message(1, created_at=at(0)), make_message(2, created_at=at(1)))
    clock = FixedClock(at(1000))
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    report = group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    assert report == GroupingReport(exchanges_created=0, messages_grouped=0, groups_deferred=0)


def test_group_pending_drops_bots_and_deleted_by_default(conn: sqlite3.Connection) -> None:
    seed(
        conn,
        make_message(1, created_at=at(0)),
        make_message(2, created_at=at(1), author_is_bot=True),
        make_message(3, created_at=at(2), deleted_at=at(2)),
    )
    clock = FixedClock(at(1000))
    report = group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    assert report.messages_grouped == 1
    assert grouped_message_ids(conn) == {1}


def test_group_pending_include_bots_flag(conn: sqlite3.Connection) -> None:
    seed(
        conn,
        make_message(1, created_at=at(0)),
        make_message(2, created_at=at(1), author_is_bot=True),
    )
    clock = FixedClock(at(1000))
    report = group_pending(conn, clock, quiet_gap=timedelta(minutes=30), include_bots=True)
    assert report.messages_grouped == 2
    assert grouped_message_ids(conn) == {1, 2}


def test_resolve_parent_falls_through_when_context_message_is_ungrouped(
    conn: sqlite3.Connection,
) -> None:
    context_message = make_message(999, created_at=at(0))
    current = make_message(1000, created_at=at(1))
    group = Group(GroupingRule.QUIET_GAP, (current,), (context_message,))
    assert _resolve_parent(conn, group) is None


def test_group_pending_no_ungrouped_messages_is_a_noop(conn: sqlite3.Connection) -> None:
    clock = FixedClock(at(1000))
    report = group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    assert report == GroupingReport(exchanges_created=0, messages_grouped=0, groups_deferred=0)


def test_group_pending_progress_events_stream_in_order(conn: sqlite3.Connection) -> None:
    seed(
        conn,
        make_message(1, channel_id=1, created_at=at(0)),
        make_message(2, channel_id=1, created_at=at(1)),
        make_message(3, channel_id=2, created_at=at(0)),
    )
    clock = FixedClock(at(1000))
    events: list[GroupingEvent] = []
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30), progress=events.append)
    assert events == [
        GroupingStarted(channels=2),
        ChannelGrouped(channel_id=1, exchanges_created=1, groups_deferred=0),
        ChannelGrouped(channel_id=2, exchanges_created=1, groups_deferred=0),
    ]


def test_group_pending_progress_counts_deferred_groups_per_channel(
    conn: sqlite3.Connection,
) -> None:
    seed(conn, make_message(1, channel_id=1, created_at=at(0)))
    clock = FixedClock(at(1))
    events: list[GroupingEvent] = []
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30), progress=events.append)
    assert events == [
        GroupingStarted(channels=1),
        ChannelGrouped(channel_id=1, exchanges_created=0, groups_deferred=1),
    ]


def test_group_pending_progress_defaults_to_noop(conn: sqlite3.Connection) -> None:
    seed(conn, make_message(1, created_at=at(0)), make_message(2, created_at=at(1)))
    clock = FixedClock(at(1000))
    report = group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    assert report == GroupingReport(exchanges_created=1, messages_grouped=2, groups_deferred=0)


def test_group_pending_split_part_parent_wins_over_late_reply(
    conn: sqlite3.Connection,
) -> None:
    seed(conn, make_message(100, created_at=at(-1000)))
    clock = FixedClock(at(1000))
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30))
    unrelated = get_exchange(conn, 1)
    assert unrelated is not None

    messages = [make_message(i, created_at=at(i), reply_to_id=100) for i in range(5)]
    seed(conn, *messages)
    clock.advance(timedelta(minutes=100000))
    group_pending(conn, clock, quiet_gap=timedelta(minutes=30), max_messages=2)
    part0 = get_exchange(conn, 2)
    part1 = get_exchange(conn, 3)
    assert part0 is not None and part1 is not None
    assert part0.parent_exchange_id == unrelated.id
    assert part1.parent_exchange_id == part0.id


@st.composite
def _grouper_message_lists(draw: st.DrawFn) -> list[MessageRow]:
    id_list = draw(
        st.lists(st.integers(min_value=1, max_value=40), unique=True, min_size=1, max_size=20)
    )
    thread_pool = [None, 501, 502]
    channel_pool = [1, 2]
    messages = []
    for index, message_id in enumerate(id_list):
        offset = draw(st.integers(min_value=0, max_value=600))
        thread_id = draw(st.sampled_from(thread_pool))
        channel_id = thread_id if thread_id is not None else draw(st.sampled_from(channel_pool))
        reply_candidates = [None, 99999, *id_list[:index]]
        reply_to_id = draw(st.sampled_from(reply_candidates))
        deleted = draw(st.booleans())
        is_bot = draw(st.booleans())
        messages.append(
            make_message(
                message_id,
                channel_id=channel_id,
                created_at=at(offset),
                thread_id=thread_id,
                reply_to_id=reply_to_id,
                deleted_at=at(offset) if deleted else None,
                author_is_bot=is_bot,
            )
        )
    return messages


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(_grouper_message_lists(), st.integers(min_value=1, max_value=6))
def test_group_pending_property_grouping_is_complete_ordered_and_idempotent(
    messages: list[MessageRow], max_messages: int
) -> None:
    conn = open_database(":memory:")
    migrate(conn)
    for message in messages:
        upsert_message(conn, message)
    by_id = {message.id: message for message in messages}
    clock = FixedClock(at(1_000_000))
    quiet_gap = timedelta(minutes=30)

    group_pending(conn, clock, quiet_gap=quiet_gap, max_messages=max_messages, include_bots=False)

    kept_ids = {message.id for message in drop_ungroupable(messages, include_bots=False)}
    assert grouped_message_ids(conn) == kept_ids

    exchange_rows = conn.execute("SELECT id FROM exchanges").fetchall()
    seen: set[int] = set()
    for row in exchange_rows:
        exchange_id = row["id"]
        member_ids = exchange_message_ids(conn, exchange_id)
        assert not seen.intersection(member_ids)
        seen.update(member_ids)
        positions = [
            position_row["position"]
            for position_row in conn.execute(
                "SELECT position FROM exchange_messages WHERE exchange_id = ? ORDER BY position",
                (exchange_id,),
            ).fetchall()
        ]
        assert positions == list(range(len(positions)))
        assert member_ids == sorted(member_ids, key=lambda mid: (by_id[mid].created_at, mid))
    assert seen == kept_ids

    second_report = group_pending(
        conn, clock, quiet_gap=quiet_gap, max_messages=max_messages, include_bots=False
    )
    assert second_report == GroupingReport(
        exchanges_created=0, messages_grouped=0, groups_deferred=0
    )
