import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import (
    DuplicateExchangeError,
    MessageAlreadyGroupedError,
    claimable_exchanges,
    exchange_for_message,
    exchange_message_ids,
    get_exchange,
    grouped_message_ids,
    has_untriaged_claimable,
    mark_stale_for_message,
    record_failure,
    set_status,
)
from infovore.db.exchanges import (
    insert_exchange as insert_exchange_row,
)
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule
from tests.cascade_marks import ARCHIVED_KINDS, RULED_OUT_KINDS, mark

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def insert_exchange(
    conn: sqlite3.Connection,
    exchange: ExchangeRow,
    message_ids: Sequence[int],
    cascade: str | None = "residue",
) -> int:
    exchange_id = insert_exchange_row(conn, exchange, message_ids)
    if cascade is not None:
        mark(conn, exchange_id, cascade)
    return exchange_id


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    return connection


def insert_message(conn: sqlite3.Connection, message_id: int, channel_id: int = 1) -> None:
    now = to_db_time(NOW)
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json) VALUES (?, ?, 1, 1, 'a', ?, 'x', ?, '{}')",
        (message_id, channel_id, now, now),
    )


def insert_messages(conn: sqlite3.Connection, message_ids: Sequence[int]) -> None:
    for message_id in message_ids:
        insert_message(conn, message_id)


def insert_channel(
    conn: sqlite3.Connection,
    channel_id: int,
    name: str,
    parent_id: int | None = None,
    kind: str = "text",
) -> None:
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind) VALUES (?, 1, ?, ?, ?)",
        (channel_id, parent_id, name, kind),
    )


def make_exchange(
    message_count: int = 2,
    content_hash: str = "hash-1",
    status: ExtractionStatus = ExtractionStatus.PENDING,
    retry_count: int = 0,
    parent_exchange_id: int | None = None,
    started_at: datetime = NOW,
    channel_id: int = 1,
) -> ExchangeRow:
    return ExchangeRow(
        id=None,
        channel_id=channel_id,
        thread_id=None,
        first_message_id=1,
        last_message_id=message_count,
        started_at=started_at,
        ended_at=started_at,
        message_count=message_count,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=content_hash,
        parent_exchange_id=parent_exchange_id,
        extraction_status=status,
        retry_count=retry_count,
        last_error=None,
    )


def exchange_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM exchanges").fetchone()
    return int(row[0])


def test_insert_exchange_persists_and_positions_are_contiguous(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1, 2, 3])
    exchange = make_exchange(message_count=3, content_hash="h1")
    exchange_id = insert_exchange(conn, exchange, [3, 1, 2])
    assert exchange_message_ids(conn, exchange_id) == [3, 1, 2]
    assert get_exchange(conn, exchange_id) == replace(exchange, id=exchange_id)


def test_insert_exchange_rejects_empty_message_ids(conn: sqlite3.Connection) -> None:
    exchange = make_exchange(message_count=0, content_hash="h-empty")
    with pytest.raises(ValueError, match="message_ids"):
        insert_exchange(conn, exchange, [])


def test_insert_exchange_rejects_mismatched_message_count(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1, 2])
    exchange = make_exchange(message_count=2, content_hash="h-mismatch")
    with pytest.raises(ValueError, match="message_ids"):
        insert_exchange(conn, exchange, [1])


def test_insert_exchange_duplicate_content_hash_writes_nothing(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1, 2])
    first = make_exchange(message_count=1, content_hash="dup")
    insert_exchange(conn, first, [1])
    second = make_exchange(message_count=1, content_hash="dup")
    with pytest.raises(DuplicateExchangeError):
        insert_exchange(conn, second, [2])
    assert exchange_count(conn) == 1
    assert grouped_message_ids(conn) == {1}


def test_insert_exchange_message_already_grouped_rolls_back(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1, 2])
    first = make_exchange(message_count=1, content_hash="a")
    insert_exchange(conn, first, [1])
    second = make_exchange(message_count=2, content_hash="b")
    with pytest.raises(MessageAlreadyGroupedError):
        insert_exchange(conn, second, [2, 1])
    assert exchange_count(conn) == 1
    assert exchange_for_message(conn, 2) is None
    assert grouped_message_ids(conn) == {1}


def test_insert_exchange_invalid_parent_exchange_id_propagates_and_rolls_back(
    conn: sqlite3.Connection,
) -> None:
    insert_messages(conn, [1])
    exchange = make_exchange(message_count=1, content_hash="bad-parent", parent_exchange_id=999)
    with pytest.raises(sqlite3.IntegrityError):
        insert_exchange(conn, exchange, [1])
    assert exchange_count(conn) == 0
    assert grouped_message_ids(conn) == set()


def test_insert_exchange_nonexistent_message_id_propagates_and_rolls_back(
    conn: sqlite3.Connection,
) -> None:
    exchange = make_exchange(message_count=1, content_hash="bad-message")
    with pytest.raises(sqlite3.IntegrityError):
        insert_exchange(conn, exchange, [12345])
    assert exchange_count(conn) == 0
    assert grouped_message_ids(conn) == set()


def test_get_exchange_returns_none_when_missing(conn: sqlite3.Connection) -> None:
    assert get_exchange(conn, 999) is None


def test_exchange_for_message_returns_none_when_ungrouped(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    assert exchange_for_message(conn, 1) is None


def test_exchange_for_message_returns_owning_exchange_id(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1, 2])
    exchange = make_exchange(message_count=2, content_hash="h2")
    exchange_id = insert_exchange(conn, exchange, [1, 2])
    assert exchange_for_message(conn, 2) == exchange_id


def test_grouped_message_ids_spans_multiple_exchanges(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1, 2, 3, 4])
    insert_exchange(conn, make_exchange(message_count=2, content_hash="g1"), [1, 2])
    insert_exchange(conn, make_exchange(message_count=2, content_hash="g2"), [3, 4])
    assert grouped_message_ids(conn) == {1, 2, 3, 4}


def test_claimable_exchanges_filters_status_and_retry_and_orders(
    conn: sqlite3.Connection,
) -> None:
    insert_messages(conn, list(range(1, 9)))
    old = NOW
    newer = NOW + timedelta(minutes=5)
    newest = NOW + timedelta(minutes=10)
    pending_id = insert_exchange(
        conn,
        make_exchange(message_count=1, content_hash="pending", started_at=newer),
        [1],
    )
    stale_id = insert_exchange(
        conn,
        make_exchange(
            message_count=1,
            content_hash="stale",
            status=ExtractionStatus.STALE,
            started_at=old,
        ),
        [2],
    )
    insert_exchange(
        conn,
        make_exchange(
            message_count=1,
            content_hash="done",
            status=ExtractionStatus.DONE,
            started_at=old,
        ),
        [3],
    )
    insert_exchange(
        conn,
        make_exchange(
            message_count=1,
            content_hash="exhausted",
            status=ExtractionStatus.PENDING,
            retry_count=3,
            started_at=old,
        ),
        [4],
    )
    insert_exchange(
        conn,
        make_exchange(
            message_count=1,
            content_hash="also-pending",
            started_at=newest,
        ),
        [5],
    )

    result = claimable_exchanges(conn, limit=10, max_retries=3)
    assert [row.id for row in result] == [stale_id, pending_id, result[2].id]
    assert {row.content_hash for row in result} == {"stale", "pending", "also-pending"}

    limited = claimable_exchanges(conn, limit=1, max_retries=3)
    assert [row.id for row in limited] == [stale_id]


def test_claimable_exchanges_excludes_denylisted_channel_by_name(
    conn: sqlite3.Connection,
) -> None:
    insert_channel(conn, 1, "general")
    insert_channel(conn, 2, "food")
    insert_messages(conn, [1, 2])
    kept_id = insert_exchange(
        conn, make_exchange(message_count=1, content_hash="kept", channel_id=1), [1]
    )
    insert_exchange(conn, make_exchange(message_count=1, content_hash="trashed", channel_id=2), [2])

    result = claimable_exchanges(
        conn, limit=10, max_retries=3, exclude_channels=frozenset({"food"})
    )

    assert [row.id for row in result] == [kept_id]


def test_claimable_exchanges_denylist_matches_channel_name_case_insensitively(
    conn: sqlite3.Connection,
) -> None:
    """`exclude_channels` is expected already normalized (lowercase, no `#`
    — `infovore.config.normalize_channel_names`'s job); `claimable_exchanges`
    itself only needs to match a differently-cased stored channel name
    case-insensitively."""
    insert_channel(conn, 1, "Food")
    insert_messages(conn, [1])
    insert_exchange(conn, make_exchange(message_count=1, content_hash="a", channel_id=1), [1])

    result = claimable_exchanges(
        conn, limit=10, max_retries=3, exclude_channels=frozenset({"food"})
    )

    assert result == []


def test_claimable_exchanges_excludes_thread_whose_parent_is_denylisted(
    conn: sqlite3.Connection,
) -> None:
    insert_channel(conn, 10, "food")
    insert_channel(conn, 11, "food-thread-1", parent_id=10, kind="thread")
    insert_messages(conn, [1])
    insert_exchange(conn, make_exchange(message_count=1, content_hash="a", channel_id=11), [1])

    result = claimable_exchanges(
        conn, limit=10, max_retries=3, exclude_channels=frozenset({"food"})
    )

    assert result == []


def test_claimable_exchanges_keeps_thread_whose_parent_is_not_denylisted(
    conn: sqlite3.Connection,
) -> None:
    insert_channel(conn, 10, "general")
    insert_channel(conn, 11, "general-thread-1", parent_id=10, kind="thread")
    insert_messages(conn, [1])
    exchange_id = insert_exchange(
        conn, make_exchange(message_count=1, content_hash="a", channel_id=11), [1]
    )

    result = claimable_exchanges(
        conn, limit=10, max_retries=3, exclude_channels=frozenset({"food"})
    )

    assert [row.id for row in result] == [exchange_id]


def test_claimable_exchanges_without_denylist_ignores_channels(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    exchange_id = insert_exchange(conn, make_exchange(message_count=1, content_hash="a"), [1])

    result = claimable_exchanges(conn, limit=10, max_retries=3, exclude_channels=frozenset())

    assert [row.id for row in result] == [exchange_id]


def test_has_untriaged_claimable_true_when_version_null(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    insert_exchange(conn, make_exchange(message_count=1, content_hash="untriaged"), [1])

    assert has_untriaged_claimable(conn, "t1", max_retries=3) is True


def test_has_untriaged_claimable_true_when_version_stale(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    exchange_id = insert_exchange(conn, make_exchange(message_count=1, content_hash="old"), [1])
    conn.execute(
        "UPDATE exchanges SET triage_score = 0.5, triage_version = 't0' WHERE id = ?",
        (exchange_id,),
    )

    assert has_untriaged_claimable(conn, "t1", max_retries=3) is True


def test_has_untriaged_claimable_false_when_all_current(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    exchange_id = insert_exchange(conn, make_exchange(message_count=1, content_hash="ok"), [1])
    conn.execute(
        "UPDATE exchanges SET triage_score = 0.5, triage_version = 't1' WHERE id = ?",
        (exchange_id,),
    )

    assert has_untriaged_claimable(conn, "t1", max_retries=3) is False


def test_has_untriaged_claimable_ignores_exhausted_retries(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    insert_exchange(
        conn, make_exchange(message_count=1, content_hash="exhausted", retry_count=3), [1]
    )

    assert has_untriaged_claimable(conn, "t1", max_retries=3) is False


def test_has_untriaged_claimable_ignores_done_exchanges(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    insert_exchange(
        conn,
        make_exchange(message_count=1, content_hash="done", status=ExtractionStatus.DONE),
        [1],
    )

    assert has_untriaged_claimable(conn, "t1", max_retries=3) is False


def test_exchange_row_carries_triage_columns(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    exchange_id = insert_exchange(conn, make_exchange(message_count=1, content_hash="tr"), [1])
    conn.execute(
        "UPDATE exchanges SET triage_score = 0.42, triage_reasons = ?, triage_version = 't1'"
        " WHERE id = ?",
        ('[["code", 0.15]]', exchange_id),
    )

    fetched = get_exchange(conn, exchange_id)

    assert fetched is not None
    assert fetched.triage_score == 0.42
    assert fetched.triage_reasons == '[["code", 0.15]]'
    assert fetched.triage_version == "t1"


def test_set_status_updates_status_and_last_error(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    exchange_id = insert_exchange(conn, make_exchange(message_count=1, content_hash="s1"), [1])
    set_status(conn, exchange_id, ExtractionStatus.DONE)
    fetched = get_exchange(conn, exchange_id)
    assert fetched is not None
    assert fetched.extraction_status == ExtractionStatus.DONE
    assert fetched.last_error is None
    set_status(conn, exchange_id, ExtractionStatus.FAILED, last_error="boom")
    refetched = get_exchange(conn, exchange_id)
    assert refetched is not None
    assert refetched.extraction_status == ExtractionStatus.FAILED
    assert refetched.last_error == "boom"


def test_record_failure_keeps_status_until_max_retries_then_fails(
    conn: sqlite3.Connection,
) -> None:
    insert_messages(conn, [1, 2])
    pending_id = insert_exchange(
        conn, make_exchange(message_count=1, content_hash="rf-pending"), [1]
    )
    stale_id = insert_exchange(
        conn,
        make_exchange(message_count=1, content_hash="rf-stale", status=ExtractionStatus.STALE),
        [2],
    )

    record_failure(conn, pending_id, "boom1", max_retries=2)
    fetched = get_exchange(conn, pending_id)
    assert fetched is not None
    assert fetched.retry_count == 1
    assert fetched.extraction_status == ExtractionStatus.PENDING
    assert fetched.last_error == "boom1"

    record_failure(conn, pending_id, "boom2", max_retries=2)
    refetched = get_exchange(conn, pending_id)
    assert refetched is not None
    assert refetched.retry_count == 2
    assert refetched.extraction_status == ExtractionStatus.FAILED
    assert refetched.last_error == "boom2"

    record_failure(conn, stale_id, "boom3", max_retries=2)
    stale_fetched = get_exchange(conn, stale_id)
    assert stale_fetched is not None
    assert stale_fetched.retry_count == 1
    assert stale_fetched.extraction_status == ExtractionStatus.STALE


def test_mark_stale_for_message_flips_done_exchange(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    exchange_id = insert_exchange(
        conn,
        make_exchange(message_count=1, content_hash="ms-done", status=ExtractionStatus.DONE),
        [1],
    )
    result = mark_stale_for_message(conn, 1)
    assert result == exchange_id
    fetched = get_exchange(conn, exchange_id)
    assert fetched is not None
    assert fetched.extraction_status == ExtractionStatus.STALE


def test_mark_stale_for_message_ignores_non_done_exchange(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    exchange_id = insert_exchange(
        conn, make_exchange(message_count=1, content_hash="ms-pending"), [1]
    )
    result = mark_stale_for_message(conn, 1)
    assert result is None
    fetched = get_exchange(conn, exchange_id)
    assert fetched is not None
    assert fetched.extraction_status == ExtractionStatus.PENDING


def test_mark_stale_for_message_returns_none_when_ungrouped(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1])
    assert mark_stale_for_message(conn, 1) is None


def test_excluded_exchange_ids_cover_threads_and_an_empty_set(tmp_path: Path) -> None:
    from infovore.db.channel_filter import excluded_exchange_ids

    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    for cid, name, parent in ((1, "tech", None), (2, "food", None), (3, "recipes", 2)):
        conn.execute(
            "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
            " VALUES (?, 9, ?, ?, 'text')",
            (cid, parent, name),
        )
    for eid, cid in ((1, 1), (2, 2), (3, 3)):
        conn.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
            " ended_at, message_count, grouping_rule, content_hash)"
            " VALUES (?, ?, 1, 1, 't', 't', 1, 'quiet_gap', ?)",
            (eid, cid, f"h{eid}"),
        )

    assert excluded_exchange_ids(conn, frozenset({"food"})) == {2, 3}
    assert excluded_exchange_ids(conn, frozenset()) == frozenset()


def test_claimable_exchanges_are_chronological(conn: sqlite3.Connection) -> None:
    insert_messages(conn, [1, 2, 3])
    ids = [
        insert_exchange(
            conn,
            make_exchange(
                message_count=1, content_hash=f"h{i}", started_at=NOW + timedelta(hours=h)
            ),
            [i],
        )
        for i, h in ((1, 2), (2, 0), (3, 1))
    ]
    result = claimable_exchanges(conn, limit=10, max_retries=3)
    assert [row.id for row in result] == [ids[1], ids[2], ids[0]]


def test_claimable_exchanges_only_return_archived_ones(conn: sqlite3.Connection) -> None:
    kinds = [*ARCHIVED_KINDS, *RULED_OUT_KINDS, None]
    insert_messages(conn, list(range(1, len(kinds) + 1)))
    ids = {
        kind: insert_exchange(
            conn, make_exchange(message_count=1, content_hash=f"h{index}"), [index], cascade=kind
        )
        for index, kind in enumerate(kinds, start=1)
    }
    result = claimable_exchanges(conn, limit=10, max_retries=3)
    assert {row.id for row in result} == {ids[kind] for kind in ARCHIVED_KINDS}
