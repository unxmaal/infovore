import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import insert_exchange
from infovore.db.raw import (
    UpsertOutcome,
    attachments_for_messages,
    get_backfill_checkpoint,
    get_channel,
    get_message,
    latest_exchange_for_thread,
    mark_deleted,
    mark_edited,
    message_revisions,
    messages_by_ids,
    opted_out_user_ids,
    reactions_for_messages,
    set_backfill_checkpoint,
    set_reaction_count,
    ungrouped_channel_ids,
    ungrouped_messages_for_channel,
    upsert_attachment,
    upsert_channel,
    upsert_message,
)
from infovore.rows import (
    AttachmentRow,
    ChannelKind,
    ChannelRow,
    ExchangeRow,
    ExtractionStatus,
    GroupingRule,
    MessageRow,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "test.db")
    migrate(connection)
    return connection


def make_channel(
    channel_id: int = 1,
    guild_id: int = 1,
    parent_id: int | None = None,
    name: str = "general",
    kind: ChannelKind = ChannelKind.TEXT,
    archived: bool = False,
    last_backfilled_message_id: int | None = None,
) -> ChannelRow:
    return ChannelRow(
        id=channel_id,
        guild_id=guild_id,
        parent_id=parent_id,
        name=name,
        kind=kind,
        archived=archived,
        last_backfilled_message_id=last_backfilled_message_id,
    )


def test_get_channel_returns_none_when_missing(conn: sqlite3.Connection) -> None:
    assert get_channel(conn, 1) is None


def test_upsert_channel_inserts_then_get_channel_returns_it(conn: sqlite3.Connection) -> None:
    upsert_channel(conn, make_channel())
    assert get_channel(conn, 1) == make_channel()


def test_upsert_channel_inserts_thread_with_parent(conn: sqlite3.Connection) -> None:
    upsert_channel(conn, make_channel(channel_id=2, parent_id=1, kind=ChannelKind.THREAD))
    fetched = get_channel(conn, 2)
    assert fetched is not None
    assert fetched.parent_id == 1
    assert fetched.kind is ChannelKind.THREAD


def test_upsert_channel_updates_mutable_fields_but_not_checkpoint(
    conn: sqlite3.Connection,
) -> None:
    upsert_channel(conn, make_channel(name="general", archived=False))
    set_backfill_checkpoint(conn, 1, 42)
    upsert_channel(conn, make_channel(name="renamed", archived=True))
    fetched = get_channel(conn, 1)
    assert fetched is not None
    assert fetched.name == "renamed"
    assert fetched.archived is True
    assert fetched.last_backfilled_message_id == 42


def test_get_backfill_checkpoint_returns_none_for_unknown_channel(
    conn: sqlite3.Connection,
) -> None:
    assert get_backfill_checkpoint(conn, 1) is None


def test_get_backfill_checkpoint_returns_none_when_not_set(conn: sqlite3.Connection) -> None:
    upsert_channel(conn, make_channel())
    assert get_backfill_checkpoint(conn, 1) is None


def test_set_backfill_checkpoint_then_get_returns_it(conn: sqlite3.Connection) -> None:
    upsert_channel(conn, make_channel())
    set_backfill_checkpoint(conn, 1, 99)
    assert get_backfill_checkpoint(conn, 1) == 99


def make_message(
    message_id: int = 1,
    channel_id: int = 1,
    guild_id: int = 1,
    author_id: int = 1,
    author_name_at_time: str = "alice",
    author_is_bot: bool = False,
    created_at: datetime = NOW,
    edited_at: datetime | None = None,
    content: str = "hello",
    reply_to_id: int | None = None,
    thread_id: int | None = None,
    deleted_at: datetime | None = None,
    ingested_at: datetime = NOW,
    raw_json: str = "{}",
) -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=channel_id,
        guild_id=guild_id,
        author_id=author_id,
        author_name_at_time=author_name_at_time,
        author_is_bot=author_is_bot,
        created_at=created_at,
        edited_at=edited_at,
        content=content,
        reply_to_id=reply_to_id,
        thread_id=thread_id,
        deleted_at=deleted_at,
        ingested_at=ingested_at,
        raw_json=raw_json,
    )


def test_get_message_returns_none_when_missing(conn: sqlite3.Connection) -> None:
    assert get_message(conn, 1) is None


def test_upsert_message_inserts_new_row(conn: sqlite3.Connection) -> None:
    outcome = upsert_message(conn, make_message())
    assert outcome is UpsertOutcome.INSERTED
    assert get_message(conn, 1) == make_message()


def test_upsert_message_identical_reupsert_is_unchanged_and_writes_nothing(
    conn: sqlite3.Connection,
) -> None:
    upsert_message(conn, make_message())
    outcome = upsert_message(conn, make_message())
    assert outcome is UpsertOutcome.UNCHANGED
    assert get_message(conn, 1) == make_message()
    assert message_revisions(conn, 1) == []


def test_upsert_message_with_different_content_is_an_edit(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message(content="original", raw_json='{"v": 1}'))
    edited_at = datetime(2026, 1, 2, tzinfo=UTC)
    outcome = upsert_message(
        conn,
        make_message(content="edited", edited_at=edited_at, raw_json='{"v": 2}'),
    )
    assert outcome is UpsertOutcome.UPDATED
    fetched = get_message(conn, 1)
    assert fetched is not None
    assert fetched.content == "edited"
    assert fetched.edited_at == edited_at
    assert fetched.raw_json == '{"v": 2}'
    revisions = message_revisions(conn, 1)
    assert len(revisions) == 1
    assert revisions[0].revision == 0
    assert revisions[0].content == "original"
    assert revisions[0].edited_at is None
    assert revisions[0].raw_json == '{"v": 1}'


def test_upsert_message_edit_never_touches_deleted_at_or_ingested_at(
    conn: sqlite3.Connection,
) -> None:
    original_ingested_at = NOW
    upsert_message(conn, make_message(content="original", ingested_at=original_ingested_at))
    mark_deleted(conn, 1, datetime(2026, 1, 3, tzinfo=UTC))
    upsert_message(
        conn,
        make_message(
            content="edited",
            ingested_at=datetime(2026, 1, 4, tzinfo=UTC),
            deleted_at=None,
        ),
    )
    fetched = get_message(conn, 1)
    assert fetched is not None
    assert fetched.ingested_at == original_ingested_at
    assert fetched.deleted_at == datetime(2026, 1, 3, tzinfo=UTC)


def test_upsert_message_second_edit_uses_next_revision_number(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message(content="v0"))
    upsert_message(conn, make_message(content="v1"))
    upsert_message(conn, make_message(content="v2"))
    revisions = message_revisions(conn, 1)
    assert [r.revision for r in revisions] == [0, 1]
    assert [r.content for r in revisions] == ["v0", "v1"]


def test_message_revisions_empty_when_no_edits(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message())
    assert message_revisions(conn, 1) == []


def test_mark_edited_unknown_message_returns_false(conn: sqlite3.Connection) -> None:
    assert mark_edited(conn, 1, "new", None, "{}") is False


def test_mark_edited_updates_content_and_archives_revision(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message(content="original", raw_json='{"v": 1}'))
    edited_at = datetime(2026, 1, 2, tzinfo=UTC)
    result = mark_edited(conn, 1, "edited", edited_at, '{"v": 2}')
    assert result is True
    fetched = get_message(conn, 1)
    assert fetched is not None
    assert fetched.content == "edited"
    assert fetched.edited_at == edited_at
    assert fetched.raw_json == '{"v": 2}'
    revisions = message_revisions(conn, 1)
    assert len(revisions) == 1
    assert revisions[0].content == "original"


def test_mark_edited_identical_content_is_noop_and_returns_true(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message(content="same"))
    result = mark_edited(conn, 1, "same", None, "{}")
    assert result is True
    assert message_revisions(conn, 1) == []


def test_mark_deleted_unknown_message_returns_false(conn: sqlite3.Connection) -> None:
    assert mark_deleted(conn, 1, NOW) is False


def test_mark_deleted_sets_deleted_at_and_returns_true(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message())
    at = datetime(2026, 1, 5, tzinfo=UTC)
    assert mark_deleted(conn, 1, at) is True
    fetched = get_message(conn, 1)
    assert fetched is not None
    assert fetched.deleted_at == at


def test_mark_deleted_twice_keeps_first_deleted_at(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message())
    first = datetime(2026, 1, 5, tzinfo=UTC)
    second = datetime(2026, 1, 6, tzinfo=UTC)
    assert mark_deleted(conn, 1, first) is True
    assert mark_deleted(conn, 1, second) is True
    fetched = get_message(conn, 1)
    assert fetched is not None
    assert fetched.deleted_at == first


def make_attachment(
    attachment_id: int = 1,
    message_id: int = 1,
    filename: str = "photo.png",
    content_type: str | None = "image/png",
    size: int = 1024,
    url: str = "https://example.com/photo.png",
    sha256: str | None = None,
    local_path: str | None = None,
) -> AttachmentRow:
    return AttachmentRow(
        id=attachment_id,
        message_id=message_id,
        filename=filename,
        content_type=content_type,
        size=size,
        url=url,
        sha256=sha256,
        local_path=local_path,
    )


def test_upsert_attachment_inserts_then_query(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message())
    upsert_attachment(conn, make_attachment())
    assert attachments_for_messages(conn, [1]) == [make_attachment()]


def test_upsert_attachment_is_idempotent_and_updates_fields(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message())
    upsert_attachment(conn, make_attachment())
    upsert_attachment(conn, make_attachment(sha256="abc123", local_path="/tmp/photo.png"))
    fetched = attachments_for_messages(conn, [1])
    assert len(fetched) == 1
    assert fetched[0].sha256 == "abc123"
    assert fetched[0].local_path == "/tmp/photo.png"


def test_attachments_for_messages_returns_empty_list_for_empty_ids(
    conn: sqlite3.Connection,
) -> None:
    assert attachments_for_messages(conn, []) == []


def test_attachments_for_messages_filters_by_message_id(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message(message_id=1))
    upsert_message(conn, make_message(message_id=2))
    upsert_attachment(conn, make_attachment(attachment_id=1, message_id=1))
    upsert_attachment(conn, make_attachment(attachment_id=2, message_id=2))
    fetched = attachments_for_messages(conn, [1])
    assert [a.id for a in fetched] == [1]


def test_set_reaction_count_inserts_new(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message())
    set_reaction_count(conn, 1, "\U0001f44d", 3)
    reactions = reactions_for_messages(conn, [1])
    assert len(reactions) == 1
    assert reactions[0].message_id == 1
    assert reactions[0].emoji == "\U0001f44d"
    assert reactions[0].count == 3


def test_set_reaction_count_updates_existing(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message())
    set_reaction_count(conn, 1, "\U0001f44d", 3)
    set_reaction_count(conn, 1, "\U0001f44d", 7)
    reactions = reactions_for_messages(conn, [1])
    assert len(reactions) == 1
    assert reactions[0].count == 7


def test_set_reaction_count_zero_deletes_row(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message())
    set_reaction_count(conn, 1, "\U0001f44d", 3)
    set_reaction_count(conn, 1, "\U0001f44d", 0)
    assert reactions_for_messages(conn, [1]) == []


def test_set_reaction_count_negative_deletes_row_when_absent(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message())
    set_reaction_count(conn, 1, "\U0001f44d", -1)
    assert reactions_for_messages(conn, [1]) == []


def test_reactions_for_messages_returns_empty_list_for_empty_ids(
    conn: sqlite3.Connection,
) -> None:
    assert reactions_for_messages(conn, []) == []


def test_reactions_for_messages_filters_by_message_id(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message(message_id=1))
    upsert_message(conn, make_message(message_id=2))
    set_reaction_count(conn, 1, "\U0001f44d", 1)
    set_reaction_count(conn, 2, "\U0001f44d", 1)
    fetched = reactions_for_messages(conn, [1])
    assert [r.message_id for r in fetched] == [1]


def make_exchange_row(
    content_hash: str,
    channel_id: int = 1,
    thread_id: int | None = None,
    first_message_id: int = 1,
    last_message_id: int = 1,
    message_count: int = 1,
    started_at: datetime = NOW,
) -> ExchangeRow:
    return ExchangeRow(
        id=None,
        channel_id=channel_id,
        thread_id=thread_id,
        first_message_id=first_message_id,
        last_message_id=last_message_id,
        started_at=started_at,
        ended_at=started_at,
        message_count=message_count,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=content_hash,
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )


def test_ungrouped_channel_ids_returns_only_channels_with_ungrouped_messages(
    conn: sqlite3.Connection,
) -> None:
    upsert_message(conn, make_message(message_id=1, channel_id=1))
    upsert_message(conn, make_message(message_id=2, channel_id=2))
    insert_exchange(conn, make_exchange_row("h1", channel_id=1, first_message_id=1), [1])
    assert ungrouped_channel_ids(conn) == [2]


def test_ungrouped_channel_ids_empty_when_all_grouped(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message(message_id=1, channel_id=1))
    insert_exchange(conn, make_exchange_row("h2", channel_id=1, first_message_id=1), [1])
    assert ungrouped_channel_ids(conn) == []


def test_ungrouped_messages_for_channel_excludes_grouped_and_other_channels(
    conn: sqlite3.Connection,
) -> None:
    upsert_message(conn, make_message(message_id=1, channel_id=1, created_at=NOW))
    upsert_message(
        conn,
        make_message(message_id=2, channel_id=1, created_at=NOW + timedelta(minutes=1)),
    )
    upsert_message(conn, make_message(message_id=3, channel_id=2, created_at=NOW))
    insert_exchange(conn, make_exchange_row("h3", channel_id=1, first_message_id=1), [1])
    result = ungrouped_messages_for_channel(conn, 1)
    assert [message.id for message in result] == [2]


def test_ungrouped_messages_for_channel_orders_by_created_at_then_id(
    conn: sqlite3.Connection,
) -> None:
    upsert_message(conn, make_message(message_id=2, channel_id=1, created_at=NOW))
    upsert_message(conn, make_message(message_id=1, channel_id=1, created_at=NOW))
    upsert_message(
        conn,
        make_message(message_id=3, channel_id=1, created_at=NOW - timedelta(minutes=1)),
    )
    result = ungrouped_messages_for_channel(conn, 1)
    assert [message.id for message in result] == [3, 1, 2]


def test_latest_exchange_for_thread_returns_none_when_no_exchanges(
    conn: sqlite3.Connection,
) -> None:
    assert latest_exchange_for_thread(conn, 77) is None


def test_latest_exchange_for_thread_returns_most_recently_started(
    conn: sqlite3.Connection,
) -> None:
    upsert_message(conn, make_message(message_id=1, channel_id=1))
    upsert_message(
        conn, make_message(message_id=2, channel_id=1, created_at=NOW + timedelta(hours=1))
    )
    older_id = insert_exchange(
        conn,
        make_exchange_row("h4", channel_id=1, thread_id=77, first_message_id=1, started_at=NOW),
        [1],
    )
    newer_id = insert_exchange(
        conn,
        make_exchange_row(
            "h5",
            channel_id=1,
            thread_id=77,
            first_message_id=2,
            started_at=NOW + timedelta(hours=1),
        ),
        [2],
    )
    assert older_id != newer_id
    assert latest_exchange_for_thread(conn, 77) == newer_id


def test_latest_exchange_for_thread_ignores_other_threads(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message(message_id=1, channel_id=1))
    insert_exchange(
        conn, make_exchange_row("h6", channel_id=1, thread_id=88, first_message_id=1), [1]
    )
    assert latest_exchange_for_thread(conn, 77) is None


def test_messages_by_ids_returns_empty_list_for_empty_ids(conn: sqlite3.Connection) -> None:
    assert messages_by_ids(conn, []) == []


def test_messages_by_ids_preserves_requested_order(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message(message_id=1))
    upsert_message(conn, make_message(message_id=2))
    upsert_message(conn, make_message(message_id=3))
    fetched = messages_by_ids(conn, [3, 1, 2])
    assert [m.id for m in fetched] == [3, 1, 2]


def test_messages_by_ids_skips_ids_that_do_not_exist(conn: sqlite3.Connection) -> None:
    upsert_message(conn, make_message(message_id=1))
    fetched = messages_by_ids(conn, [1, 999])
    assert [m.id for m in fetched] == [1]


def test_opted_out_user_ids_empty_when_none(conn: sqlite3.Connection) -> None:
    assert opted_out_user_ids(conn) == frozenset()


def test_opted_out_user_ids_returns_all_opted_out_users(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (42, '2026-01-01T00:00:00+00:00')")
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (7, '2026-01-01T00:00:00+00:00')")
    assert opted_out_user_ids(conn) == frozenset({42, 7})
