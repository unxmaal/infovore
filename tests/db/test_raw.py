import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.raw import (
    UpsertOutcome,
    get_backfill_checkpoint,
    get_channel,
    get_message,
    mark_deleted,
    mark_edited,
    message_revisions,
    set_backfill_checkpoint,
    upsert_channel,
    upsert_message,
)
from infovore.rows import ChannelKind, ChannelRow, MessageRow

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
