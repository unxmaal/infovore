import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from infovore.rows import ChannelKind
from infovore.source.fake import FakeDiscordSource
from infovore.source.protocol import (
    DiscordSource,
    MessageCreated,
    MessageDeleted,
    MessageEdited,
    ReactionChanged,
    SourceChannel,
    SourceMessage,
    SourceRateLimitedError,
    SourceUnavailableError,
    ThreadCreated,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def make_message(msg_id: int, channel_id: int = 1, content: str = "hi") -> SourceMessage:
    return SourceMessage(
        id=msg_id,
        channel_id=channel_id,
        guild_id=100,
        author_id=5,
        author_name="alice",
        author_is_bot=False,
        is_system=False,
        created_at=BASE + timedelta(minutes=msg_id),
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        attachments=(),
        reactions=(),
        raw={},
    )


def make_channel(
    channel_id: int,
    guild_id: int = 100,
    parent_id: int | None = None,
    kind: ChannelKind = ChannelKind.TEXT,
) -> SourceChannel:
    return SourceChannel(channel_id, guild_id, parent_id, f"channel-{channel_id}", kind, False)


async def collect_history(
    source: DiscordSource, channel_id: int, after_id: int | None, page_size: int
) -> list[tuple[SourceMessage, ...]]:
    pages: list[tuple[SourceMessage, ...]] = []
    async for page in source.history(channel_id, after_id, page_size):
        pages.append(tuple(page))
    return pages


async def test_list_channels_filters_by_guild_including_threads() -> None:
    text_channel = make_channel(1, guild_id=100)
    thread = make_channel(2, guild_id=100, parent_id=1, kind=ChannelKind.THREAD)
    other_guild_channel = make_channel(3, guild_id=200)
    source: DiscordSource = FakeDiscordSource(channels=[text_channel, thread, other_guild_channel])
    channels = await source.list_channels(100)
    assert set(channels) == {text_channel, thread}


async def test_list_channels_unknown_guild_returns_empty() -> None:
    source = FakeDiscordSource(channels=[make_channel(1)])
    assert await source.list_channels(999) == ()


async def test_history_pages_oldest_first_respecting_page_size() -> None:
    messages = [make_message(i) for i in (3, 1, 2, 4, 5)]
    source = FakeDiscordSource(messages=messages)
    pages = await collect_history(source, 1, None, 2)
    assert pages == [
        (make_message(1), make_message(2)),
        (make_message(3), make_message(4)),
        (make_message(5),),
    ]


async def test_history_respects_after_id_cursor() -> None:
    messages = [make_message(i) for i in range(1, 6)]
    source = FakeDiscordSource(messages=messages)
    pages = await collect_history(source, 1, 3, 10)
    assert pages == [(make_message(4), make_message(5))]


async def test_history_unknown_channel_yields_nothing() -> None:
    source = FakeDiscordSource(messages=[make_message(1)])
    pages = await collect_history(source, 999, None, 10)
    assert pages == []


async def test_history_orders_by_created_at_then_id_on_ties() -> None:
    tied_later = SourceMessage(
        id=1,
        channel_id=1,
        guild_id=100,
        author_id=5,
        author_name="alice",
        author_is_bot=False,
        is_system=False,
        created_at=BASE,
        edited_at=None,
        content="a",
        reply_to_id=None,
        thread_id=None,
        attachments=(),
        reactions=(),
        raw={},
    )
    tied_earlier_id = SourceMessage(
        id=0,
        channel_id=1,
        guild_id=100,
        author_id=5,
        author_name="alice",
        author_is_bot=False,
        is_system=False,
        created_at=BASE,
        edited_at=None,
        content="b",
        reply_to_id=None,
        thread_id=None,
        attachments=(),
        reactions=(),
        raw={},
    )
    source = FakeDiscordSource(messages=[tied_later, tied_earlier_id])
    pages = await collect_history(source, 1, None, 10)
    assert pages == [(tied_earlier_id, tied_later)]


async def test_fail_next_history_call_raises_once_then_succeeds() -> None:
    source = FakeDiscordSource(messages=[make_message(1), make_message(2)])
    source.fail_next_history_call(SourceRateLimitedError(2.0), channel_id=1)
    with pytest.raises(SourceRateLimitedError) as exc_info:
        await collect_history(source, 1, None, 10)
    assert exc_info.value.retry_after == 2.0
    pages = await collect_history(source, 1, None, 10)
    assert pages == [(make_message(1), make_message(2))]


async def test_fail_next_history_call_applies_to_any_channel_when_unscoped() -> None:
    source = FakeDiscordSource(messages=[make_message(1, channel_id=7)])
    source.fail_next_history_call(SourceUnavailableError("down"))
    with pytest.raises(SourceUnavailableError):
        await collect_history(source, 7, None, 10)
    pages = await collect_history(source, 7, None, 10)
    assert pages == [(make_message(1, channel_id=7),)]


async def test_interrupt_history_after_pages_then_resume_matches_uninterrupted_walk() -> None:
    messages = [make_message(i) for i in range(1, 6)]
    baseline_source = FakeDiscordSource(messages=messages)
    baseline = await collect_history(baseline_source, 1, None, 2)

    source = FakeDiscordSource(messages=messages)
    source.interrupt_history_after(1, pages=1, error=SourceUnavailableError("boom"))
    collected: list[tuple[SourceMessage, ...]] = []
    with pytest.raises(SourceUnavailableError):
        async for page in source.history(1, None, 2):
            collected.append(tuple(page))
    assert collected == [baseline[0]]

    resumed = await collect_history(source, 1, collected[-1][-1].id, 2)
    assert collected + resumed == baseline


async def test_role_member_ids_known_and_unknown() -> None:
    source = FakeDiscordSource(role_members={100: {"mod": {1, 2, 3}}})
    assert await source.role_member_ids(100, "mod") == frozenset({1, 2, 3})
    assert await source.role_member_ids(100, "missing") == frozenset()
    assert await source.role_member_ids(999, "mod") == frozenset()


async def test_events_push_and_close_terminate_iterator() -> None:
    source = FakeDiscordSource()
    collected: list[object] = []

    async def consume() -> None:
        async for event in source.events():
            collected.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    message = make_message(1)
    source.push(MessageCreated(message))
    source.push(ReactionChanged(message_id=1, emoji="x", count=1))
    source.close()
    await task
    assert collected == [MessageCreated(message), ReactionChanged(message_id=1, emoji="x", count=1)]


async def test_push_message_created_updates_history_store() -> None:
    source = FakeDiscordSource()
    message = make_message(1)
    source.push(MessageCreated(message))
    source.close()
    pages = await collect_history(source, 1, None, 10)
    assert pages == [(message,)]


async def test_push_message_edited_updates_history_store() -> None:
    source = FakeDiscordSource(messages=[make_message(1, content="original")])
    edited = make_message(1, content="edited")
    source.push(MessageEdited(edited))
    source.close()
    pages = await collect_history(source, 1, None, 10)
    assert pages == [(edited,)]


async def test_push_message_deleted_removes_from_history_store() -> None:
    source = FakeDiscordSource(messages=[make_message(1), make_message(2)])
    source.push(MessageDeleted(message_id=1, channel_id=1))
    source.close()
    pages = await collect_history(source, 1, None, 10)
    assert pages == [(make_message(2),)]


async def test_push_non_message_events_do_not_touch_message_store() -> None:
    source = FakeDiscordSource(messages=[make_message(1)])
    source.push(ReactionChanged(message_id=1, emoji="x", count=5))
    source.push(ThreadCreated(make_channel(2, parent_id=1, kind=ChannelKind.THREAD)))
    source.close()
    pages = await collect_history(source, 1, None, 10)
    assert pages == [(make_message(1),)]


async def test_add_channel_mutation_helper() -> None:
    source = FakeDiscordSource()
    channel = make_channel(5)
    source.add_channel(channel)
    assert await source.list_channels(100) == (channel,)


async def test_add_message_mutation_helper() -> None:
    source = FakeDiscordSource()
    message = make_message(1)
    source.add_message(message)
    pages = await collect_history(source, 1, None, 10)
    assert pages == [(message,)]


async def test_edit_message_mutation_helper() -> None:
    source = FakeDiscordSource(messages=[make_message(1, content="original")])
    edited = make_message(1, content="edited")
    source.edit_message(edited)
    pages = await collect_history(source, 1, None, 10)
    assert pages == [(edited,)]


async def test_delete_message_mutation_helper() -> None:
    source = FakeDiscordSource(messages=[make_message(1), make_message(2)])
    source.delete_message(channel_id=1, message_id=1)
    pages = await collect_history(source, 1, None, 10)
    assert pages == [(make_message(2),)]
