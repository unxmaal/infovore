import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace

import discord

from infovore.rows import ChannelKind
from infovore.source.live import DiscordPySource, build_client, to_source_channel, to_source_message
from infovore.source.protocol import (
    SourceForbiddenError,
    SourceMessage,
    SourceNotFoundError,
    SourceRateLimitedError,
    SourceUnavailableError,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_http_exception(
    status: int,
    retry_after: str | None = None,
    cls: type[discord.HTTPException] = discord.HTTPException,
) -> discord.HTTPException:
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    response = SimpleNamespace(status=status, reason="error", headers=headers)
    return cls(response, "boom")


@dataclass
class FakeGuildRef:
    id: int


@dataclass
class FakeChannel:
    id: int
    guild: FakeGuildRef
    name: str


@dataclass
class FakeAuthor:
    id: int
    display_name: str
    bot: bool


@dataclass
class FakeAttachment:
    id: int
    filename: str
    content_type: str | None
    size: int
    url: str


@dataclass
class FakeReaction:
    emoji: object
    count: int


@dataclass
class FakeReference:
    message_id: int | None


@dataclass
class FakeMessageType:
    name: str


@dataclass
class FakeMessage:
    id: int
    channel: FakeChannel | discord.Thread
    guild: FakeGuildRef | None
    author: FakeAuthor
    content: str
    created_at: datetime
    edited_at: datetime | None
    reference: FakeReference | None
    attachments: tuple[FakeAttachment, ...]
    reactions: tuple[FakeReaction, ...]
    type: FakeMessageType
    system: bool = False

    def is_system(self) -> bool:
        return self.system


def bare_thread(
    thread_id: int, guild_id: int, parent_id: int, name: str, archived: bool
) -> discord.Thread:
    thread = discord.Thread.__new__(discord.Thread)
    thread.id = thread_id
    thread.parent_id = parent_id
    thread.archived = archived
    thread.name = name
    thread.guild = FakeGuildRef(guild_id)  # type: ignore[assignment]
    return thread


def make_message(**overrides: object) -> FakeMessage:
    fields: dict[str, object] = {
        "id": 1,
        "channel": FakeChannel(10, FakeGuildRef(100), "general"),
        "guild": FakeGuildRef(100),
        "author": FakeAuthor(5, "alice", False),
        "content": "hi",
        "created_at": NOW,
        "edited_at": None,
        "reference": None,
        "attachments": (),
        "reactions": (),
        "type": FakeMessageType("default"),
        "system": False,
    }
    fields.update(overrides)
    return FakeMessage(**fields)  # type: ignore[arg-type]


def test_to_source_channel_text_channel() -> None:
    channel = FakeChannel(10, FakeGuildRef(100), "general")
    result = to_source_channel(channel)
    assert result.id == 10
    assert result.guild_id == 100
    assert result.parent_id is None
    assert result.name == "general"
    assert result.kind == ChannelKind.TEXT
    assert result.archived is False


def test_to_source_channel_thread() -> None:
    thread = bare_thread(20, 100, parent_id=10, name="sub-thread", archived=True)
    result = to_source_channel(thread)
    assert result.id == 20
    assert result.guild_id == 100
    assert result.parent_id == 10
    assert result.name == "sub-thread"
    assert result.kind == ChannelKind.THREAD
    assert result.archived is True


def test_to_source_message_basic_channel_message() -> None:
    message = make_message()
    result = to_source_message(message)
    assert result.id == 1
    assert result.channel_id == 10
    assert result.guild_id == 100
    assert result.author_id == 5
    assert result.author_name == "alice"
    assert result.author_is_bot is False
    assert result.is_system is False
    assert result.created_at == NOW
    assert result.edited_at is None
    assert result.content == "hi"
    assert result.reply_to_id is None
    assert result.thread_id is None
    assert result.attachments == ()
    assert result.reactions == ()


def test_to_source_message_system_message() -> None:
    message = make_message(system=True, type=FakeMessageType("pins_add"))
    result = to_source_message(message)
    assert result.is_system is True
    assert result.raw["type"] == "pins_add"


def test_to_source_message_reply() -> None:
    message = make_message(reference=FakeReference(message_id=99))
    result = to_source_message(message)
    assert result.reply_to_id == 99
    assert result.raw["reference"] == 99


def test_to_source_message_reply_reference_without_message_id() -> None:
    message = make_message(reference=FakeReference(message_id=None))
    result = to_source_message(message)
    assert result.reply_to_id is None


def test_to_source_message_in_thread() -> None:
    thread = bare_thread(20, 100, parent_id=10, name="sub-thread", archived=False)
    message = make_message(channel=thread)
    result = to_source_message(message)
    assert result.channel_id == 20
    assert result.thread_id == 20


def test_to_source_message_attachments_and_reactions() -> None:
    message = make_message(
        attachments=(FakeAttachment(7, "a.txt", "text/plain", 3, "https://x"),),
        reactions=(FakeReaction("👍", 2),),
    )
    result = to_source_message(message)
    assert result.attachments[0].id == 7
    assert result.attachments[0].filename == "a.txt"
    assert result.attachments[0].content_type == "text/plain"
    assert result.attachments[0].size == 3
    assert result.attachments[0].url == "https://x"
    assert result.reactions[0].emoji == "👍"
    assert result.reactions[0].count == 2
    assert result.raw["attachments"] == [
        {"id": 7, "filename": "a.txt", "content_type": "text/plain", "size": 3, "url": "https://x"}
    ]


def test_to_source_message_edited() -> None:
    edited_at = NOW.replace(hour=2)
    message = make_message(edited_at=edited_at)
    result = to_source_message(message)
    assert result.edited_at == edited_at
    assert result.raw["edited_at"] == edited_at.isoformat()


def test_to_source_message_bot_author() -> None:
    message = make_message(author=FakeAuthor(9, "helper-bot", True))
    result = to_source_message(message)
    assert result.author_is_bot is True
    assert result.raw["author"] == {"id": 9, "name": "helper-bot", "bot": True}


def test_to_source_message_no_guild_defaults_zero() -> None:
    message = make_message(guild=None)
    result = to_source_message(message)
    assert result.guild_id == 0
    assert result.raw["guild_id"] == 0


def test_to_source_message_raw_is_json_safe() -> None:
    message = make_message()
    result = to_source_message(message)
    json.dumps(result.raw)


@dataclass
class FakeFetchableChannel:
    id: int
    guild: FakeGuildRef
    name: str
    messages: list[FakeMessage] = field(default_factory=list)
    archived: list[FakeChannel | discord.Thread] = field(default_factory=list)
    history_error: Exception | None = None
    archived_error: Exception | None = None
    fetch_message_result: FakeMessage | None = None
    fetch_message_error: Exception | None = None
    readable: bool = True

    def permissions_for(self, member: object) -> "FakePermissions":
        return FakePermissions(self.readable, self.readable)

    async def history(
        self, *, limit: int | None, after: object | None, oldest_first: bool | None
    ) -> AsyncIterator[FakeMessage]:
        if self.history_error is not None:
            raise self.history_error
        after_id = after.id if isinstance(after, discord.Object) else None
        for message in self.messages:
            if after_id is None or message.id > after_id:
                yield message

    async def archived_threads(
        self, *, limit: int | None
    ) -> AsyncIterator[FakeChannel | discord.Thread]:
        if self.archived_error is not None:
            raise self.archived_error
        for thread in self.archived:
            yield thread

    async def fetch_message(self, message_id: int) -> FakeMessage:
        if self.fetch_message_error is not None:
            raise self.fetch_message_error
        assert self.fetch_message_result is not None
        return self.fetch_message_result


@dataclass
class FakePermissions:
    view_channel: bool
    read_message_history: bool


@dataclass
class FakeRole:
    id: int
    name: str


@dataclass
class FakeMember:
    id: int
    roles: tuple[FakeRole, ...]


@dataclass
class FakeGuild:
    id: int
    text_channels: tuple[FakeFetchableChannel, ...] = ()
    threads: tuple[discord.Thread, ...] = ()
    roles: tuple[FakeRole, ...] = ()
    members: tuple[FakeMember, ...] = ()
    me: object = None


@dataclass
class FakeClient:
    guilds: dict[int, FakeGuild] = field(default_factory=dict)
    channels: dict[int, FakeFetchableChannel] = field(default_factory=dict)
    fetch_channel_result: FakeFetchableChannel | None = None
    fetch_channel_error: Exception | None = None

    def get_guild(self, guild_id: int) -> FakeGuild | None:
        return self.guilds.get(guild_id)

    def get_channel(self, channel_id: int) -> FakeFetchableChannel | None:
        return self.channels.get(channel_id)

    async def fetch_channel(self, channel_id: int) -> FakeFetchableChannel:
        if self.fetch_channel_error is not None:
            raise self.fetch_channel_error
        assert self.fetch_channel_result is not None
        return self.fetch_channel_result


async def collect_history(
    source: DiscordPySource, channel_id: int, after_id: int | None, page_size: int
) -> list[tuple[SourceMessage, ...]]:
    pages: list[tuple[SourceMessage, ...]] = []
    async for page in source.history(channel_id, after_id, page_size):
        pages.append(tuple(page))
    return pages


async def test_list_channels_includes_text_active_and_archived_threads() -> None:
    guild_ref = FakeGuildRef(100)
    archived_thread = bare_thread(30, 100, parent_id=10, name="closed", archived=True)
    active_thread = bare_thread(20, 100, parent_id=10, name="open", archived=False)
    text_channel = FakeFetchableChannel(10, guild_ref, "general", archived=[archived_thread])
    guild = FakeGuild(100, text_channels=(text_channel,), threads=(active_thread,))
    client = FakeClient(guilds={100: guild})
    source = DiscordPySource(client)
    channels = await source.list_channels(100)
    assert {c.id for c in channels} == {10, 20, 30}
    by_id = {c.id: c for c in channels}
    assert by_id[10].kind == ChannelKind.TEXT
    assert by_id[20].archived is False
    assert by_id[30].archived is True


async def test_list_channels_unknown_guild_returns_empty() -> None:
    client = FakeClient()
    source = DiscordPySource(client)
    assert await source.list_channels(999) == ()


async def test_list_channels_maps_rate_limit_error() -> None:
    guild_ref = FakeGuildRef(100)
    text_channel = FakeFetchableChannel(
        10, guild_ref, "general", archived_error=make_http_exception(429, retry_after="2.5")
    )
    guild = FakeGuild(100, text_channels=(text_channel,))
    client = FakeClient(guilds={100: guild})
    source = DiscordPySource(client)
    try:
        await source.list_channels(100)
        raise AssertionError("expected SourceRateLimitedError")
    except SourceRateLimitedError as error:
        assert error.retry_after == 2.5


async def test_list_channels_maps_other_http_error() -> None:
    guild_ref = FakeGuildRef(100)
    text_channel = FakeFetchableChannel(
        10, guild_ref, "general", archived_error=make_http_exception(500)
    )
    guild = FakeGuild(100, text_channels=(text_channel,))
    client = FakeClient(guilds={100: guild})
    source = DiscordPySource(client)
    try:
        await source.list_channels(100)
        raise AssertionError("expected SourceUnavailableError")
    except SourceUnavailableError:
        pass


async def test_list_channels_maps_connection_error() -> None:
    guild_ref = FakeGuildRef(100)
    text_channel = FakeFetchableChannel(
        10, guild_ref, "general", archived_error=ConnectionError("no route")
    )
    guild = FakeGuild(100, text_channels=(text_channel,))
    client = FakeClient(guilds={100: guild})
    source = DiscordPySource(client)
    try:
        await source.list_channels(100)
        raise AssertionError("expected SourceUnavailableError")
    except SourceUnavailableError:
        pass


async def test_history_pages_and_respects_after_id() -> None:
    channel = FakeFetchableChannel(
        10, FakeGuildRef(100), "general", messages=[make_message(id=i) for i in range(1, 6)]
    )
    client = FakeClient(channels={10: channel})
    source = DiscordPySource(client)
    pages = await collect_history(source, 10, None, 2)
    assert [msg.id for page in pages for msg in page] == [1, 2, 3, 4, 5]
    assert [len(page) for page in pages] == [2, 2, 1]


async def test_history_pages_evenly_with_no_leftover() -> None:
    channel = FakeFetchableChannel(
        10, FakeGuildRef(100), "general", messages=[make_message(id=i) for i in range(1, 5)]
    )
    client = FakeClient(channels={10: channel})
    source = DiscordPySource(client)
    pages = await collect_history(source, 10, None, 2)
    assert [len(page) for page in pages] == [2, 2]


async def test_history_respects_after_id_cursor() -> None:
    channel = FakeFetchableChannel(
        10, FakeGuildRef(100), "general", messages=[make_message(id=i) for i in range(1, 4)]
    )
    client = FakeClient(channels={10: channel})
    source = DiscordPySource(client)
    pages = await collect_history(source, 10, 1, 10)
    assert [msg.id for page in pages for msg in page] == [2, 3]


async def test_history_falls_back_to_fetch_channel_when_not_cached() -> None:
    channel = FakeFetchableChannel(
        10, FakeGuildRef(100), "general", messages=[make_message(id=i) for i in range(1, 3)]
    )
    client = FakeClient(fetch_channel_result=channel)
    source = DiscordPySource(client)
    pages = await collect_history(source, 10, None, 10)
    assert [msg.id for page in pages for msg in page] == [1, 2]


async def test_history_fetch_channel_not_found_raises_source_not_found_error() -> None:
    client = FakeClient(fetch_channel_error=make_http_exception(404, cls=discord.NotFound))
    source = DiscordPySource(client)
    try:
        await collect_history(source, 999, None, 10)
        raise AssertionError("expected SourceNotFoundError")
    except SourceNotFoundError as error:
        assert isinstance(error, SourceUnavailableError)


async def test_history_fetch_channel_forbidden_raises_source_forbidden_error() -> None:
    client = FakeClient(fetch_channel_error=make_http_exception(403, cls=discord.Forbidden))
    source = DiscordPySource(client)
    try:
        await collect_history(source, 999, None, 10)
        raise AssertionError("expected SourceForbiddenError")
    except SourceForbiddenError as error:
        assert isinstance(error, SourceUnavailableError)


async def test_history_fetch_channel_other_http_error_raises_unavailable() -> None:
    client = FakeClient(fetch_channel_error=make_http_exception(500))
    source = DiscordPySource(client)
    try:
        await collect_history(source, 999, None, 10)
        raise AssertionError("expected SourceUnavailableError")
    except SourceForbiddenError:
        raise AssertionError("did not expect SourceForbiddenError") from None
    except SourceNotFoundError:
        raise AssertionError("did not expect SourceNotFoundError") from None
    except SourceUnavailableError:
        pass


async def test_history_fetch_channel_connection_error_raises_unavailable() -> None:
    client = FakeClient(fetch_channel_error=OSError("no route"))
    source = DiscordPySource(client)
    try:
        await collect_history(source, 999, None, 10)
        raise AssertionError("expected SourceUnavailableError")
    except SourceUnavailableError:
        pass


async def test_history_maps_rate_limit_error() -> None:
    channel = FakeFetchableChannel(
        10, FakeGuildRef(100), "general", history_error=make_http_exception(429, retry_after="1.0")
    )
    client = FakeClient(channels={10: channel})
    source = DiscordPySource(client)
    try:
        await collect_history(source, 10, None, 10)
        raise AssertionError("expected SourceRateLimitedError")
    except SourceRateLimitedError as error:
        assert error.retry_after == 1.0


async def test_history_maps_connection_error() -> None:
    channel = FakeFetchableChannel(10, FakeGuildRef(100), "general", history_error=OSError("down"))
    client = FakeClient(channels={10: channel})
    source = DiscordPySource(client)
    try:
        await collect_history(source, 10, None, 10)
        raise AssertionError("expected SourceUnavailableError")
    except SourceUnavailableError:
        pass


async def test_role_member_ids_filters_by_role_name() -> None:
    role = FakeRole(1, "opted-out")
    other_role = FakeRole(2, "other")
    member_with_role = FakeMember(5, (role,))
    member_without_role = FakeMember(6, (other_role,))
    guild = FakeGuild(
        100, roles=(role, other_role), members=(member_with_role, member_without_role)
    )
    client = FakeClient(guilds={100: guild})
    source = DiscordPySource(client)
    ids = await source.role_member_ids(100, "opted-out")
    assert ids == frozenset({5})


async def test_role_member_ids_unknown_guild_returns_empty() -> None:
    client = FakeClient()
    source = DiscordPySource(client)
    assert await source.role_member_ids(999, "opted-out") == frozenset()


async def test_role_member_ids_unknown_role_returns_empty() -> None:
    guild = FakeGuild(100, roles=())
    client = FakeClient(guilds={100: guild})
    source = DiscordPySource(client)
    assert await source.role_member_ids(100, "missing") == frozenset()


@dataclass
class FakeRawMessageRef:
    channel_id: int
    message_id: int


@dataclass
class FakeRawReaction:
    channel_id: int
    message_id: int
    emoji: object


async def collect_events(source: DiscordPySource) -> list[object]:
    events: list[object] = []
    async for event in source.events():
        events.append(event)
    return events


async def test_events_on_message_relays_created() -> None:
    client = FakeClient()
    source = DiscordPySource(client)
    message = make_message()
    await client.on_message(message)  # type: ignore[attr-defined]
    source.close()
    events = await collect_events(source)
    assert len(events) == 1
    assert events[0].message.id == message.id  # type: ignore[attr-defined]


async def test_events_on_raw_message_edit_fetches_and_relays() -> None:
    edited_message = make_message(content="edited")
    channel = FakeFetchableChannel(
        10, FakeGuildRef(100), "general", fetch_message_result=edited_message
    )
    client = FakeClient(channels={10: channel})
    source = DiscordPySource(client)
    await client.on_raw_message_edit(FakeRawMessageRef(10, 1))  # type: ignore[attr-defined]
    source.close()
    events = await collect_events(source)
    assert len(events) == 1
    assert events[0].message.content == "edited"  # type: ignore[attr-defined]


async def test_events_on_raw_message_edit_dropped_when_message_gone() -> None:
    channel = FakeFetchableChannel(
        10, FakeGuildRef(100), "general", fetch_message_error=make_http_exception(404)
    )
    client = FakeClient(channels={10: channel})
    source = DiscordPySource(client)
    await client.on_raw_message_edit(FakeRawMessageRef(10, 1))  # type: ignore[attr-defined]
    source.close()
    events = await collect_events(source)
    assert events == []


async def test_events_on_raw_message_edit_unknown_channel_is_dropped() -> None:
    client = FakeClient(fetch_channel_error=make_http_exception(404, cls=discord.NotFound))
    source = DiscordPySource(client)
    await client.on_raw_message_edit(FakeRawMessageRef(999, 1))  # type: ignore[attr-defined]
    source.close()
    events = await collect_events(source)
    assert events == []


async def test_events_on_raw_message_edit_fetches_channel_when_not_cached() -> None:
    edited_message = make_message(content="edited")
    channel = FakeFetchableChannel(
        10, FakeGuildRef(100), "general", fetch_message_result=edited_message
    )
    client = FakeClient(fetch_channel_result=channel)
    source = DiscordPySource(client)
    await client.on_raw_message_edit(FakeRawMessageRef(10, 1))  # type: ignore[attr-defined]
    source.close()
    events = await collect_events(source)
    assert len(events) == 1
    assert events[0].message.content == "edited"  # type: ignore[attr-defined]


async def test_events_on_raw_message_delete_relays() -> None:
    client = FakeClient()
    source = DiscordPySource(client)
    await client.on_raw_message_delete(FakeRawMessageRef(10, 1))  # type: ignore[attr-defined]
    source.close()
    events = await collect_events(source)
    assert len(events) == 1
    assert events[0].message_id == 1  # type: ignore[attr-defined]
    assert events[0].channel_id == 10  # type: ignore[attr-defined]


async def test_events_on_raw_reaction_add_relays_current_count() -> None:
    message = make_message(reactions=(FakeReaction("👍", 3),))
    channel = FakeFetchableChannel(10, FakeGuildRef(100), "general", fetch_message_result=message)
    client = FakeClient(channels={10: channel})
    source = DiscordPySource(client)
    await client.on_raw_reaction_add(FakeRawReaction(10, 1, "👍"))  # type: ignore[attr-defined]
    source.close()
    events = await collect_events(source)
    assert len(events) == 1
    assert events[0].emoji == "👍"  # type: ignore[attr-defined]
    assert events[0].count == 3  # type: ignore[attr-defined]


async def test_events_on_raw_reaction_add_skips_non_matching_reactions() -> None:
    message = make_message(reactions=(FakeReaction("😀", 1), FakeReaction("👍", 5)))
    channel = FakeFetchableChannel(10, FakeGuildRef(100), "general", fetch_message_result=message)
    client = FakeClient(channels={10: channel})
    source = DiscordPySource(client)
    await client.on_raw_reaction_add(FakeRawReaction(10, 1, "👍"))  # type: ignore[attr-defined]
    source.close()
    events = await collect_events(source)
    assert events[0].count == 5  # type: ignore[attr-defined]


async def test_events_on_raw_reaction_remove_zero_when_reaction_gone() -> None:
    message = make_message(reactions=())
    channel = FakeFetchableChannel(10, FakeGuildRef(100), "general", fetch_message_result=message)
    client = FakeClient(channels={10: channel})
    source = DiscordPySource(client)
    await client.on_raw_reaction_remove(FakeRawReaction(10, 1, "👍"))  # type: ignore[attr-defined]
    source.close()
    events = await collect_events(source)
    assert events[0].count == 0  # type: ignore[attr-defined]


async def test_events_reaction_fetch_failure_defaults_zero_count() -> None:
    channel = FakeFetchableChannel(
        10, FakeGuildRef(100), "general", fetch_message_error=make_http_exception(404)
    )
    client = FakeClient(channels={10: channel})
    source = DiscordPySource(client)
    await client.on_raw_reaction_add(FakeRawReaction(10, 1, "👍"))  # type: ignore[attr-defined]
    source.close()
    events = await collect_events(source)
    assert events[0].count == 0  # type: ignore[attr-defined]


async def test_events_on_thread_create_relays() -> None:
    client = FakeClient()
    source = DiscordPySource(client)
    thread = bare_thread(20, 100, parent_id=10, name="new-thread", archived=False)
    await client.on_thread_create(thread)  # type: ignore[attr-defined]
    source.close()
    events = await collect_events(source)
    assert len(events) == 1
    assert events[0].channel.id == 20  # type: ignore[attr-defined]


async def test_events_close_ends_stream_immediately() -> None:
    client = FakeClient()
    source = DiscordPySource(client)
    source.close()
    events = await collect_events(source)
    assert events == []


def test_build_client_enables_required_intents() -> None:
    client = build_client()
    assert isinstance(client, discord.Client)
    assert client.intents.message_content is True
    assert client.intents.members is True
    assert client.intents.guilds is True
    assert client.intents.guild_messages is True
    assert client.intents.guild_reactions is True


async def test_list_channels_skips_channels_the_bot_cannot_read() -> None:
    guild_ref = FakeGuildRef(100)
    readable = FakeFetchableChannel(10, guild_ref, "general")
    hidden = FakeFetchableChannel(
        11, guild_ref, "mods", readable=False, archived_error=make_http_exception(403)
    )
    hidden_thread = bare_thread(21, 100, parent_id=11, name="secret", archived=False)
    open_thread = bare_thread(20, 100, parent_id=10, name="open", archived=False)
    guild = FakeGuild(100, text_channels=(readable, hidden), threads=(open_thread, hidden_thread))
    source = DiscordPySource(FakeClient(guilds={100: guild}))
    assert {c.id for c in await source.list_channels(100)} == {10, 20}


async def test_list_channels_skips_archived_threads_it_is_forbidden_to_list() -> None:
    guild_ref = FakeGuildRef(100)
    text_channel = FakeFetchableChannel(
        10, guild_ref, "general", archived_error=make_http_exception(403)
    )
    guild = FakeGuild(100, text_channels=(text_channel,))
    source = DiscordPySource(FakeClient(guilds={100: guild}))
    assert {c.id for c in await source.list_channels(100)} == {10}


async def test_history_forbidden_maps_to_source_forbidden_error() -> None:
    guild_ref = FakeGuildRef(100)
    channel = FakeFetchableChannel(10, guild_ref, "general", history_error=make_http_exception(403))
    source = DiscordPySource(FakeClient(channels={10: channel}))
    try:
        await collect_history(source, 10, None, 10)
        raise AssertionError("expected SourceForbiddenError")
    except SourceForbiddenError as error:
        assert isinstance(error, SourceUnavailableError)
