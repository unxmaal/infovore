import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Protocol

import discord

from infovore.rows import ChannelKind
from infovore.source.protocol import (
    MessageCreated,
    MessageDeleted,
    MessageEdited,
    ReactionChanged,
    SourceAttachment,
    SourceChannel,
    SourceEvent,
    SourceMessage,
    SourceRateLimitedError,
    SourceReaction,
    SourceUnavailableError,
    ThreadCreated,
)


class _GuildRefLike(Protocol):
    @property
    def id(self) -> int: ...


class _ChannelLike(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def guild(self) -> _GuildRefLike: ...

    @property
    def name(self) -> str: ...


class _MessageChannelLike(Protocol):
    @property
    def id(self) -> int: ...


class _MessageTypeLike(Protocol):
    @property
    def name(self) -> str: ...


class _AuthorLike(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def display_name(self) -> str: ...

    @property
    def bot(self) -> bool: ...


class _AttachmentLike(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def filename(self) -> str: ...

    @property
    def content_type(self) -> str | None: ...

    @property
    def size(self) -> int: ...

    @property
    def url(self) -> str: ...


class _ReactionLike(Protocol):
    @property
    def emoji(self) -> object: ...

    @property
    def count(self) -> int: ...


class _ReferenceLike(Protocol):
    @property
    def message_id(self) -> int | None: ...


class _MessageLike(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def channel(self) -> _MessageChannelLike: ...

    @property
    def guild(self) -> _GuildRefLike | None: ...

    @property
    def author(self) -> _AuthorLike: ...

    @property
    def content(self) -> str: ...

    @property
    def created_at(self) -> datetime: ...

    @property
    def edited_at(self) -> datetime | None: ...

    @property
    def reference(self) -> _ReferenceLike | None: ...

    @property
    def attachments(self) -> Sequence[_AttachmentLike]: ...

    @property
    def reactions(self) -> Sequence[_ReactionLike]: ...

    @property
    def type(self) -> _MessageTypeLike: ...

    def is_system(self) -> bool: ...


def to_source_channel(channel: _ChannelLike) -> SourceChannel:
    if isinstance(channel, discord.Thread):
        return SourceChannel(
            id=channel.id,
            guild_id=channel.guild.id,
            parent_id=channel.parent_id,
            name=channel.name,
            kind=ChannelKind.THREAD,
            archived=channel.archived,
        )
    return SourceChannel(
        id=channel.id,
        guild_id=channel.guild.id,
        parent_id=None,
        name=channel.name,
        kind=ChannelKind.TEXT,
        archived=False,
    )


def _build_raw(
    message: _MessageLike,
    guild_id: int,
    reply_to_id: int | None,
    attachments: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "id": message.id,
        "channel_id": message.channel.id,
        "guild_id": guild_id,
        "author": {
            "id": message.author.id,
            "name": message.author.display_name,
            "bot": message.author.bot,
        },
        "content": message.content,
        "created_at": message.created_at.isoformat(),
        "edited_at": message.edited_at.isoformat() if message.edited_at is not None else None,
        "reference": reply_to_id,
        "attachments": attachments,
        "type": message.type.name,
    }


def to_source_message(message: _MessageLike) -> SourceMessage:
    channel = message.channel
    thread_id = channel.id if isinstance(channel, discord.Thread) else None
    reference = message.reference
    reply_to_id = reference.message_id if reference is not None else None
    guild = message.guild
    guild_id = guild.id if guild is not None else 0
    attachments = tuple(
        SourceAttachment(a.id, a.filename, a.content_type, a.size, a.url)
        for a in message.attachments
    )
    raw_attachments: list[dict[str, object]] = [
        {
            "id": a.id,
            "filename": a.filename,
            "content_type": a.content_type,
            "size": a.size,
            "url": a.url,
        }
        for a in message.attachments
    ]
    reactions = tuple(SourceReaction(str(r.emoji), r.count) for r in message.reactions)
    raw = _build_raw(message, guild_id, reply_to_id, raw_attachments)
    return SourceMessage(
        id=message.id,
        channel_id=channel.id,
        guild_id=guild_id,
        author_id=message.author.id,
        author_name=message.author.display_name,
        author_is_bot=message.author.bot,
        is_system=message.is_system(),
        created_at=message.created_at,
        edited_at=message.edited_at,
        content=message.content,
        reply_to_id=reply_to_id,
        thread_id=thread_id,
        attachments=attachments,
        reactions=reactions,
        raw=raw,
    )


def _map_http_error(
    error: discord.HTTPException,
) -> SourceRateLimitedError | SourceUnavailableError:
    if error.status == 429:
        return SourceRateLimitedError(_retry_after(error))
    return SourceUnavailableError(str(error))


def _retry_after(error: discord.HTTPException) -> float:
    value = error.response.headers.get("Retry-After")
    return float(value) if value is not None else 0.0


class _FetchableChannelLike(Protocol):
    def history(
        self, *, limit: int | None, after: object | None, oldest_first: bool | None
    ) -> AsyncIterator[_MessageLike]: ...

    async def fetch_message(self, message_id: int) -> _MessageLike: ...


class _MessageableLike(_ChannelLike, _FetchableChannelLike, Protocol):
    def archived_threads(self, *, limit: int | None) -> AsyncIterator[_ChannelLike]: ...


class _RoleLike(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def name(self) -> str: ...


class _MemberLike(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def roles(self) -> Sequence[_RoleLike]: ...


class _GuildLike(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def text_channels(self) -> Sequence[_MessageableLike]: ...

    @property
    def threads(self) -> Sequence[_ChannelLike]: ...

    @property
    def roles(self) -> Sequence[_RoleLike]: ...

    @property
    def members(self) -> Sequence[_MemberLike]: ...


class _ClientLike(Protocol):
    def get_guild(self, guild_id: int) -> _GuildLike | None: ...

    def get_channel(self, channel_id: int) -> _FetchableChannelLike | None: ...


class _RawMessageRefLike(Protocol):
    @property
    def channel_id(self) -> int: ...

    @property
    def message_id(self) -> int: ...


class _RawReactionLike(Protocol):
    @property
    def channel_id(self) -> int: ...

    @property
    def message_id(self) -> int: ...

    @property
    def emoji(self) -> object: ...


class DiscordPySource:
    def __init__(self, client: _ClientLike) -> None:
        self._client = client
        self._queue: asyncio.Queue[SourceEvent | None] = asyncio.Queue()
        setattr(client, "on_message", self._on_message)
        setattr(client, "on_raw_message_edit", self._on_raw_message_edit)
        setattr(client, "on_raw_message_delete", self._on_raw_message_delete)
        setattr(client, "on_raw_reaction_add", self._on_raw_reaction_add)
        setattr(client, "on_raw_reaction_remove", self._on_raw_reaction_remove)
        setattr(client, "on_thread_create", self._on_thread_create)

    async def list_channels(self, guild_id: int) -> Sequence[SourceChannel]:
        guild = self._client.get_guild(guild_id)
        if guild is None:
            return ()
        channels = [to_source_channel(channel) for channel in guild.text_channels]
        channels += [to_source_channel(thread) for thread in guild.threads]
        try:
            for text_channel in guild.text_channels:
                async for archived in text_channel.archived_threads(limit=None):
                    channels.append(to_source_channel(archived))
        except discord.HTTPException as error:
            raise _map_http_error(error) from error
        except OSError as error:
            raise SourceUnavailableError(str(error)) from error
        return tuple(channels)

    async def history(
        self, channel_id: int, after_id: int | None, page_size: int
    ) -> AsyncIterator[Sequence[SourceMessage]]:
        channel = self._client.get_channel(channel_id)
        if channel is None:
            raise SourceUnavailableError(f"channel {channel_id} not found")
        after = discord.Object(after_id) if after_id is not None else None
        page: list[SourceMessage] = []
        try:
            async for message in channel.history(limit=None, after=after, oldest_first=True):
                page.append(to_source_message(message))
                if len(page) >= page_size:
                    yield tuple(page)
                    page = []
        except discord.HTTPException as error:
            raise _map_http_error(error) from error
        except OSError as error:
            raise SourceUnavailableError(str(error)) from error
        if page:
            yield tuple(page)

    async def role_member_ids(self, guild_id: int, role_name: str) -> frozenset[int]:
        guild = self._client.get_guild(guild_id)
        if guild is None:
            return frozenset()
        role = next((candidate for candidate in guild.roles if candidate.name == role_name), None)
        if role is None:
            return frozenset()
        return frozenset(
            member.id
            for member in guild.members
            if any(candidate.id == role.id for candidate in member.roles)
        )

    async def events(self) -> AsyncIterator[SourceEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def _fetch_message(self, channel_id: int, message_id: int) -> _MessageLike | None:
        channel = self._client.get_channel(channel_id)
        if channel is None:
            return None
        try:
            return await channel.fetch_message(message_id)
        except discord.HTTPException:
            return None

    async def _on_message(self, message: _MessageLike) -> None:
        self._queue.put_nowait(MessageCreated(to_source_message(message)))

    async def _on_raw_message_edit(self, payload: _RawMessageRefLike) -> None:
        message = await self._fetch_message(payload.channel_id, payload.message_id)
        if message is not None:
            self._queue.put_nowait(MessageEdited(to_source_message(message)))

    async def _on_raw_message_delete(self, payload: _RawMessageRefLike) -> None:
        self._queue.put_nowait(
            MessageDeleted(message_id=payload.message_id, channel_id=payload.channel_id)
        )

    async def _on_raw_reaction_add(self, payload: _RawReactionLike) -> None:
        await self._push_reaction(payload)

    async def _on_raw_reaction_remove(self, payload: _RawReactionLike) -> None:
        await self._push_reaction(payload)

    async def _push_reaction(self, payload: _RawReactionLike) -> None:
        message = await self._fetch_message(payload.channel_id, payload.message_id)
        emoji_str = str(payload.emoji)
        count = 0
        if message is not None:
            for reaction in message.reactions:
                if str(reaction.emoji) == emoji_str:
                    count = reaction.count
                    break
        self._queue.put_nowait(
            ReactionChanged(message_id=payload.message_id, emoji=emoji_str, count=count)
        )

    async def _on_thread_create(self, thread: _ChannelLike) -> None:
        self._queue.put_nowait(ThreadCreated(to_source_channel(thread)))
