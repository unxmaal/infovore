from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

import discord

from infovore.rows import ChannelKind
from infovore.source.protocol import (
    SourceAttachment,
    SourceChannel,
    SourceMessage,
    SourceReaction,
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
