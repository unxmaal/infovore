from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from infovore.rows import ChannelKind


@dataclass(frozen=True)
class SourceAttachment:
    id: int
    filename: str
    content_type: str | None
    size: int
    url: str


@dataclass(frozen=True)
class SourceReaction:
    emoji: str
    count: int


@dataclass(frozen=True)
class SourceMessage:
    id: int
    channel_id: int
    guild_id: int
    author_id: int
    author_name: str
    author_is_bot: bool
    is_system: bool
    created_at: datetime
    edited_at: datetime | None
    content: str
    reply_to_id: int | None
    thread_id: int | None
    attachments: tuple[SourceAttachment, ...]
    reactions: tuple[SourceReaction, ...]
    raw: Mapping[str, object]


@dataclass(frozen=True)
class SourceChannel:
    id: int
    guild_id: int
    parent_id: int | None
    name: str
    kind: ChannelKind
    archived: bool


@dataclass(frozen=True)
class MessageCreated:
    message: SourceMessage


@dataclass(frozen=True)
class MessageEdited:
    message: SourceMessage


@dataclass(frozen=True)
class MessageDeleted:
    message_id: int
    channel_id: int


@dataclass(frozen=True)
class ReactionChanged:
    message_id: int
    emoji: str
    count: int


@dataclass(frozen=True)
class ThreadCreated:
    channel: SourceChannel


SourceEvent = MessageCreated | MessageEdited | MessageDeleted | ReactionChanged | ThreadCreated


class SourceRateLimitedError(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited; retry after {retry_after}s")
        self.retry_after = retry_after


class SourceUnavailableError(Exception):
    pass


class SourceForbiddenError(SourceUnavailableError):
    pass


class SourceNotFoundError(SourceUnavailableError):
    pass


class DiscordSource(Protocol):
    async def list_channels(self, guild_id: int) -> Sequence[SourceChannel]: ...

    def history(
        self, channel_id: int, after_id: int | None, page_size: int
    ) -> AsyncIterator[Sequence[SourceMessage]]: ...

    def events(self) -> AsyncIterator[SourceEvent]: ...

    async def role_member_ids(self, guild_id: int, role_name: str) -> frozenset[int]: ...
