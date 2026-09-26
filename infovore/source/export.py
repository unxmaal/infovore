import json
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from infovore.rows import ChannelKind
from infovore.source.protocol import (
    SourceAttachment,
    SourceChannel,
    SourceEvent,
    SourceMessage,
    SourceReaction,
    SourceUnavailableError,
)

_NORMAL_TYPES = frozenset({"Default", "Reply"})


def _is_thread_type(channel_type: str) -> bool:
    return "Thread" in channel_type


def _emoji_string(emoji: Any) -> str:
    name = emoji.get("name")
    if isinstance(name, str) and name:
        return name
    code = emoji.get("code")
    return code if isinstance(code, str) else ""


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


class _NoEvents:
    def __aiter__(self) -> "_NoEvents":
        return self

    async def __anext__(self) -> SourceEvent:
        raise StopAsyncIteration


class ExportDiscordSource:
    def __init__(self, root: Path) -> None:
        self._channels: dict[int, SourceChannel] = {}
        self._messages: dict[int, dict[int, SourceMessage]] = {}
        self._guild_ids: set[int] = set()
        self._author_roles: dict[tuple[int, int], set[str]] = {}
        for path in sorted(root.rglob("*.json")):
            self._load_file(path)

    def _load_file(self, path: Path) -> None:
        try:
            data: Any = json.loads(path.read_text())
            guild_id = int(data["guild"]["id"])
            channel = data["channel"]
            channel_id = int(channel["id"])
            channel_type = channel["type"]
            is_thread = _is_thread_type(channel_type)
            category_id = channel.get("categoryId")
            parent_id = int(category_id) if is_thread and category_id else None
            source_channel = SourceChannel(
                id=channel_id,
                guild_id=guild_id,
                parent_id=parent_id,
                name=channel["name"],
                kind=ChannelKind.THREAD if is_thread else ChannelKind.TEXT,
                archived=False,  # DCE JSON export omits archived state; see README "Configuration".
            )
            messages_bucket = self._messages.setdefault(channel_id, {})
            for raw_message in data["messages"]:
                message = self._parse_message(raw_message, channel_id, guild_id, is_thread)
                messages_bucket[message.id] = message
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise SourceUnavailableError(f"malformed export file: {path}") from error
        self._guild_ids.add(guild_id)
        self._channels[channel_id] = source_channel

    def _parse_message(
        self,
        raw: Any,
        channel_id: int,
        guild_id: int,
        is_thread: bool,
    ) -> SourceMessage:
        message_id = int(raw["id"])
        kind = raw["type"]
        author = raw["author"]
        author_id = int(author["id"])
        nickname = author.get("nickname")
        name = author["name"]
        author_name = nickname if isinstance(nickname, str) and nickname else name
        for role in author.get("roles") or ():
            role_name = role.get("name")
            if isinstance(role_name, str) and role_name:
                self._author_roles.setdefault((guild_id, author_id), set()).add(role_name.lower())
        reference = raw.get("reference")
        reply_to_id: int | None = None
        if reference is not None and reference.get("messageId"):
            reply_to_id = int(reference["messageId"])
        created_at = _parse_timestamp(raw["timestamp"])
        edited_raw = raw.get("timestampEdited")
        edited_at = _parse_timestamp(edited_raw) if isinstance(edited_raw, str) else None
        attachments = tuple(
            SourceAttachment(
                id=int(a["id"]),
                filename=a["fileName"],
                content_type=None,
                size=int(a["fileSizeBytes"]),
                url=a["url"],
            )
            for a in raw.get("attachments") or ()
        )
        reactions = tuple(
            SourceReaction(emoji=_emoji_string(r["emoji"]), count=int(r["count"]))
            for r in raw.get("reactions") or ()
        )
        return SourceMessage(
            id=message_id,
            channel_id=channel_id,
            guild_id=guild_id,
            author_id=author_id,
            author_name=author_name,
            author_is_bot=bool(author.get("isBot", False)),
            is_system=kind not in _NORMAL_TYPES,
            created_at=created_at,
            edited_at=edited_at,
            content=raw["content"],
            reply_to_id=reply_to_id,
            thread_id=channel_id if is_thread else None,
            attachments=attachments,
            reactions=reactions,
            raw=raw,
        )

    async def list_channels(self, guild_id: int) -> Sequence[SourceChannel]:
        return tuple(channel for channel in self._channels.values() if channel.guild_id == guild_id)

    async def history(
        self, channel_id: int, after_id: int | None, page_size: int
    ) -> AsyncIterator[Sequence[SourceMessage]]:
        messages = [
            message
            for message in self._messages.get(channel_id, {}).values()
            if after_id is None or message.id > after_id
        ]
        messages.sort(key=lambda message: (message.created_at, message.id))
        for start in range(0, len(messages), page_size):
            yield tuple(messages[start : start + page_size])

    def events(self) -> AsyncIterator[SourceEvent]:
        return _NoEvents()

    async def role_member_ids(self, guild_id: int, role_name: str) -> frozenset[int]:
        lowered = role_name.lower()
        return frozenset(
            author_id
            for (g, author_id), roles in self._author_roles.items()
            if g == guild_id and lowered in roles
        )

    def guild_ids(self) -> frozenset[int]:
        return frozenset(self._guild_ids)
