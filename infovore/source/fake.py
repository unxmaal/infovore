import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence

from infovore.source.protocol import (
    MessageCreated,
    MessageDeleted,
    MessageEdited,
    SourceChannel,
    SourceEvent,
    SourceMessage,
)


class FakeDiscordSource:
    def __init__(
        self,
        channels: Iterable[SourceChannel] = (),
        messages: Iterable[SourceMessage] = (),
        role_members: Mapping[int, Mapping[str, Iterable[int]]] | None = None,
    ) -> None:
        self._channels: dict[int, SourceChannel] = {channel.id: channel for channel in channels}
        self._messages: dict[int, dict[int, SourceMessage]] = {}
        for message in messages:
            self._store_message(message)
        self._role_members: dict[int, dict[str, frozenset[int]]] = {
            guild_id: {role: frozenset(ids) for role, ids in roles.items()}
            for guild_id, roles in (role_members or {}).items()
        }
        self._call_failures: dict[int | None, list[Exception]] = {}
        self._interrupts: dict[int, tuple[int, Exception]] = {}
        self._event_queue: asyncio.Queue[SourceEvent | None] = asyncio.Queue()

    def _store_message(self, message: SourceMessage) -> None:
        self._messages.setdefault(message.channel_id, {})[message.id] = message

    def _remove_message(self, channel_id: int, message_id: int) -> None:
        self._messages.get(channel_id, {}).pop(message_id, None)

    def add_channel(self, channel: SourceChannel) -> None:
        self._channels[channel.id] = channel

    def add_message(self, message: SourceMessage) -> None:
        self._store_message(message)

    def edit_message(self, message: SourceMessage) -> None:
        self._store_message(message)

    def delete_message(self, channel_id: int, message_id: int) -> None:
        self._remove_message(channel_id, message_id)

    def fail_next_history_call(self, error: Exception, channel_id: int | None = None) -> None:
        self._call_failures.setdefault(channel_id, []).append(error)

    def interrupt_history_after(self, channel_id: int, pages: int, error: Exception) -> None:
        self._interrupts[channel_id] = (pages, error)

    def _pop_call_failure(self, channel_id: int) -> Exception | None:
        for key in (None, channel_id):
            queue = self._call_failures.get(key)
            if queue:
                return queue.pop(0)
        return None

    def push(self, event: SourceEvent) -> None:
        if isinstance(event, MessageCreated | MessageEdited):
            self._store_message(event.message)
        elif isinstance(event, MessageDeleted):
            self._remove_message(event.channel_id, event.message_id)
        self._event_queue.put_nowait(event)

    def close(self) -> None:
        self._event_queue.put_nowait(None)

    async def list_channels(self, guild_id: int) -> Sequence[SourceChannel]:
        return tuple(channel for channel in self._channels.values() if channel.guild_id == guild_id)

    async def history(
        self, channel_id: int, after_id: int | None, page_size: int
    ) -> AsyncIterator[Sequence[SourceMessage]]:
        error = self._pop_call_failure(channel_id)
        if error is not None:
            raise error
        messages = [
            message
            for message in self._messages.get(channel_id, {}).values()
            if after_id is None or message.id > after_id
        ]
        messages.sort(key=lambda message: (message.created_at, message.id))
        interrupt = self._interrupts.pop(channel_id, None)
        for index, start in enumerate(range(0, len(messages), page_size)):
            if interrupt is not None and index == interrupt[0]:
                raise interrupt[1]
            yield tuple(messages[start : start + page_size])

    async def events(self) -> AsyncIterator[SourceEvent]:
        while True:
            event = await self._event_queue.get()
            if event is None:
                return
            yield event

    async def role_member_ids(self, guild_id: int, role_name: str) -> frozenset[int]:
        return self._role_members.get(guild_id, {}).get(role_name, frozenset())
