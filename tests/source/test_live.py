import json
from dataclasses import dataclass
from datetime import UTC, datetime

import discord

from infovore.rows import ChannelKind
from infovore.source.live import to_source_channel, to_source_message

NOW = datetime(2026, 1, 1, tzinfo=UTC)


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
    result = to_source_message(message)  # type: ignore[arg-type]
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
    result = to_source_message(message)  # type: ignore[arg-type]
    assert result.is_system is True
    assert result.raw["type"] == "pins_add"


def test_to_source_message_reply() -> None:
    message = make_message(reference=FakeReference(message_id=99))
    result = to_source_message(message)  # type: ignore[arg-type]
    assert result.reply_to_id == 99
    assert result.raw["reference"] == 99


def test_to_source_message_reply_reference_without_message_id() -> None:
    message = make_message(reference=FakeReference(message_id=None))
    result = to_source_message(message)  # type: ignore[arg-type]
    assert result.reply_to_id is None


def test_to_source_message_in_thread() -> None:
    thread = bare_thread(20, 100, parent_id=10, name="sub-thread", archived=False)
    message = make_message(channel=thread)
    result = to_source_message(message)  # type: ignore[arg-type]
    assert result.channel_id == 20
    assert result.thread_id == 20


def test_to_source_message_attachments_and_reactions() -> None:
    message = make_message(
        attachments=(FakeAttachment(7, "a.txt", "text/plain", 3, "https://x"),),
        reactions=(FakeReaction("👍", 2),),
    )
    result = to_source_message(message)  # type: ignore[arg-type]
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
    result = to_source_message(message)  # type: ignore[arg-type]
    assert result.edited_at == edited_at
    assert result.raw["edited_at"] == edited_at.isoformat()


def test_to_source_message_bot_author() -> None:
    message = make_message(author=FakeAuthor(9, "helper-bot", True))
    result = to_source_message(message)  # type: ignore[arg-type]
    assert result.author_is_bot is True
    assert result.raw["author"] == {"id": 9, "name": "helper-bot", "bot": True}


def test_to_source_message_no_guild_defaults_zero() -> None:
    message = make_message(guild=None)
    result = to_source_message(message)  # type: ignore[arg-type]
    assert result.guild_id == 0
    assert result.raw["guild_id"] == 0


def test_to_source_message_raw_is_json_safe() -> None:
    message = make_message()
    result = to_source_message(message)  # type: ignore[arg-type]
    json.dumps(result.raw)
