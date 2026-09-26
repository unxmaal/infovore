import json
from pathlib import Path

import pytest

from infovore.rows import ChannelKind
from infovore.source.export import ExportDiscordSource
from infovore.source.protocol import SourceUnavailableError

GUILD_ID = 900
CHANNEL_ID = 100
BASE_TIMESTAMP = "2023-08-01T12:00:00.0000000+00:00"


def make_author(
    author_id: int,
    name: str = "alice",
    nickname: str = "Alice N.",
    is_bot: bool = False,
    roles: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": str(author_id),
        "name": name,
        "discriminator": "0000",
        "nickname": nickname,
        "color": None,
        "isBot": is_bot,
        "roles": roles or [],
        "avatarUrl": "https://example.invalid/avatar.png",
    }


def make_message(
    message_id: int,
    *,
    kind: str = "Default",
    timestamp: str = BASE_TIMESTAMP,
    timestamp_edited: str | None = None,
    content: str = "hello",
    author: dict[str, object] | None = None,
    attachments: list[dict[str, object]] | None = None,
    reactions: list[dict[str, object]] | None = None,
    reference: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": str(message_id),
        "type": kind,
        "timestamp": timestamp,
        "timestampEdited": timestamp_edited,
        "callEndedTimestamp": None,
        "isPinned": False,
        "content": content,
        "author": author or make_author(1),
        "attachments": attachments or [],
        "embeds": [],
        "stickers": [],
        "reactions": reactions or [],
        "mentions": [],
        **({"reference": reference} if reference is not None else {}),
        "inlineEmojis": [],
    }


def make_export(
    *,
    guild_id: int = GUILD_ID,
    channel_id: int = CHANNEL_ID,
    channel_type: str = "GuildTextChat",
    category_id: int | None = None,
    category: str | None = None,
    channel_name: str = "general",
    messages: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "guild": {"id": str(guild_id), "name": "Test Guild", "iconUrl": "https://x/icon.png"},
        "channel": {
            "id": str(channel_id),
            "type": channel_type,
            "categoryId": str(category_id) if category_id is not None else None,
            "category": category,
            "name": channel_name,
            "topic": None,
        },
        "dateRange": {"after": None, "before": None},
        "exportedAt": BASE_TIMESTAMP,
        "messages": messages if messages is not None else [make_message(1)],
        "messageCount": len(messages) if messages is not None else 1,
    }


def write_export(path: Path, export: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export))


async def test_single_channel_lists_and_pages_messages(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(messages=[make_message(1), make_message(2), make_message(3)]),
    )
    source = ExportDiscordSource(tmp_path)
    channels = await source.list_channels(GUILD_ID)
    assert len(channels) == 1
    channel = channels[0]
    assert channel.id == CHANNEL_ID
    assert channel.guild_id == GUILD_ID
    assert channel.kind is ChannelKind.TEXT
    assert channel.parent_id is None
    assert channel.name == "general"


async def test_list_channels_filters_by_guild(tmp_path: Path) -> None:
    write_export(tmp_path / "general.json", make_export())
    source = ExportDiscordSource(tmp_path)
    assert await source.list_channels(GUILD_ID) != ()
    assert await source.list_channels(GUILD_ID + 1) == ()


async def test_history_pages_oldest_first(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(
            messages=[
                make_message(3, timestamp="2023-08-01T12:00:03.0000000+00:00"),
                make_message(1, timestamp="2023-08-01T12:00:01.0000000+00:00"),
                make_message(2, timestamp="2023-08-01T12:00:02.0000000+00:00"),
            ]
        ),
    )
    source = ExportDiscordSource(tmp_path)
    pages = []
    async for page in source.history(CHANNEL_ID, None, 2):
        pages.append(tuple(m.id for m in page))
    assert pages == [(1, 2), (3,)]


async def test_history_respects_after_id(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(messages=[make_message(i) for i in range(1, 6)]),
    )
    source = ExportDiscordSource(tmp_path)
    pages = []
    async for page in source.history(CHANNEL_ID, 3, 10):
        pages.append(tuple(m.id for m in page))
    assert pages == [(4, 5)]


async def test_history_unknown_channel_yields_nothing(tmp_path: Path) -> None:
    write_export(tmp_path / "general.json", make_export())
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(999, None, 10)]
    assert pages == []


async def test_events_stream_is_empty(tmp_path: Path) -> None:
    write_export(tmp_path / "general.json", make_export())
    source = ExportDiscordSource(tmp_path)
    events = [event async for event in source.events()]
    assert events == []


async def test_thread_channel_sets_kind_parent_and_message_thread_id(tmp_path: Path) -> None:
    parent_id = 55
    thread_id = 200
    write_export(
        tmp_path / "thread.json",
        make_export(
            channel_id=thread_id,
            channel_type="GuildPublicThread",
            category_id=parent_id,
            category="general",
            channel_name="a lore thread",
            messages=[make_message(10)],
        ),
    )
    source = ExportDiscordSource(tmp_path)
    channels = await source.list_channels(GUILD_ID)
    assert len(channels) == 1
    channel = channels[0]
    assert channel.kind is ChannelKind.THREAD
    assert channel.parent_id == parent_id
    pages = [page async for page in source.history(thread_id, None, 10)]
    assert pages[0][0].thread_id == thread_id


async def test_text_channel_category_id_is_not_treated_as_parent(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(category_id=42, category="folder"),
    )
    source = ExportDiscordSource(tmp_path)
    channels = await source.list_channels(GUILD_ID)
    assert channels[0].parent_id is None
    assert channels[0].kind is ChannelKind.TEXT


async def test_partitioned_channel_merges_and_dedupes_by_message_id(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(messages=[make_message(1), make_message(2)]),
    )
    write_export(
        tmp_path / "general [part 2].json",
        make_export(messages=[make_message(2), make_message(3)]),
    )
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    ids = [m.id for page in pages for m in page]
    assert ids == [1, 2, 3]


async def test_system_message_type_is_marked_system(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(messages=[make_message(1, kind="GuildMemberJoin")]),
    )
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    assert pages[0][0].is_system is True


async def test_default_and_reply_types_are_not_system(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(
            messages=[
                make_message(1, kind="Default"),
                make_message(
                    2,
                    kind="Reply",
                    timestamp="2023-08-01T12:00:02.0000000+00:00",
                    reference={
                        "type": "Default",
                        "messageId": "1",
                        "channelId": str(CHANNEL_ID),
                        "guildId": str(GUILD_ID),
                    },
                ),
            ]
        ),
    )
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    messages = pages[0]
    assert messages[0].is_system is False
    assert messages[1].is_system is False
    assert messages[1].reply_to_id == 1


async def test_reference_without_reply_type_still_sets_reply_to(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(
            messages=[
                make_message(
                    1,
                    kind="Default",
                    reference={
                        "type": "Default",
                        "messageId": "999",
                        "channelId": str(CHANNEL_ID),
                        "guildId": str(GUILD_ID),
                    },
                ),
            ]
        ),
    )
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    assert pages[0][0].reply_to_id == 999


async def test_reference_with_null_message_id_leaves_reply_to_none(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(
            messages=[
                make_message(
                    1,
                    reference={
                        "type": "Default",
                        "messageId": None,
                        "channelId": str(CHANNEL_ID),
                        "guildId": str(GUILD_ID),
                    },
                ),
            ]
        ),
    )
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    assert pages[0][0].reply_to_id is None


async def test_attachments_and_reactions_are_mapped(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(
            messages=[
                make_message(
                    1,
                    attachments=[
                        {
                            "id": "555",
                            "url": "https://x/file.png",
                            "fileName": "file.png",
                            "fileSizeBytes": 1234,
                        }
                    ],
                    reactions=[
                        {
                            "emoji": {
                                "id": "0",
                                "name": "👍",
                                "code": "thumbsup",
                                "isAnimated": False,
                                "imageUrl": "",
                            },
                            "count": 4,
                            "users": [],
                        }
                    ],
                )
            ]
        ),
    )
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    message = pages[0][0]
    assert len(message.attachments) == 1
    attachment = message.attachments[0]
    assert attachment.id == 555
    assert attachment.filename == "file.png"
    assert attachment.content_type is None
    assert attachment.size == 1234
    assert attachment.url == "https://x/file.png"
    assert len(message.reactions) == 1
    assert message.reactions[0].emoji == "👍"
    assert message.reactions[0].count == 4


async def test_reaction_emoji_falls_back_to_code_when_name_blank(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(
            messages=[
                make_message(
                    1,
                    reactions=[
                        {
                            "emoji": {
                                "id": "0",
                                "name": "",
                                "code": "thumbsup",
                                "isAnimated": False,
                                "imageUrl": "",
                            },
                            "count": 1,
                            "users": [],
                        }
                    ],
                )
            ]
        ),
    )
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    assert pages[0][0].reactions[0].emoji == "thumbsup"


async def test_bot_author_is_marked(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(messages=[make_message(1, author=make_author(2, is_bot=True))]),
    )
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    assert pages[0][0].author_is_bot is True


async def test_nickname_preferred_over_name(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(messages=[make_message(1, author=make_author(2, name="alice", nickname="Al"))]),
    )
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    assert pages[0][0].author_name == "Al"


async def test_empty_nickname_falls_back_to_name(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(messages=[make_message(1, author=make_author(2, name="alice", nickname=""))]),
    )
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    assert pages[0][0].author_name == "alice"


async def test_edited_timestamp_is_parsed_when_present(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(
            messages=[
                make_message(
                    1, timestamp_edited="2023-08-01T13:00:00.0000000+00:00", content="edited"
                )
            ]
        ),
    )
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    message = pages[0][0]
    assert message.edited_at is not None
    assert message.edited_at.tzinfo is not None


async def test_role_member_ids_matches_case_insensitively(tmp_path: Path) -> None:
    write_export(
        tmp_path / "general.json",
        make_export(
            messages=[
                make_message(
                    1,
                    author=make_author(
                        2,
                        roles=[
                            {"id": "0", "name": "", "color": None, "position": 0},
                            {"id": "1", "name": "Moderator", "color": None, "position": 1},
                        ],
                    ),
                ),
                make_message(
                    2,
                    timestamp="2023-08-01T12:00:02.0000000+00:00",
                    author=make_author(3, roles=[]),
                ),
            ]
        ),
    )
    source = ExportDiscordSource(tmp_path)
    assert await source.role_member_ids(GUILD_ID, "moderator") == frozenset({2})
    assert await source.role_member_ids(GUILD_ID, "MODERATOR") == frozenset({2})
    assert await source.role_member_ids(GUILD_ID, "missing") == frozenset()
    assert await source.role_member_ids(GUILD_ID + 1, "moderator") == frozenset()


def test_guild_ids_reflects_all_guilds_seen(tmp_path: Path) -> None:
    write_export(tmp_path / "a.json", make_export(guild_id=1, channel_id=10))
    write_export(tmp_path / "b.json", make_export(guild_id=2, channel_id=20))
    source = ExportDiscordSource(tmp_path)
    assert source.guild_ids() == frozenset({1, 2})


def test_guild_ids_single_guild(tmp_path: Path) -> None:
    write_export(tmp_path / "a.json", make_export(guild_id=7))
    source = ExportDiscordSource(tmp_path)
    assert source.guild_ids() == frozenset({7})


def test_files_are_loaded_recursively(tmp_path: Path) -> None:
    write_export(tmp_path / "nested" / "deep" / "general.json", make_export())
    source = ExportDiscordSource(tmp_path)
    assert source.guild_ids() == frozenset({GUILD_ID})


def test_empty_directory_has_no_channels_or_guilds(tmp_path: Path) -> None:
    source = ExportDiscordSource(tmp_path)
    assert source.guild_ids() == frozenset()


def test_malformed_json_syntax_raises_source_unavailable_naming_file(tmp_path: Path) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text("{not json")
    with pytest.raises(SourceUnavailableError, match=r"broken\.json"):
        ExportDiscordSource(tmp_path)


def test_malformed_missing_messages_key_raises_source_unavailable(tmp_path: Path) -> None:
    export = make_export()
    del export["messages"]
    bad = tmp_path / "broken.json"
    write_export(bad, export)
    with pytest.raises(SourceUnavailableError, match=r"broken\.json"):
        ExportDiscordSource(tmp_path)


def test_malformed_message_missing_author_raises_source_unavailable(tmp_path: Path) -> None:
    message = make_message(1)
    del message["author"]
    bad = tmp_path / "broken.json"
    write_export(bad, make_export(messages=[message]))
    with pytest.raises(SourceUnavailableError, match=r"broken\.json"):
        ExportDiscordSource(tmp_path)


def test_malformed_bad_timestamp_raises_source_unavailable(tmp_path: Path) -> None:
    bad = tmp_path / "broken.json"
    write_export(bad, make_export(messages=[make_message(1, timestamp="not-a-timestamp")]))
    with pytest.raises(SourceUnavailableError, match=r"broken\.json"):
        ExportDiscordSource(tmp_path)


async def test_raw_is_the_messages_own_json_object(tmp_path: Path) -> None:
    write_export(tmp_path / "general.json", make_export(messages=[make_message(1)]))
    source = ExportDiscordSource(tmp_path)
    pages = [page async for page in source.history(CHANNEL_ID, None, 10)]
    message = pages[0][0]
    assert message.raw["id"] == "1"
    assert message.raw["type"] == "Default"
