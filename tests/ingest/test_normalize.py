import json
from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from infovore.ingest.normalize import NormalizedMessage, normalize_channel, normalize_message
from infovore.rows import ChannelKind, ReactionRow
from infovore.source.protocol import SourceAttachment, SourceChannel, SourceMessage, SourceReaction

NOW = datetime(2026, 1, 1, tzinfo=UTC)
INGESTED_AT = datetime(2026, 1, 2, tzinfo=UTC)


def make_source_message(**overrides: object) -> SourceMessage:
    fields: dict[str, object] = {
        "id": 1,
        "channel_id": 10,
        "guild_id": 100,
        "author_id": 5,
        "author_name": "alice",
        "author_is_bot": False,
        "is_system": False,
        "created_at": NOW,
        "edited_at": None,
        "content": "hello",
        "reply_to_id": None,
        "thread_id": None,
        "attachments": (),
        "reactions": (),
        "raw": {"id": "1"},
    }
    fields.update(overrides)
    return SourceMessage(**fields)  # type: ignore[arg-type]


def test_normalize_message_basic_fields() -> None:
    message = make_source_message()
    result = normalize_message(message, INGESTED_AT, include_bots=False)
    assert result is not None
    row = result.message
    assert row.id == 1
    assert row.channel_id == 10
    assert row.guild_id == 100
    assert row.author_id == 5
    assert row.author_name_at_time == "alice"
    assert row.author_is_bot is False
    assert row.created_at == NOW
    assert row.edited_at is None
    assert row.content == "hello"
    assert row.reply_to_id is None
    assert row.thread_id is None
    assert row.deleted_at is None
    assert row.ingested_at == INGESTED_AT
    assert row.raw_json == '{"id":"1"}'
    assert result.attachments == ()
    assert result.reactions == ()


def test_normalize_message_reply() -> None:
    message = make_source_message(reply_to_id=42)
    result = normalize_message(message, INGESTED_AT, include_bots=False)
    assert result is not None
    assert result.message.reply_to_id == 42


def test_normalize_message_thread_keeps_channel_and_thread_ids_as_given() -> None:
    message = make_source_message(channel_id=555, thread_id=555)
    result = normalize_message(message, INGESTED_AT, include_bots=False)
    assert result is not None
    assert result.message.channel_id == 555
    assert result.message.thread_id == 555


def test_normalize_channel_for_thread_parent() -> None:
    thread_channel = SourceChannel(
        id=555, guild_id=100, parent_id=10, name="thread-name", kind=ChannelKind.THREAD, archived=False
    )
    row = normalize_channel(thread_channel)
    assert row.id == 555
    assert row.parent_id == 10
    assert row.kind == ChannelKind.THREAD
    assert row.archived is False
    assert row.last_backfilled_message_id is None


def test_normalize_channel_text() -> None:
    channel = SourceChannel(
        id=10, guild_id=100, parent_id=None, name="general", kind=ChannelKind.TEXT, archived=False
    )
    row = normalize_channel(channel)
    assert row.id == 10
    assert row.guild_id == 100
    assert row.parent_id is None
    assert row.name == "general"
    assert row.kind == ChannelKind.TEXT
    assert row.archived is False
    assert row.last_backfilled_message_id is None


def test_normalize_message_attachments() -> None:
    attachment = SourceAttachment(id=7, filename="a.txt", content_type="text/plain", size=3, url="https://x")
    message = make_source_message(attachments=(attachment,))
    result = normalize_message(message, INGESTED_AT, include_bots=False)
    assert result is not None
    assert len(result.attachments) == 1
    row = result.attachments[0]
    assert row.id == 7
    assert row.message_id == 1
    assert row.filename == "a.txt"
    assert row.content_type == "text/plain"
    assert row.size == 3
    assert row.url == "https://x"
    assert row.sha256 is None
    assert row.local_path is None


def test_normalize_message_reactions_drops_non_positive_counts() -> None:
    message = make_source_message(
        reactions=(
            SourceReaction("👍", 2),
            SourceReaction("👎", 0),
            SourceReaction("😀", -1),
        )
    )
    result = normalize_message(message, INGESTED_AT, include_bots=False)
    assert result is not None
    assert result.reactions == (ReactionRow(message_id=1, emoji="👍", count=2),)


def test_normalize_message_edited() -> None:
    edited_at = datetime(2026, 1, 3, tzinfo=UTC)
    message = make_source_message(edited_at=edited_at)
    result = normalize_message(message, INGESTED_AT, include_bots=False)
    assert result is not None
    assert result.message.edited_at == edited_at


def test_normalize_message_system_is_skipped() -> None:
    message = make_source_message(is_system=True)
    assert normalize_message(message, INGESTED_AT, include_bots=True) is None


def test_normalize_message_bot_skipped_by_default() -> None:
    message = make_source_message(author_is_bot=True)
    assert normalize_message(message, INGESTED_AT, include_bots=False) is None


def test_normalize_message_bot_kept_when_include_bots() -> None:
    message = make_source_message(author_is_bot=True)
    result = normalize_message(message, INGESTED_AT, include_bots=True)
    assert result is not None
    assert result.message.author_is_bot is True


def test_normalize_message_non_bot_kept_regardless_of_include_bots() -> None:
    message = make_source_message(author_is_bot=False)
    assert normalize_message(message, INGESTED_AT, include_bots=True) is not None


def test_raw_json_is_deterministic_regardless_of_key_order() -> None:
    message_a = make_source_message(raw={"b": 1, "a": 2})
    message_b = make_source_message(raw={"a": 2, "b": 1})
    result_a = normalize_message(message_a, INGESTED_AT, include_bots=False)
    result_b = normalize_message(message_b, INGESTED_AT, include_bots=False)
    assert result_a is not None
    assert result_b is not None
    assert result_a.message.raw_json == result_b.message.raw_json
    assert json.loads(result_a.message.raw_json) == {"a": 2, "b": 1}


def test_naive_created_at_is_rejected() -> None:
    message = make_source_message(created_at=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="timezone"):
        normalize_message(message, INGESTED_AT, include_bots=False)


def test_naive_edited_at_is_rejected() -> None:
    message = make_source_message(edited_at=datetime(2026, 1, 3))
    with pytest.raises(ValueError, match="timezone"):
        normalize_message(message, INGESTED_AT, include_bots=False)


def test_naive_ingested_at_is_rejected() -> None:
    message = make_source_message()
    with pytest.raises(ValueError, match="timezone"):
        normalize_message(message, datetime(2026, 1, 2), include_bots=False)


json_scalars = st.one_of(st.none(), st.booleans(), st.integers(), st.text())
json_values = st.recursive(
    json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=3), st.dictionaries(st.text(), children, max_size=3)
    ),
    max_leaves=5,
)


@given(
    message_id=st.integers(min_value=1, max_value=10_000_000),
    content=st.text(max_size=50),
    raw=st.dictionaries(st.text(min_size=1, max_size=10), json_values, max_size=5),
    author_is_bot=st.booleans(),
    include_bots=st.booleans(),
)
def test_normalizing_same_message_twice_is_equal(
    message_id: int, content: str, raw: dict[str, object], author_is_bot: bool, include_bots: bool
) -> None:
    message = make_source_message(
        id=message_id, content=content, raw=raw, author_is_bot=author_is_bot, is_system=False
    )
    first = normalize_message(message, INGESTED_AT, include_bots)
    second = normalize_message(message, INGESTED_AT, include_bots)
    assert first == second
    if isinstance(first, NormalizedMessage):
        assert isinstance(second, NormalizedMessage)
        assert first.message.raw_json == second.message.raw_json
