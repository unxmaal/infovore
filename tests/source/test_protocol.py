from datetime import UTC, datetime

from infovore.rows import ChannelKind
from infovore.source.protocol import (
    MessageCreated,
    MessageDeleted,
    MessageEdited,
    ReactionChanged,
    SourceAttachment,
    SourceChannel,
    SourceMessage,
    SourceRateLimitedError,
    SourceReaction,
    SourceUnavailableError,
    ThreadCreated,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_source_message_and_events_construct() -> None:
    message = SourceMessage(
        id=1,
        channel_id=10,
        guild_id=100,
        author_id=5,
        author_name="alice",
        author_is_bot=False,
        is_system=False,
        created_at=NOW,
        edited_at=None,
        content="hi",
        reply_to_id=None,
        thread_id=None,
        attachments=(SourceAttachment(7, "a.txt", "text/plain", 3, "https://x"),),
        reactions=(SourceReaction("👍", 2),),
        raw={"id": "1"},
    )
    channel = SourceChannel(10, 100, None, "general", ChannelKind.TEXT, False)
    events = [
        MessageCreated(message),
        MessageEdited(message),
        MessageDeleted(message_id=1, channel_id=10),
        ReactionChanged(message_id=1, emoji="👍", count=3),
        ThreadCreated(channel),
    ]
    assert len(events) == 5


def test_errors_carry_details() -> None:
    limited = SourceRateLimitedError(retry_after=1.5)
    assert limited.retry_after == 1.5
    assert "1.5" in str(limited)
    assert isinstance(SourceUnavailableError("down"), Exception)
