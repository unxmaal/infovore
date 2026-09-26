import json
from dataclasses import dataclass
from datetime import datetime

from infovore.rows import AttachmentRow, ChannelRow, MessageRow, ReactionRow
from infovore.source.protocol import SourceChannel, SourceMessage


@dataclass(frozen=True)
class NormalizedMessage:
    message: MessageRow
    attachments: tuple[AttachmentRow, ...]
    reactions: tuple[ReactionRow, ...]


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")


def normalize_message(
    message: SourceMessage, ingested_at: datetime, include_bots: bool
) -> NormalizedMessage | None:
    if message.is_system:
        return None
    if message.author_is_bot and not include_bots:
        return None
    _require_aware(message.created_at)
    if message.edited_at is not None:
        _require_aware(message.edited_at)
    _require_aware(ingested_at)
    raw_json = json.dumps(message.raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    row = MessageRow(
        id=message.id,
        channel_id=message.channel_id,
        guild_id=message.guild_id,
        author_id=message.author_id,
        author_name_at_time=message.author_name,
        author_is_bot=message.author_is_bot,
        created_at=message.created_at,
        edited_at=message.edited_at,
        content=message.content,
        reply_to_id=message.reply_to_id,
        thread_id=message.thread_id,
        deleted_at=None,
        ingested_at=ingested_at,
        raw_json=raw_json,
    )
    attachments = tuple(
        AttachmentRow(
            id=attachment.id,
            message_id=message.id,
            filename=attachment.filename,
            content_type=attachment.content_type,
            size=attachment.size,
            url=attachment.url,
            sha256=None,
            local_path=None,
        )
        for attachment in message.attachments
    )
    reactions = tuple(
        ReactionRow(message_id=message.id, emoji=reaction.emoji, count=reaction.count)
        for reaction in message.reactions
        if reaction.count > 0
    )
    return NormalizedMessage(message=row, attachments=attachments, reactions=reactions)


def normalize_channel(channel: SourceChannel) -> ChannelRow:
    return ChannelRow(
        id=channel.id,
        guild_id=channel.guild_id,
        parent_id=channel.parent_id,
        name=channel.name,
        kind=channel.kind,
        archived=channel.archived,
        last_backfilled_message_id=None,
    )
