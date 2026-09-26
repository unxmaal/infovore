import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from infovore.db.claims import retract_claims_with_all_sources_deleted
from infovore.db.exchanges import mark_stale_for_message
from infovore.db.raw import (
    get_channel,
    get_message,
    mark_deleted,
    mark_edited,
    set_reaction_count,
    upsert_attachment,
    upsert_channel,
    upsert_message,
)
from infovore.ingest.allowlist import is_channel_allowed
from infovore.ingest.normalize import NormalizedMessage, normalize_channel, normalize_message
from infovore.privacy.optout import opted_out_user_ids, redact_normalized
from infovore.source.protocol import (
    DiscordSource,
    MessageCreated,
    MessageDeleted,
    MessageEdited,
    ReactionChanged,
    SourceEvent,
    ThreadCreated,
)
from infovore.timing import Clock

logger = logging.getLogger(__name__)


class EventOutcome(StrEnum):
    MESSAGE_CREATED = "message_created"
    MESSAGE_SKIPPED = "message_skipped"
    MESSAGE_EDITED = "message_edited"
    MESSAGE_DELETED = "message_deleted"
    REACTION_UPDATED = "reaction_updated"
    REACTION_SKIPPED = "reaction_skipped"
    THREAD_CREATED = "thread_created"
    CHANNEL_IGNORED = "channel_ignored"


@dataclass
class ConsumeReport:
    processed: int = 0
    failed: int = 0
    ignored: int = 0


def _persist_message(conn: sqlite3.Connection, normalized: NormalizedMessage) -> None:
    upsert_message(conn, normalized.message)
    for attachment in normalized.attachments:
        upsert_attachment(conn, attachment)
    for reaction in normalized.reactions:
        set_reaction_count(conn, reaction.message_id, reaction.emoji, reaction.count)


async def _channel_allowed(
    conn: sqlite3.Connection,
    source: DiscordSource | None,
    channel_id: int,
    guild_id: int,
    channel_ids: Sequence[int],
) -> bool:
    if not channel_ids or channel_id in channel_ids:
        return True
    channel = get_channel(conn, channel_id)
    if channel is None and source is not None:
        for candidate in await source.list_channels(guild_id):
            if candidate.id == channel_id:
                upsert_channel(conn, normalize_channel(candidate))
                channel = get_channel(conn, channel_id)
                break
    if channel is None:
        return False
    return is_channel_allowed(channel_id, channel.parent_id, channel_ids)


async def _handle_message_created(
    conn: sqlite3.Connection,
    event: MessageCreated,
    clock: Clock,
    include_bots: bool,
    channel_ids: Sequence[int],
    source: DiscordSource | None,
) -> EventOutcome:
    message = event.message
    if not await _channel_allowed(conn, source, message.channel_id, message.guild_id, channel_ids):
        return EventOutcome.CHANNEL_IGNORED
    normalized = normalize_message(message, clock.now(), include_bots)
    if normalized is None:
        return EventOutcome.MESSAGE_SKIPPED
    normalized = redact_normalized(normalized, opted_out_user_ids(conn))
    _persist_message(conn, normalized)
    return EventOutcome.MESSAGE_CREATED


async def _handle_message_edited(
    conn: sqlite3.Connection,
    event: MessageEdited,
    clock: Clock,
    include_bots: bool,
    channel_ids: Sequence[int],
    source: DiscordSource | None,
) -> EventOutcome:
    message = event.message
    if not await _channel_allowed(conn, source, message.channel_id, message.guild_id, channel_ids):
        return EventOutcome.CHANNEL_IGNORED
    normalized = normalize_message(message, clock.now(), include_bots)
    if normalized is None:
        return EventOutcome.MESSAGE_SKIPPED
    normalized = redact_normalized(normalized, opted_out_user_ids(conn))
    row = normalized.message
    updated = mark_edited(conn, row.id, row.content, row.edited_at, row.raw_json)
    if not updated:
        _persist_message(conn, normalized)
        return EventOutcome.MESSAGE_CREATED
    mark_stale_for_message(conn, row.id)
    return EventOutcome.MESSAGE_EDITED


async def _handle_message_deleted(
    conn: sqlite3.Connection, event: MessageDeleted, clock: Clock, channel_ids: Sequence[int]
) -> EventOutcome:
    if channel_ids and event.channel_id not in channel_ids:
        channel = get_channel(conn, event.channel_id)
        parent_id = channel.parent_id if channel is not None else None
        if not is_channel_allowed(event.channel_id, parent_id, channel_ids):
            return EventOutcome.CHANNEL_IGNORED
    mark_deleted(conn, event.message_id, clock.now())
    retract_claims_with_all_sources_deleted(conn, clock.now())
    return EventOutcome.MESSAGE_DELETED


async def _handle_reaction_changed(
    conn: sqlite3.Connection, event: ReactionChanged
) -> EventOutcome:
    if get_message(conn, event.message_id) is None:
        return EventOutcome.REACTION_SKIPPED
    set_reaction_count(conn, event.message_id, event.emoji, event.count)
    return EventOutcome.REACTION_UPDATED


async def _handle_thread_created(
    conn: sqlite3.Connection, event: ThreadCreated, channel_ids: Sequence[int]
) -> EventOutcome:
    channel = event.channel
    if not is_channel_allowed(channel.id, channel.parent_id, channel_ids):
        return EventOutcome.CHANNEL_IGNORED
    upsert_channel(conn, normalize_channel(channel))
    return EventOutcome.THREAD_CREATED


async def handle_event(
    conn: sqlite3.Connection,
    event: SourceEvent,
    clock: Clock,
    include_bots: bool,
    channel_ids: Sequence[int] = (),
    source: DiscordSource | None = None,
) -> EventOutcome:
    match event:
        case MessageCreated():
            return await _handle_message_created(
                conn, event, clock, include_bots, channel_ids, source
            )
        case MessageEdited():
            return await _handle_message_edited(
                conn, event, clock, include_bots, channel_ids, source
            )
        case MessageDeleted():
            return await _handle_message_deleted(conn, event, clock, channel_ids)
        case ReactionChanged():
            return await _handle_reaction_changed(conn, event)
        case ThreadCreated():
            return await _handle_thread_created(conn, event, channel_ids)
        case _:
            raise TypeError(f"unhandled event type: {type(event)!r}")


async def consume(
    conn: sqlite3.Connection,
    source: DiscordSource,
    clock: Clock,
    include_bots: bool,
    channel_ids: Sequence[int] = (),
) -> ConsumeReport:
    report = ConsumeReport()
    async for event in source.events():
        try:
            outcome = await handle_event(conn, event, clock, include_bots, channel_ids, source)
        except Exception:
            logger.exception("event handling failed: %r", event)
            report.failed += 1
        else:
            if outcome is EventOutcome.CHANNEL_IGNORED:
                report.ignored += 1
            else:
                report.processed += 1
    return report
