import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from infovore.db.connection import transaction
from infovore.db.raw import (
    UpsertOutcome,
    _set_backfill_checkpoint,
    _set_reaction_count,
    _upsert_attachment,
    _upsert_message,
    get_backfill_checkpoint,
    upsert_channel,
)
from infovore.ingest.allowlist import is_channel_allowed
from infovore.ingest.normalize import normalize_channel, normalize_message
from infovore.privacy.optout import opted_out_user_ids, redact_normalized
from infovore.source.protocol import (
    DiscordSource,
    SourceChannel,
    SourceForbiddenError,
    SourceMessage,
    SourceRateLimitedError,
    SourceUnavailableError,
)
from infovore.timing import Clock, Sleeper


@dataclass
class ChannelReport:
    channel_id: int
    pages: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0


@dataclass
class FailedChannel:
    channel_id: int
    reason: str


@dataclass
class BackfillReport:
    channels: dict[int, ChannelReport] = field(default_factory=dict)
    failed: list[FailedChannel] = field(default_factory=list)


def _select_channels(
    channels: Sequence[SourceChannel], channel_ids: Sequence[int]
) -> list[SourceChannel]:
    selected = [
        channel
        for channel in channels
        if is_channel_allowed(channel.id, channel.parent_id, channel_ids)
    ]
    selected.sort(key=lambda channel: channel.id)
    return selected


def _commit_page(
    conn: sqlite3.Connection,
    channel_id: int,
    page: Sequence[SourceMessage],
    ingested_at: datetime,
    include_bots: bool,
    channel_report: ChannelReport,
) -> None:
    with transaction(conn):
        opted_out = opted_out_user_ids(conn)
        for message in page:
            normalized = normalize_message(message, ingested_at, include_bots)
            if normalized is None:
                channel_report.skipped += 1
                continue
            normalized = redact_normalized(normalized, opted_out)
            outcome = _upsert_message(conn, normalized.message)
            if outcome is UpsertOutcome.INSERTED:
                channel_report.inserted += 1
            elif outcome is UpsertOutcome.UPDATED:
                channel_report.updated += 1
            else:
                channel_report.unchanged += 1
            for attachment in normalized.attachments:
                _upsert_attachment(conn, attachment)
            for reaction in normalized.reactions:
                _set_reaction_count(conn, reaction.message_id, reaction.emoji, reaction.count)
        _set_backfill_checkpoint(conn, channel_id, page[-1].id)
        channel_report.pages += 1


async def _walk_channel(
    conn: sqlite3.Connection,
    source: DiscordSource,
    channel_id: int,
    clock: Clock,
    sleeper: Sleeper,
    include_bots: bool,
    page_size: int,
    max_attempts: int,
    channel_report: ChannelReport,
) -> str | None:
    attempts = 0
    while True:
        checkpoint = get_backfill_checkpoint(conn, channel_id)
        try:
            async for page in source.history(channel_id, checkpoint, page_size):
                _commit_page(conn, channel_id, page, clock.now(), include_bots, channel_report)
                attempts = 0
            return None
        except SourceForbiddenError as error:
            return f"forbidden: {error}"
        except SourceRateLimitedError as error:
            attempts += 1
            if attempts >= max_attempts:
                return f"rate limited after {attempts} attempts: {error}"
            await sleeper.sleep(error.retry_after)
        except SourceUnavailableError as error:
            attempts += 1
            if attempts >= max_attempts:
                return f"unavailable after {attempts} attempts: {error}"
            await sleeper.sleep(2.0 ** (attempts - 1))


async def backfill(
    conn: sqlite3.Connection,
    source: DiscordSource,
    guild_id: int,
    channel_ids: Sequence[int],
    clock: Clock,
    sleeper: Sleeper,
    include_bots: bool,
    page_size: int = 100,
    max_attempts: int = 5,
) -> BackfillReport:
    report = BackfillReport()
    channels = await source.list_channels(guild_id)
    for channel in _select_channels(channels, channel_ids):
        upsert_channel(conn, normalize_channel(channel))
        channel_report = ChannelReport(channel_id=channel.id)
        report.channels[channel.id] = channel_report
        failure = await _walk_channel(
            conn,
            source,
            channel.id,
            clock,
            sleeper,
            include_bots,
            page_size,
            max_attempts,
            channel_report,
        )
        if failure is not None:
            report.failed.append(FailedChannel(channel_id=channel.id, reason=failure))
    return report
