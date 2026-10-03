import hashlib
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta

from infovore.chunk.gaps import fold_gap, recorded_gap
from infovore.chunk.recipe import ChunkRecipe
from infovore.chunk.rules import (
    DEFAULT_MAX_MESSAGES,
    DEFAULT_QUIET_GAP,
    Group,
    group_messages,
    is_closed,
)
from infovore.db.chunk_recipes import register_recipe
from infovore.db.exchanges import DuplicateExchangeError, exchange_for_message, insert_exchange
from infovore.db.raw import (
    latest_exchange_for_thread,
    ungrouped_channel_ids,
    ungrouped_messages_for_channel,
)
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule
from infovore.timing import Clock


@dataclass(frozen=True)
class GroupingReport:
    exchanges_created: int
    messages_grouped: int
    groups_deferred: int


@dataclass(frozen=True)
class GroupingStarted:
    channels: int


@dataclass(frozen=True)
class ChannelGrouped:
    channel_id: int
    exchanges_created: int
    groups_deferred: int


GroupingEvent = GroupingStarted | ChannelGrouped
GroupingProgress = Callable[[GroupingEvent], None]


def _ignore_grouping_progress(event: GroupingEvent) -> None:
    return None


def content_hash(message_ids: Sequence[int]) -> str:
    joined = ",".join(str(message_id) for message_id in message_ids)
    return hashlib.sha256(joined.encode()).hexdigest()


def _resolve_parent(conn: sqlite3.Connection, group: Group) -> int | None:
    if group.context:
        parent = exchange_for_message(conn, group.context[0].id)
        if parent is not None:
            return parent
    first_message = group.messages[0]
    if first_message.reply_to_id is not None:
        parent = exchange_for_message(conn, first_message.reply_to_id)
        if parent is not None:
            return parent
    if group.rule == GroupingRule.THREAD and first_message.thread_id is not None:
        return latest_exchange_for_thread(conn, first_message.thread_id)
    return None


def _persist_group(
    conn: sqlite3.Connection, channel_id: int, group: Group, version: int | None
) -> int:
    message_ids = [message.id for message in group.messages]
    exchange = ExchangeRow(
        id=None,
        channel_id=channel_id,
        thread_id=group.messages[0].thread_id,
        first_message_id=message_ids[0],
        last_message_id=message_ids[-1],
        started_at=group.messages[0].created_at,
        ended_at=group.messages[-1].created_at,
        message_count=len(message_ids),
        grouping_rule=group.rule,
        content_hash=content_hash(message_ids),
        parent_exchange_id=_resolve_parent(conn, group),
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
        chunk_recipe=version,
    )
    insert_exchange(conn, exchange, message_ids)
    return len(message_ids)


def group_pending(
    conn: sqlite3.Connection,
    clock: Clock,
    quiet_gap: timedelta = DEFAULT_QUIET_GAP,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    include_bots: bool = False,
    progress: GroupingProgress = _ignore_grouping_progress,
    recipe: ChunkRecipe | None = None,
) -> GroupingReport:
    exchanges_created = 0
    messages_grouped = 0
    groups_deferred = 0
    now = clock.now()
    version = register_recipe(conn, recipe, now) if recipe is not None else None
    channel_ids = ungrouped_channel_ids(conn)
    progress(GroupingStarted(channels=len(channel_ids)))
    for channel_id in channel_ids:
        channel_exchanges = 0
        channel_deferred = 0
        messages = ungrouped_messages_for_channel(conn, channel_id)
        gap, folding, fold_size = quiet_gap, None, 1
        if recipe is not None and version is not None:
            gap = recorded_gap(conn, version, recipe, channel_id, include_bots)
            folding, fold_size = fold_gap(recipe, gap), recipe.fold_size
        groups = group_messages(messages, gap, max_messages, include_bots, folding, fold_size)
        for group in groups:
            if not is_closed(group, now, gap):
                groups_deferred += 1
                channel_deferred += 1
                continue
            try:
                grouped_count = _persist_group(conn, channel_id, group, version)
            except DuplicateExchangeError:
                continue
            exchanges_created += 1
            channel_exchanges += 1
            messages_grouped += grouped_count
        progress(
            ChannelGrouped(
                channel_id=channel_id,
                exchanges_created=channel_exchanges,
                groups_deferred=channel_deferred,
            )
        )
    return GroupingReport(exchanges_created, messages_grouped, groups_deferred)
