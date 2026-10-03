import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise

from infovore.rows import GroupingRule, MessageRow

DEFAULT_QUIET_GAP = timedelta(minutes=30)
DEFAULT_MAX_MESSAGES = 50
DEFAULT_OVERLAP = 3
ZERO = timedelta(0)


@dataclass(frozen=True)
class Group:
    rule: GroupingRule
    messages: tuple[MessageRow, ...]
    context: tuple[MessageRow, ...] = ()


def _sort_key(message: MessageRow) -> tuple[datetime, int]:
    return (message.created_at, message.id)


def _ordered(messages: Sequence[MessageRow]) -> tuple[MessageRow, ...]:
    return tuple(sorted(messages, key=_sort_key))


def drop_ungroupable(messages: Sequence[MessageRow], include_bots: bool) -> list[MessageRow]:
    return [
        message
        for message in messages
        if message.deleted_at is None and (include_bots or not message.author_is_bot)
    ]


def group_by_thread(
    messages: Sequence[MessageRow],
) -> tuple[list[Group], list[MessageRow]]:
    threads: dict[int, list[MessageRow]] = {}
    remaining: list[MessageRow] = []
    for message in messages:
        if message.thread_id is not None:
            threads.setdefault(message.thread_id, []).append(message)
        else:
            remaining.append(message)
    groups = [
        Group(GroupingRule.THREAD, _ordered(thread_messages), ())
        for thread_messages in threads.values()
    ]
    return groups, remaining


def _reply_components(messages: Sequence[MessageRow]) -> list[list[MessageRow]]:
    by_id = {message.id: message for message in messages}
    parent = {message.id: message.id for message in messages}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    for message in messages:
        if message.reply_to_id is not None and message.reply_to_id in by_id:
            union(message.id, message.reply_to_id)

    components: dict[int, list[MessageRow]] = {}
    for message in messages:
        components.setdefault(find(message.id), []).append(message)
    return list(components.values())


def group_by_reply_chain(
    messages: Sequence[MessageRow],
) -> tuple[list[Group], list[MessageRow]]:
    chains: list[Group] = []
    leftover: list[MessageRow] = []
    for component in _reply_components(messages):
        if len(component) > 1:
            chains.append(Group(GroupingRule.REPLY_CHAIN, _ordered(component), ()))
        else:
            leftover.extend(component)
    return chains, leftover


def group_by_quiet_gap(
    messages: Sequence[MessageRow], quiet_gap: timedelta = DEFAULT_QUIET_GAP
) -> list[Group]:
    ordered = _ordered(messages)
    groups: list[Group] = []
    current: list[MessageRow] = []
    for message in ordered:
        if current and message.created_at - current[-1].created_at > quiet_gap:
            groups.append(Group(GroupingRule.QUIET_GAP, tuple(current), ()))
            current = []
        current.append(message)
    if current:
        groups.append(Group(GroupingRule.QUIET_GAP, tuple(current), ()))
    return groups


def channel_gap(
    messages: Sequence[MessageRow],
    percentile: float,
    floor: timedelta,
    ceiling: timedelta,
    include_bots: bool = False,
) -> timedelta:
    times = sorted(m.created_at for m in drop_ungroupable(messages, include_bots))
    gaps = sorted(gap for earlier, later in pairwise(times) if (gap := later - earlier) <= ceiling)
    if not gaps:
        return floor
    rank = max(math.ceil(percentile / 100 * len(gaps)) - 1, 0)
    return max(gaps[rank], floor)


def _related(group: Group, target: Group) -> bool:
    mine = {m.id for m in group.messages}
    theirs = {m.id for m in target.messages}
    return any(
        (m.reply_to_id in theirs) or any(o.reply_to_id == m.id for o in target.messages)
        for m in group.messages
    ) or any(o.reply_to_id in mine for o in target.messages)


def fold_small(
    groups: Sequence[Group], gap: timedelta, max_messages: int, small: int = 1
) -> list[Group]:
    work = sorted(groups, key=lambda group: _sort_key(group.messages[0]))
    result: list[Group] = []
    for index, group in enumerate(work):
        if len(group.messages) > small or group.rule == GroupingRule.THREAD:
            result.append(group)
            continue
        first = group.messages[0].created_at
        last = max(m.created_at for m in group.messages)
        options: list[tuple[bool, timedelta, int, Group]] = []
        if result:
            end = max(m.created_at for m in result[-1].messages)
            options.append((_related(group, result[-1]), first - end, 0, result[-1]))
        if index + 1 < len(work):
            nxt = work[index + 1]
            options.append((_related(group, nxt), nxt.messages[0].created_at - last, 1, nxt))
        eligible = [
            (not related, max(distance, ZERO), side, target)
            for related, distance, side, target in options
            if distance <= gap
            and small < len(target.messages)
            and len(target.messages) + len(group.messages) <= max_messages
            and (related or target.rule != GroupingRule.THREAD)
        ]
        if not eligible:
            result.append(group)
            continue
        *_, side, target = min(eligible, key=lambda option: option[:3])
        merged = Group(target.rule, _ordered((*target.messages, *group.messages)), target.context)
        if side == 0:
            result[-1] = merged
        else:
            work[index + 1] = merged
    return result


def split_oversized(group: Group, max_messages: int, overlap: int = DEFAULT_OVERLAP) -> list[Group]:
    if len(group.messages) <= max_messages:
        return [group]

    parts: list[tuple[MessageRow, ...]] = []
    remaining = list(group.messages)
    while len(remaining) > max_messages:
        window = remaining[: max_messages + 1]
        cut = max(
            range(1, len(window)),
            key=lambda i: (window[i].created_at - window[i - 1].created_at, i),
        )
        parts.append(tuple(remaining[:cut]))
        remaining = remaining[cut:]
    parts.append(tuple(remaining))

    result: list[Group] = []
    for index, part in enumerate(parts):
        context = parts[index - 1][-overlap:] if index > 0 else ()
        result.append(Group(group.rule, part, context))
    return result


def group_messages(
    messages: Sequence[MessageRow],
    quiet_gap: timedelta = DEFAULT_QUIET_GAP,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    include_bots: bool = False,
    fold_gap: timedelta | None = None,
    fold_size: int = 1,
) -> list[Group]:
    kept = drop_ungroupable(messages, include_bots)
    thread_groups, after_threads = group_by_thread(kept)
    chain_groups, after_chains = group_by_reply_chain(after_threads)
    gap_groups = group_by_quiet_gap(after_chains, quiet_gap)

    groups = [*thread_groups, *chain_groups, *gap_groups]
    groups.sort(key=lambda group: _sort_key(group.messages[0]))
    if fold_gap is not None:
        groups = fold_small(groups, fold_gap, max_messages, fold_size)

    split: list[Group] = []
    for group in groups:
        split.extend(split_oversized(group, max_messages, DEFAULT_OVERLAP))
    return split


def is_closed(group: Group, now: datetime, quiet_gap: timedelta = DEFAULT_QUIET_GAP) -> bool:
    newest = max(message.created_at for message in group.messages)
    return now - newest > quiet_gap
