from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st

from infovore.chunk.rules import (
    DEFAULT_MAX_MESSAGES,
    DEFAULT_QUIET_GAP,
    Group,
    drop_ungroupable,
    group_by_quiet_gap,
    group_by_reply_chain,
    group_by_thread,
    group_messages,
    is_closed,
    split_oversized,
)
from infovore.rows import GroupingRule, MessageRow

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def make_message(id: int, **overrides: object) -> MessageRow:
    fields: dict[str, object] = {
        "id": id,
        "channel_id": 10,
        "guild_id": 100,
        "author_id": 5,
        "author_name_at_time": "alice",
        "author_is_bot": False,
        "created_at": BASE,
        "edited_at": None,
        "content": "hello",
        "reply_to_id": None,
        "thread_id": None,
        "deleted_at": None,
        "ingested_at": BASE,
        "raw_json": "{}",
    }
    fields.update(overrides)
    return MessageRow(**fields)  # type: ignore[arg-type]


def at(minutes: int) -> datetime:
    return BASE + timedelta(minutes=minutes)


def ids(messages: Sequence[MessageRow]) -> list[int]:
    return [m.id for m in messages]


def test_drop_ungroupable_drops_deleted_messages() -> None:
    kept = make_message(1)
    deleted = make_message(2, deleted_at=at(1))
    assert drop_ungroupable([kept, deleted], include_bots=True) == [kept]


def test_drop_ungroupable_drops_bots_by_default() -> None:
    human = make_message(1)
    bot = make_message(2, author_is_bot=True)
    assert drop_ungroupable([human, bot], include_bots=False) == [human]


def test_drop_ungroupable_keeps_bots_when_included() -> None:
    human = make_message(1)
    bot = make_message(2, author_is_bot=True)
    result = drop_ungroupable([human, bot], include_bots=True)
    assert ids(result) == [1, 2]


def test_drop_ungroupable_deleted_bot_is_still_dropped() -> None:
    deleted_bot = make_message(1, author_is_bot=True, deleted_at=at(1))
    assert drop_ungroupable([deleted_bot], include_bots=True) == []


def test_group_by_thread_groups_shared_thread_id() -> None:
    a = make_message(1, thread_id=77, created_at=at(5))
    b = make_message(2, thread_id=77, created_at=at(1))
    c = make_message(3, thread_id=None, created_at=at(3))
    groups, remaining = group_by_thread([a, b, c])
    assert len(groups) == 1
    assert groups[0].rule == GroupingRule.THREAD
    assert ids(groups[0].messages) == [2, 1]
    assert groups[0].context == ()
    assert remaining == [c]


def test_group_by_thread_separates_distinct_threads() -> None:
    a = make_message(1, thread_id=77)
    b = make_message(2, thread_id=88)
    groups, remaining = group_by_thread([a, b])
    assert {g.messages[0].id for g in groups} == {1, 2}
    assert remaining == []


def test_group_by_thread_with_no_threads_passes_everything_through() -> None:
    a = make_message(1)
    b = make_message(2)
    groups, remaining = group_by_thread([a, b])
    assert groups == []
    assert remaining == [a, b]


def test_reply_chain_connects_transitive_replies() -> None:
    root = make_message(1, created_at=at(0))
    reply = make_message(2, created_at=at(1), reply_to_id=1)
    reply_to_reply = make_message(3, created_at=at(2), reply_to_id=2)
    chains, leftover = group_by_reply_chain([root, reply, reply_to_reply])
    assert len(chains) == 1
    assert chains[0].rule == GroupingRule.REPLY_CHAIN
    assert ids(chains[0].messages) == [1, 2, 3]
    assert leftover == []


def test_reply_chain_single_message_is_not_a_chain() -> None:
    lone = make_message(1)
    chains, leftover = group_by_reply_chain([lone])
    assert chains == []
    assert leftover == [lone]


def test_reply_chain_parent_outside_input_still_groups_children_together() -> None:
    reply_a = make_message(2, created_at=at(1), reply_to_id=999)
    reply_b = make_message(3, created_at=at(2), reply_to_id=999)
    chains, leftover = group_by_reply_chain([reply_a, reply_b])
    assert chains == []
    assert {m.id for m in leftover} == {2, 3}


def test_reply_chain_missing_parent_single_child_falls_through() -> None:
    orphan = make_message(2, reply_to_id=999)
    chains, leftover = group_by_reply_chain([orphan])
    assert chains == []
    assert leftover == [orphan]


def test_quiet_gap_splits_on_large_gap() -> None:
    a = make_message(1, created_at=at(0))
    b = make_message(2, created_at=at(10))
    c = make_message(3, created_at=at(100))
    groups = group_by_quiet_gap([a, b, c], timedelta(minutes=30))
    assert len(groups) == 2
    assert ids(groups[0].messages) == [1, 2]
    assert ids(groups[1].messages) == [3]
    assert all(g.rule == GroupingRule.QUIET_GAP for g in groups)
    assert all(g.context == () for g in groups)


def test_quiet_gap_keeps_close_messages_together() -> None:
    a = make_message(1, created_at=at(0))
    b = make_message(2, created_at=at(29))
    groups = group_by_quiet_gap([a, b], timedelta(minutes=30))
    assert len(groups) == 1
    assert ids(groups[0].messages) == [1, 2]


def test_quiet_gap_exact_boundary_is_not_a_split() -> None:
    a = make_message(1, created_at=at(0))
    b = make_message(2, created_at=at(30))
    groups = group_by_quiet_gap([a, b], timedelta(minutes=30))
    assert len(groups) == 1


def test_quiet_gap_empty_input_yields_no_groups() -> None:
    assert group_by_quiet_gap([], timedelta(minutes=30)) == []


def test_default_quiet_gap_is_thirty_minutes() -> None:
    assert DEFAULT_QUIET_GAP == timedelta(minutes=30)


def test_thread_precedence_wins_over_reply_to_message_outside_thread() -> None:
    outside_parent = make_message(1, created_at=at(0))
    in_thread_reply = make_message(2, created_at=at(1), thread_id=77, reply_to_id=1)
    in_thread_other = make_message(3, created_at=at(2), thread_id=77)
    groups = group_messages([outside_parent, in_thread_reply, in_thread_other])
    thread_groups = [g for g in groups if g.rule == GroupingRule.THREAD]
    assert len(thread_groups) == 1
    assert ids(thread_groups[0].messages) == [2, 3]
    other_groups = [g for g in groups if g.rule != GroupingRule.THREAD]
    assert len(other_groups) == 1
    assert other_groups[0].rule == GroupingRule.QUIET_GAP
    assert ids(other_groups[0].messages) == [1]


def test_group_messages_orders_groups_by_first_message() -> None:
    early = make_message(1, created_at=at(0))
    late = make_message(2, created_at=at(1000))
    groups = group_messages([late, early], quiet_gap=timedelta(minutes=1))
    assert [g.messages[0].id for g in groups] == [1, 2]


def test_group_messages_drops_deleted_and_bots() -> None:
    kept = make_message(1)
    deleted = make_message(2, deleted_at=at(1))
    bot = make_message(3, author_is_bot=True)
    groups = group_messages([kept, deleted, bot])
    all_ids = [m.id for g in groups for m in g.messages]
    assert all_ids == [1]


def test_group_messages_include_bots_flag() -> None:
    human = make_message(1, created_at=at(0))
    bot = make_message(2, author_is_bot=True, created_at=at(1))
    groups = group_messages([human, bot], include_bots=True, quiet_gap=timedelta(minutes=30))
    all_ids = [m.id for g in groups for m in g.messages]
    assert all_ids == [1, 2]


def test_split_oversized_noop_under_cap() -> None:
    messages = tuple(make_message(i, created_at=at(i)) for i in range(3))
    group = Group(GroupingRule.QUIET_GAP, messages, ())
    result = split_oversized(group, max_messages=5, overlap=3)
    assert result == [group]


def test_split_oversized_splits_at_largest_internal_gap() -> None:
    messages = (
        make_message(1, created_at=at(0)),
        make_message(2, created_at=at(1)),
        make_message(3, created_at=at(2)),
        make_message(4, created_at=at(100)),
        make_message(5, created_at=at(101)),
    )
    group = Group(GroupingRule.QUIET_GAP, messages, ())
    result = split_oversized(group, max_messages=4, overlap=3)
    assert len(result) == 2
    assert ids(result[0].messages) == [1, 2, 3]
    assert result[0].context == ()
    assert ids(result[1].messages) == [4, 5]
    assert ids(result[1].context) == [1, 2, 3]


def test_split_oversized_context_is_capped_at_overlap() -> None:
    messages = tuple(make_message(i, created_at=at(i)) for i in range(6))
    group = Group(GroupingRule.QUIET_GAP, messages, ())
    result = split_oversized(group, max_messages=3, overlap=2)
    assert len(result) == 2
    assert ids(result[0].messages) == [0, 1, 2]
    assert ids(result[1].messages) == [3, 4, 5]
    assert ids(result[1].context) == [1, 2]


def test_split_oversized_context_smaller_than_overlap_when_first_part_short() -> None:
    messages = tuple(make_message(i, created_at=at(i)) for i in range(4))
    group = Group(GroupingRule.QUIET_GAP, messages, ())
    result = split_oversized(group, max_messages=1, overlap=3)
    assert len(result) == 4
    assert ids(result[1].context) == [0]
    assert ids(result[2].context) == [1]


def test_split_oversized_many_parts_all_within_cap() -> None:
    messages = tuple(make_message(i, created_at=at(i)) for i in range(11))
    group = Group(GroupingRule.QUIET_GAP, messages, ())
    result = split_oversized(group, max_messages=4, overlap=3)
    assert all(len(part.messages) <= 4 for part in result)
    assert sum(len(part.messages) for part in result) == 11


def test_group_messages_applies_size_cap() -> None:
    messages = [make_message(i, created_at=at(i)) for i in range(12)]
    groups = group_messages(messages, quiet_gap=timedelta(minutes=30), max_messages=5)
    assert all(len(g.messages) <= 5 for g in groups)
    assert sum(len(g.messages) for g in groups) == 12


def test_default_max_messages_is_fifty() -> None:
    assert DEFAULT_MAX_MESSAGES == 50


def test_is_closed_true_when_newest_message_older_than_gap() -> None:
    messages = (make_message(1, created_at=at(0)),)
    group = Group(GroupingRule.QUIET_GAP, messages, ())
    now = at(31)
    assert is_closed(group, now, timedelta(minutes=30)) is True


def test_is_closed_false_when_newest_message_within_gap() -> None:
    messages = (make_message(1, created_at=at(0)),)
    group = Group(GroupingRule.QUIET_GAP, messages, ())
    now = at(10)
    assert is_closed(group, now, timedelta(minutes=30)) is False


def test_is_closed_exact_boundary_is_not_closed() -> None:
    messages = (make_message(1, created_at=at(0)),)
    group = Group(GroupingRule.QUIET_GAP, messages, ())
    now = at(30)
    assert is_closed(group, now, timedelta(minutes=30)) is False


def test_is_closed_uses_newest_message_in_group() -> None:
    messages = (make_message(1, created_at=at(0)), make_message(2, created_at=at(20)))
    group = Group(GroupingRule.QUIET_GAP, messages, ())
    assert is_closed(group, at(25), timedelta(minutes=30)) is False
    assert is_closed(group, at(51), timedelta(minutes=30)) is True


@st.composite
def _message_lists(draw: st.DrawFn) -> list[MessageRow]:
    id_list = draw(st.lists(st.integers(min_value=1, max_value=30), unique=True, min_size=1, max_size=20))
    thread_pool = [None, 1001, 1002, 1003]
    messages = []
    for index, message_id in enumerate(id_list):
        offset = draw(st.integers(min_value=0, max_value=400))
        thread_id = draw(st.sampled_from(thread_pool))
        reply_candidates = [None, 99999, *id_list[:index]]
        reply_to_id = draw(st.sampled_from(reply_candidates))
        deleted = draw(st.booleans())
        is_bot = draw(st.booleans())
        messages.append(
            make_message(
                message_id,
                created_at=at(offset),
                thread_id=thread_id,
                reply_to_id=reply_to_id,
                deleted_at=at(offset) if deleted else None,
                author_is_bot=is_bot,
            )
        )
    return messages


@given(_message_lists(), st.integers(min_value=1, max_value=8))
def test_every_kept_message_lands_in_exactly_one_group(
    messages: list[MessageRow], max_messages: int
) -> None:
    kept = drop_ungroupable(messages, include_bots=False)
    groups = group_messages(
        messages, quiet_gap=timedelta(minutes=30), max_messages=max_messages, include_bots=False
    )
    flat = [m.id for g in groups for m in g.messages]
    assert sorted(flat) == sorted(m.id for m in kept)
    assert len(flat) == len(set(flat))


@given(_message_lists(), st.integers(min_value=1, max_value=8))
def test_messages_within_each_group_are_ordered(
    messages: list[MessageRow], max_messages: int
) -> None:
    groups = group_messages(messages, quiet_gap=timedelta(minutes=30), max_messages=max_messages)
    for group in groups:
        keys = [(m.created_at, m.id) for m in group.messages]
        assert keys == sorted(keys)


@given(_message_lists(), st.integers(min_value=1, max_value=8))
def test_every_group_respects_the_size_cap(
    messages: list[MessageRow], max_messages: int
) -> None:
    groups = group_messages(messages, quiet_gap=timedelta(minutes=30), max_messages=max_messages)
    assert all(len(g.messages) <= max_messages for g in groups)


@given(_message_lists(), st.integers(min_value=1, max_value=8))
def test_context_messages_never_overlap_their_own_group(
    messages: list[MessageRow], max_messages: int
) -> None:
    groups = group_messages(messages, quiet_gap=timedelta(minutes=30), max_messages=max_messages)
    for group in groups:
        own_ids = {m.id for m in group.messages}
        context_ids = {m.id for m in group.context}
        assert own_ids.isdisjoint(context_ids)
