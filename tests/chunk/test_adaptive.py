from datetime import timedelta

from infovore.chunk.rules import Group, channel_gap, fold_small, group_messages
from infovore.rows import GroupingRule
from tests.chunk.test_rules import at, ids, make_message

GAP = timedelta(minutes=30)
FLOOR = timedelta(minutes=30)
CEILING = timedelta(hours=6)


def test_channel_gap_is_a_percentile_of_the_gaps_inside_conversations() -> None:
    messages = [make_message(i, created_at=at(i * 40)) for i in range(11)]
    assert channel_gap(messages, 90, FLOOR, CEILING) == timedelta(minutes=40)


def test_channel_gap_ignores_gaps_beyond_the_ceiling() -> None:
    messages = [
        make_message(1, created_at=at(0)),
        make_message(2, created_at=at(100)),
        make_message(3, created_at=at(100 + 60 * 24)),
    ]
    assert channel_gap(messages, 90, FLOOR, CEILING) == timedelta(minutes=100)


def test_channel_gap_is_clamped_to_the_floor() -> None:
    quick = [make_message(i, created_at=at(i)) for i in range(5)]
    assert channel_gap(quick, 90, FLOOR, CEILING) == FLOOR
    slow = [make_message(i, created_at=at(i * 340)) for i in range(5)]
    assert channel_gap(slow, 90, FLOOR, CEILING) == timedelta(minutes=340)


def test_channel_gap_with_no_evidence_is_the_floor() -> None:
    assert channel_gap([], 90, FLOOR, CEILING) == FLOOR
    one = [make_message(1)]
    assert channel_gap(one, 90, FLOOR, CEILING) == FLOOR
    far = [make_message(1, created_at=at(0)), make_message(2, created_at=at(60 * 24))]
    assert channel_gap(far, 90, FLOOR, CEILING) == FLOOR


def test_channel_gap_skips_deleted_messages_and_bots() -> None:
    messages = [
        make_message(1, created_at=at(0)),
        make_message(2, created_at=at(500), deleted_at=at(501)),
        make_message(3, created_at=at(20), author_is_bot=True),
        make_message(4, created_at=at(100)),
    ]
    assert channel_gap(messages, 90, FLOOR, CEILING) == timedelta(minutes=100)
    assert channel_gap(messages, 90, FLOOR, CEILING, include_bots=True) == timedelta(minutes=80)


def lone(id: int, minute: int, **kw: object) -> Group:
    return Group(GroupingRule.QUIET_GAP, (make_message(id, created_at=at(minute), **kw),))


def pair(first_id: int, minute: int) -> Group:
    return Group(
        GroupingRule.QUIET_GAP,
        (
            make_message(first_id, created_at=at(minute)),
            make_message(first_id + 1, created_at=at(minute + 1)),
        ),
    )


def test_a_lone_message_joins_the_nearer_neighbour_within_the_gap() -> None:
    folded = fold_small([pair(1, 0), lone(3, 30), pair(4, 55)], GAP, 50)
    assert [ids(g.messages) for g in folded] == [[1, 2], [3, 4, 5]]


def test_a_lone_message_beyond_the_gap_of_both_sides_stays_alone() -> None:
    folded = fold_small([pair(1, 0), lone(3, 100), pair(4, 200)], GAP, 50)
    assert [ids(g.messages) for g in folded] == [[1, 2], [3], [4, 5]]


def test_a_lone_message_prefers_the_neighbour_it_replies_to() -> None:
    reply = lone(3, 25, reply_to_id=1)
    folded = fold_small([pair(1, 0), reply, pair(4, 45)], GAP, 50)
    assert [ids(g.messages) for g in folded] == [[1, 2, 3], [4, 5]]


def test_a_lone_message_prefers_the_neighbour_that_replies_to_it() -> None:
    follow = Group(
        GroupingRule.QUIET_GAP,
        (
            make_message(4, created_at=at(45), reply_to_id=3),
            make_message(5, created_at=at(46)),
        ),
    )
    folded = fold_small([pair(1, 0), lone(3, 20), follow], GAP, 50)
    assert [ids(g.messages) for g in folded] == [[1, 2], [3, 4, 5]]


def test_folding_never_exceeds_the_message_cap() -> None:
    full = Group(
        GroupingRule.QUIET_GAP,
        tuple(make_message(i, created_at=at(i)) for i in range(1, 4)),
    )
    folded = fold_small([full, lone(9, 10)], GAP, 3)
    assert [ids(g.messages) for g in folded] == [[1, 2, 3], [9]]


def test_a_lone_message_does_not_join_another_lone_message() -> None:
    folded = fold_small([lone(1, 0), lone(2, 20)], GAP, 50)
    assert [ids(g.messages) for g in folded] == [[1], [2]]


def test_a_lone_message_joins_a_thread_only_by_reply() -> None:
    thread = Group(
        GroupingRule.THREAD,
        (
            make_message(1, created_at=at(0), thread_id=7),
            make_message(2, created_at=at(1), thread_id=7),
        ),
    )
    unrelated = fold_small([thread, lone(3, 10)], GAP, 50)
    assert [ids(g.messages) for g in unrelated] == [[1, 2], [3]]
    replying = fold_small([thread, lone(3, 10, reply_to_id=2)], GAP, 50)
    assert [ids(g.messages) for g in replying] == [[1, 2, 3]]
    assert replying[0].rule == GroupingRule.THREAD


def test_a_lone_message_inside_a_thread_span_counts_as_adjacent() -> None:
    thread = Group(
        GroupingRule.THREAD,
        (
            make_message(1, created_at=at(0), thread_id=7),
            make_message(2, created_at=at(100), thread_id=7),
        ),
    )
    folded = fold_small([thread, lone(3, 50, reply_to_id=1)], GAP, 50)
    assert [ids(g.messages) for g in folded] == [[1, 3, 2]]


def test_a_lone_message_with_no_neighbours_stays() -> None:
    assert [ids(g.messages) for g in fold_small([lone(1, 0)], GAP, 50)] == [[1]]
    assert fold_small([], GAP, 50) == []


def test_group_messages_folds_with_its_own_wider_gap() -> None:
    messages = [
        make_message(1, created_at=at(0)),
        make_message(2, created_at=at(1)),
        make_message(3, created_at=at(50)),
    ]
    plain = group_messages(messages, GAP, 50)
    assert [ids(g.messages) for g in plain] == [[1, 2], [3]]
    folded = group_messages(messages, GAP, 50, fold_gap=timedelta(minutes=60))
    assert [ids(g.messages) for g in folded] == [[1, 2, 3]]


def trio(first_id: int, minute: int) -> Group:
    return Group(
        GroupingRule.QUIET_GAP,
        tuple(make_message(first_id + i, created_at=at(minute + i)) for i in range(3)),
    )


def test_a_small_group_folds_whole_into_a_larger_neighbour() -> None:
    chain = Group(
        GroupingRule.REPLY_CHAIN,
        (
            make_message(10, created_at=at(100)),
            make_message(11, created_at=at(101), reply_to_id=10),
        ),
    )
    folded = fold_small([trio(1, 0), chain], timedelta(minutes=120), 50, small=2)
    assert [ids(g.messages) for g in folded] == [[1, 2, 3, 10, 11]]
    assert fold_small([trio(1, 0), chain], timedelta(minutes=120), 50, small=1) != folded


def test_a_small_group_does_not_join_a_group_no_larger_than_itself() -> None:
    folded = fold_small([pair(1, 0), pair(3, 40)], timedelta(minutes=120), 50, small=2)
    assert [ids(g.messages) for g in folded] == [[1, 2], [3, 4]]


def test_a_small_group_that_would_overflow_the_cap_stays() -> None:
    folded = fold_small([trio(1, 0), pair(4, 10)], timedelta(minutes=120), 4, small=2)
    assert [ids(g.messages) for g in folded] == [[1, 2, 3], [4, 5]]


def test_a_small_group_prefers_the_neighbour_it_replies_to() -> None:
    reply = Group(
        GroupingRule.REPLY_CHAIN,
        (
            make_message(10, created_at=at(40)),
            make_message(11, created_at=at(41), reply_to_id=2),
        ),
    )
    folded = fold_small([trio(1, 0), reply, trio(20, 50)], timedelta(minutes=120), 50, small=2)
    assert [ids(g.messages) for g in folded] == [[1, 2, 3, 10, 11], [20, 21, 22]]


def test_group_messages_passes_the_fold_size() -> None:
    messages = [
        make_message(1, created_at=at(0)),
        make_message(2, created_at=at(1)),
        make_message(3, created_at=at(2)),
        make_message(4, created_at=at(100)),
        make_message(5, created_at=at(101)),
    ]
    wide = timedelta(minutes=120)
    assert len(group_messages(messages, GAP, 50, fold_gap=wide, fold_size=1)) == 2
    assert len(group_messages(messages, GAP, 50, fold_gap=wide, fold_size=2)) == 1
