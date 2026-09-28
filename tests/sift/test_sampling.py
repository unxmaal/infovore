import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.message_labels import set_message_label
from infovore.rows import MessageLabel, MessageLabelSource
from infovore.sift.sampling import (
    NoScoredMessagesError,
    SiftStrategy,
    eligible_message_pool,
    select_sift_sample,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC).isoformat()


def _channel(
    conn: sqlite3.Connection, channel_id: int, name: str, parent_id: int | None = None
) -> None:
    kind = "thread" if parent_id is not None else "text"
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind) VALUES (?, 1, ?, ?, ?)",
        (channel_id, parent_id, name, kind),
    )


def _message_with_exchange(
    conn: sqlite3.Connection,
    message_id: int,
    channel_id: int,
    author_id: int = 1,
    p_trash: float | None = None,
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json, p_trash)"
        " VALUES (?, ?, 1, ?, 'author', ?, 'hello', ?, '{}', ?)",
        (message_id, channel_id, author_id, NOW, NOW, p_trash),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, ?, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
        (message_id, channel_id, message_id, message_id, NOW, NOW, f"h{message_id}"),
    )
    conn.execute(
        "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
        (message_id, message_id),
    )


def seeded(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def test_eligible_message_pool_excludes_opted_out_authors(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1, author_id=1)
    _message_with_exchange(conn, 2, 1, author_id=2)
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (2, ?)", (NOW,))
    pool = eligible_message_pool(conn)
    assert {row.id for row in pool} == {1}


def test_eligible_message_pool_excludes_human_labeled_messages(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1)
    _message_with_exchange(conn, 2, 1)
    set_message_label(
        conn,
        1,
        MessageLabel.TRASH,
        MessageLabelSource.HUMAN,
        None,
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    pool = eligible_message_pool(conn)
    assert {row.id for row in pool} == {2}


def test_eligible_message_pool_excludes_messages_with_no_exchange(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1)
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json) VALUES (99, 1, 1, 1, 'a', ?, 'x', ?, '{}')",
        (NOW, NOW),
    )
    pool = eligible_message_pool(conn)
    assert {row.id for row in pool} == {1}


def test_select_random_returns_everything_when_n_exceeds_pool(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    for i in range(1, 4):
        _message_with_exchange(conn, i, 1)
    selected = select_sift_sample(conn, 100, seed=0, strategy=SiftStrategy.RANDOM)
    assert sorted(selected) == [1, 2, 3]


def test_select_random_stratifies_round_robin_across_channels(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    for i in range(1, 4):
        _message_with_exchange(conn, i, 1)
    for i in range(4, 7):
        _message_with_exchange(conn, i, 2)
    selected = select_sift_sample(conn, 4, seed=0, strategy=SiftStrategy.RANDOM)
    assert len(selected) == 4
    from_general = [m for m in selected if m < 4]
    from_food = [m for m in selected if m >= 4]
    assert len(from_general) == 2
    assert len(from_food) == 2


def test_select_random_is_deterministic_under_the_same_seed(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    for i in range(1, 10):
        _message_with_exchange(conn, i, 1)
    first = select_sift_sample(conn, 4, seed=7, strategy=SiftStrategy.RANDOM)
    second = select_sift_sample(conn, 4, seed=7, strategy=SiftStrategy.RANDOM)
    assert first == second


def test_select_uncertain_raises_when_nothing_is_scored(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1)
    with pytest.raises(NoScoredMessagesError):
        select_sift_sample(conn, 1, seed=0, strategy=SiftStrategy.UNCERTAIN)


def test_select_uncertain_prefers_scores_closest_to_half(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1, p_trash=0.5)
    _message_with_exchange(conn, 2, 1, p_trash=0.99)
    _message_with_exchange(conn, 3, 1, p_trash=0.01)
    selected = select_sift_sample(conn, 1, seed=0, strategy=SiftStrategy.UNCERTAIN)
    assert selected == [1]


def test_select_uncertain_ignores_unscored_messages(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1, p_trash=None)
    _message_with_exchange(conn, 2, 1, p_trash=0.4)
    selected = select_sift_sample(conn, 5, seed=0, strategy=SiftStrategy.UNCERTAIN)
    assert selected == [2]


def test_select_mixed_falls_back_to_random_when_nothing_is_scored(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    for i in range(1, 6):
        _message_with_exchange(conn, i, 1)
    selected = select_sift_sample(conn, 3, seed=0, strategy=SiftStrategy.MIXED)
    assert len(selected) == 3


def test_select_mixed_with_mix_one_behaves_like_uncertain(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1, p_trash=0.5)
    _message_with_exchange(conn, 2, 1, p_trash=0.99)
    selected = select_sift_sample(conn, 1, seed=0, strategy=SiftStrategy.MIXED, mix=1.0)
    assert selected == [1]


def test_select_mixed_consumes_the_whole_scored_pool_via_uncertain(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1, p_trash=0.5)
    _message_with_exchange(conn, 2, 1, p_trash=0.99)
    selected = select_sift_sample(conn, 2, seed=0, strategy=SiftStrategy.MIXED, mix=1.0)
    assert sorted(selected) == [1, 2]


def test_allocate_round_robin_skips_an_exhausted_channel(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    _message_with_exchange(conn, 1, 1)
    for i in range(2, 7):
        _message_with_exchange(conn, i, 2)
    selected = select_sift_sample(conn, 3, seed=0, strategy=SiftStrategy.RANDOM)
    assert len(selected) == 3
    assert 1 in selected


def test_select_mixed_with_mix_zero_behaves_like_random(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1, p_trash=0.5)
    _message_with_exchange(conn, 2, 1, p_trash=0.99)
    selected = select_sift_sample(conn, 2, seed=0, strategy=SiftStrategy.MIXED, mix=0.0)
    assert sorted(selected) == [1, 2]


# --- channel denylist / include filter (issue #138) --------------------------


def test_eligible_message_pool_excludes_denylisted_channel(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    _message_with_exchange(conn, 1, 1)
    _message_with_exchange(conn, 2, 2)
    pool = eligible_message_pool(conn, exclude_channels=frozenset({"food"}))
    assert {row.id for row in pool} == {1}


def test_eligible_message_pool_excludes_thread_whose_parent_is_denylisted(
    tmp_path: Path,
) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _channel(conn, 10, "food")
    _channel(conn, 11, "food-thread-1", parent_id=10)
    _message_with_exchange(conn, 1, 1)
    _message_with_exchange(conn, 2, 11)
    pool = eligible_message_pool(conn, exclude_channels=frozenset({"food"}))
    assert {row.id for row in pool} == {1}


def test_eligible_message_pool_include_channels_restricts_the_pool(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    _message_with_exchange(conn, 1, 1)
    _message_with_exchange(conn, 2, 2)
    pool = eligible_message_pool(conn, include_channels=frozenset({"general"}))
    assert {row.id for row in pool} == {1}


def test_eligible_message_pool_include_channels_covers_threads_of_the_named_channel(
    tmp_path: Path,
) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _channel(conn, 2, "general-thread-1", parent_id=1)
    _message_with_exchange(conn, 1, 1)
    _message_with_exchange(conn, 2, 2)
    pool = eligible_message_pool(conn, include_channels=frozenset({"general"}))
    assert {row.id for row in pool} == {1, 2}


def test_eligible_message_pool_denylist_wins_over_include_channels(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "food")
    _message_with_exchange(conn, 1, 1)
    pool = eligible_message_pool(
        conn, exclude_channels=frozenset({"food"}), include_channels=frozenset({"food"})
    )
    assert pool == []


def test_select_sift_sample_respects_exclude_channels(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    _message_with_exchange(conn, 1, 1)
    _message_with_exchange(conn, 2, 2)
    selected = select_sift_sample(
        conn, 10, seed=0, strategy=SiftStrategy.RANDOM, exclude_channels=frozenset({"food"})
    )
    assert selected == [1]


def test_select_sift_sample_respects_include_channels(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    _message_with_exchange(conn, 1, 1)
    _message_with_exchange(conn, 2, 2)
    selected = select_sift_sample(
        conn, 10, seed=0, strategy=SiftStrategy.RANDOM, include_channels=frozenset({"general"})
    )
    assert selected == [1]
