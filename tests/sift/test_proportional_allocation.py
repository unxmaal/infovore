import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.connection import migrate, open_database
from infovore.sift.sampling import (
    SiftAllocation,
    SiftCandidate,
    SiftStrategy,
    allocate,
    select_sift_sample,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC).isoformat()


def _strata(sizes: dict[int, int]) -> dict[int, list[SiftCandidate]]:
    return {
        channel_id: [
            SiftCandidate(
                id=channel_id * 100_000 + i, channel_id=channel_id, exchange_id=1, p_trash=None
            )
            for i in range(size)
        ]
        for channel_id, size in sizes.items()
    }


def test_proportional_allocation_mirrors_channel_share() -> None:
    strata = _strata({1: 900, 2: 100})

    allocation = allocate(strata, 100, SiftAllocation.PROPORTIONAL)

    assert allocation == {1: 90, 2: 10}


def test_proportional_allocation_sums_exactly_to_n() -> None:
    strata = _strata({1: 333, 2: 333, 3: 334})

    allocation = allocate(strata, 100, SiftAllocation.PROPORTIONAL)

    assert sum(allocation.values()) == 100


def test_proportional_allocation_uses_largest_remainder_not_truncation() -> None:
    strata = _strata({1: 1, 2: 1, 3: 1})

    allocation = allocate(strata, 2, SiftAllocation.PROPORTIONAL)

    assert sum(allocation.values()) == 2
    assert all(count <= 1 for count in allocation.values())


def test_proportional_allocation_drops_channels_too_small_to_earn_a_slot() -> None:
    strata = _strata({1: 9_999, 2: 1})

    allocation = allocate(strata, 10, SiftAllocation.PROPORTIONAL)

    assert allocation[1] == 10
    assert allocation[2] == 0


def test_proportional_allocation_never_exceeds_a_channels_pool() -> None:
    strata = _strata({1: 2, 2: 1_000})

    allocation = allocate(strata, 100, SiftAllocation.PROPORTIONAL)

    assert allocation[1] <= 2
    assert sum(allocation.values()) == 100


def test_proportional_allocation_takes_everything_when_n_exceeds_the_pool() -> None:
    strata = _strata({1: 3, 2: 2})

    allocation = allocate(strata, 99, SiftAllocation.PROPORTIONAL)

    assert allocation == {1: 3, 2: 2}


def test_round_robin_remains_the_default_allocation() -> None:
    strata = _strata({1: 900, 2: 100})

    allocation = allocate(strata, 100, SiftAllocation.ROUND_ROBIN)

    assert allocation == {1: 50, 2: 50}


def _corpus(tmp_path: Path, sizes: dict[str, int]) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    message_id = 1
    for channel_id, (name, size) in enumerate(sizes.items(), start=1):
        conn.execute(
            "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
            " VALUES (?, 1, NULL, ?, 'text')",
            (channel_id, name),
        )
        for _ in range(size):
            conn.execute(
                "INSERT INTO messages (id, channel_id, guild_id, author_id,"
                " author_name_at_time, created_at, content, ingested_at, raw_json)"
                " VALUES (?, ?, 1, 1, 'a', ?, 'hello', ?, '{}')",
                (message_id, channel_id, NOW, NOW),
            )
            conn.execute(
                "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
                " started_at, ended_at, message_count, grouping_rule, content_hash)"
                " VALUES (?, ?, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
                (message_id, channel_id, message_id, message_id, NOW, NOW, f"h{message_id}"),
            )
            conn.execute(
                "INSERT INTO exchange_messages (exchange_id, message_id, position)"
                " VALUES (?, ?, 1)",
                (message_id, message_id),
            )
            message_id += 1
    return conn


def _channel_counts(conn: sqlite3.Connection, ids: list[int]) -> dict[str, int]:
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT c.name, COUNT(*) FROM messages m JOIN channels c ON c.id = m.channel_id"
        f" WHERE m.id IN ({placeholders}) GROUP BY 1",
        ids,
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def test_select_sift_sample_proportional_follows_the_corpus(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, {"general": 800, "hardware": 200})

    picked = select_sift_sample(
        conn,
        n=100,
        seed=1,
        strategy=SiftStrategy.RANDOM,
        allocation=SiftAllocation.PROPORTIONAL,
    )

    assert _channel_counts(conn, picked) == {"general": 80, "hardware": 20}


def test_select_sift_sample_defaults_to_round_robin(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, {"general": 800, "hardware": 200})

    picked = select_sift_sample(conn, n=100, seed=1, strategy=SiftStrategy.RANDOM)

    assert _channel_counts(conn, picked) == {"general": 50, "hardware": 50}


def test_proportional_allocation_of_an_empty_pool_is_empty() -> None:
    assert allocate({}, 100, SiftAllocation.PROPORTIONAL) == {}


def test_proportional_allocation_of_empty_channels_gives_them_nothing() -> None:
    assert allocate(_strata({1: 0, 2: 0}), 10, SiftAllocation.PROPORTIONAL) == {1: 0, 2: 0}
