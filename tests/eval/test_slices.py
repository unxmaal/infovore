import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.eval.slices import (
    BUILD,
    GOLD,
    GOLD_REPEATS,
    HOLDOUT,
    REJECTED,
    SliceExistsError,
    SliceTooSmallError,
    allocate,
    bucket_of,
    freeze_slices,
    queue_population,
    rejected_population,
    slice_ids,
    slice_names,
    slice_summary,
)

AT = datetime(2026, 10, 2, tzinfo=UTC)
GATE = 0.9
SIZES = (1, 2, 4, 9, 20, 60)


def _seed(conn: sqlite3.Connection, passing: int, rejected: int) -> None:
    """`passing` gate-passing and `rejected` gate-rejected pending exchanges,
    cycling through every size bucket."""
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'c', 'text')"
    )
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (2, 1, NULL, 'food', 'text')"
    )
    rows = [(GATE + 0.05, 1)] * passing + [(GATE - 0.5, 1)] * rejected
    for index, (p_lore, channel) in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
            " ended_at, message_count, grouping_rule, content_hash, p_lore, triage_score)"
            " VALUES (?, ?, 1, 1, ?, ?, ?, 'quiet_gap', ?, ?, 0.5)",
            (
                index,
                channel,
                AT.isoformat(),
                AT.isoformat(),
                SIZES[index % len(SIZES)],
                f"h{index}",
                p_lore,
            ),
        )


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    _seed(connection, passing=900, rejected=300)
    return connection


def _freeze(conn: sqlite3.Connection, **overrides: object) -> dict[str, list[int]]:
    args: dict[str, object] = {
        "max_retries": 3,
        "min_score": 0.3,
        "min_p_lore": GATE,
        "exclude_channels": frozenset(),
        "at": AT,
    }
    args.update(overrides)
    return freeze_slices(conn, **args)  # type: ignore[arg-type]


def test_allocation_sums_exactly_and_follows_the_mix() -> None:
    mix = {(1, 2): 9949, (3, 5): 14899, (6, 15): 18974, (16, 49): 12052, (50, 10**9): 76}

    counts = allocate(mix, 200)

    assert sum(counts.values()) == 200
    assert counts == {(1, 2): 36, (3, 5): 53, (6, 15): 68, (16, 49): 43, (50, 10**9): 0}


def test_an_empty_mix_allocates_nothing() -> None:
    assert sum(allocate({}, 200).values()) == 0


def test_buckets_match_the_comparison_harness() -> None:
    """One definition of exchange size, shared with prompt_compare, so a
    slice and a comparison sample are comparable."""
    assert bucket_of(2) == (1, 2)
    assert bucket_of(3) == (3, 5)
    assert bucket_of(50) == (50, 10**9)


def test_the_slices_have_the_planned_sizes(conn: sqlite3.Connection) -> None:
    frozen = _freeze(conn)

    assert {name: len(ids) for name, ids in frozen.items()} == {
        BUILD: 200,
        HOLDOUT: 200,
        REJECTED: 50,
        GOLD: 50,
        GOLD_REPEATS: 5,
    }


def test_build_and_holdout_never_share_an_exchange(conn: sqlite3.Connection) -> None:
    frozen = _freeze(conn)

    assert not set(frozen[BUILD]) & set(frozen[HOLDOUT])


def test_the_control_comes_only_from_gate_rejected_exchanges(conn: sqlite3.Connection) -> None:
    frozen = _freeze(conn)
    p_lore = {r["id"]: r["p_lore"] for r in conn.execute("SELECT id, p_lore FROM exchanges")}

    assert all(p_lore[i] < GATE for i in frozen[REJECTED])
    assert all(p_lore[i] >= GATE for i in frozen[BUILD] + frozen[HOLDOUT])


def test_the_gold_set_is_forty_from_build_and_ten_from_the_control(
    conn: sqlite3.Connection,
) -> None:
    frozen = _freeze(conn)

    assert len(set(frozen[GOLD]) & set(frozen[BUILD])) == 40
    assert len(set(frozen[GOLD]) & set(frozen[REJECTED])) == 10


def test_repeats_come_from_the_first_half_of_the_gold_set(conn: sqlite3.Connection) -> None:
    """So the second judgment of an exchange lands far from the first."""
    frozen = _freeze(conn)

    assert set(frozen[GOLD_REPEATS]) <= set(frozen[GOLD][:25])


def test_freezing_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    results = []
    for name in ("a.db", "b.db"):
        connection = open_database(tmp_path / name)
        migrate(connection)
        _seed(connection, passing=900, rejected=300)
        results.append(_freeze(connection))

    assert results[0] == results[1]


def test_a_second_freeze_is_refused(conn: sqlite3.Connection) -> None:
    """Every judgment is keyed to these exact exchanges."""
    _freeze(conn)

    with pytest.raises(SliceExistsError):
        _freeze(conn)


def test_a_frozen_slice_cannot_be_edited(conn: sqlite3.Connection) -> None:
    _freeze(conn)

    with pytest.raises(sqlite3.IntegrityError, match="frozen"):
        conn.execute("UPDATE eval_slices SET exchange_id = 1")
    with pytest.raises(sqlite3.IntegrityError, match="frozen"):
        conn.execute("DELETE FROM eval_slices")


def test_a_population_too_small_for_its_bucket_is_refused(tmp_path: Path) -> None:
    connection = open_database(tmp_path / "small.db")
    migrate(connection)
    _seed(connection, passing=30, rejected=300)

    with pytest.raises(SliceTooSmallError):
        _freeze(connection)


def test_denylisted_channels_are_excluded_like_the_queue(tmp_path: Path) -> None:
    """The slice uses the queue's own predicate, so a denylisted channel is
    absent from both."""
    connection = open_database(tmp_path / "deny.db")
    migrate(connection)
    _seed(connection, passing=900, rejected=300)
    connection.execute("UPDATE exchanges SET channel_id = 2 WHERE id <= 10")

    pools = queue_population(
        connection,
        max_retries=3,
        min_score=0.3,
        min_p_lore=GATE,
        exclude_channels=frozenset({"food"}),
    )
    rejected = rejected_population(
        connection,
        max_retries=3,
        min_score=0.3,
        min_p_lore=GATE,
        exclude_channels=frozenset({"food"}),
    )

    every = {i for ids in pools.values() for i in ids} | {
        i for ids in rejected.values() for i in ids
    }
    assert not every & set(range(1, 11))


def test_slice_ids_come_back_in_frozen_order(conn: sqlite3.Connection) -> None:
    frozen = _freeze(conn)

    assert slice_ids(conn, GOLD) == frozen[GOLD]
    assert slice_names(conn) == [BUILD, HOLDOUT, REJECTED, GOLD, GOLD_REPEATS]


def test_the_summary_counts_exchanges_and_messages_per_bucket(conn: sqlite3.Connection) -> None:
    _freeze(conn)

    summary = {row.bucket: row for row in slice_summary(conn, BUILD)}

    assert sum(row.exchanges for row in summary.values()) == 200
    total = conn.execute(
        "SELECT SUM(e.message_count) AS n FROM eval_slices s"
        " JOIN exchanges e ON e.id = s.exchange_id WHERE s.name = ?",
        (BUILD,),
    ).fetchone()["n"]
    assert sum(row.messages for row in summary.values()) == total
