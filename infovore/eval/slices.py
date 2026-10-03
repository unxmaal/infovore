import random
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from infovore.db.codec import to_db_time
from infovore.db.exchanges import claimable_condition
from infovore.extract.prompt_compare import SIZE_BUCKETS
from infovore.triage.gate import GATE_SQL_CLAUSE

SLICE_SEED = 190

BUILD = "s1"
HOLDOUT = "s2"
REJECTED = "c1"
GOLD = "gold"
GOLD_REPEATS = "gold-repeats"

BUILD_SIZE = 200
REJECTED_SIZE = 50
GOLD_FROM_BUILD = 40
GOLD_FROM_REJECTED = 10
GOLD_REPEAT_COUNT = 5

Bucket = tuple[int, int]


class SliceExistsError(ValueError):
    pass


class SliceTooSmallError(ValueError):
    pass


@dataclass(frozen=True)
class BucketSummary:
    bucket: Bucket
    exchanges: int
    messages: int


def bucket_of(message_count: int) -> Bucket:
    for low, high in SIZE_BUCKETS:
        if low <= message_count <= high:
            return (low, high)
    return SIZE_BUCKETS[-1]  # pragma: no cover - the last bucket is unbounded


def allocate(mix: Mapping[Bucket, int], size: int) -> dict[Bucket, int]:
    """Split `size` across buckets in proportion to `mix`, by largest
    remainder, so the parts sum to exactly `size`."""
    total = sum(mix.values())
    if total == 0:
        return {bucket: 0 for bucket in SIZE_BUCKETS}
    exact = {bucket: size * mix.get(bucket, 0) / total for bucket in SIZE_BUCKETS}
    counts = {bucket: int(value) for bucket, value in exact.items()}
    by_remainder = sorted(
        SIZE_BUCKETS, key=lambda b: (exact[b] - counts[b], -SIZE_BUCKETS.index(b)), reverse=True
    )
    for bucket in by_remainder[: size - sum(counts.values())]:
        counts[bucket] += 1
    return counts


def _population(
    conn: sqlite3.Connection, condition: str, params: Sequence[object]
) -> dict[Bucket, list[int]]:
    pools: dict[Bucket, list[int]] = {bucket: [] for bucket in SIZE_BUCKETS}
    for row in conn.execute(
        f"SELECT id, message_count FROM exchanges WHERE {condition} ORDER BY id",
        tuple(params),
    ):
        pools[bucket_of(row["message_count"])].append(row["id"])
    return pools


def queue_population(
    conn: sqlite3.Connection,
    *,
    max_retries: int,
    min_score: float,
    min_p_lore: float,
    exclude_channels: frozenset[str],
) -> dict[Bucket, list[int]]:
    """What extraction would work on next, by the same predicate the queue
    and `status` use, so the slice cannot describe a different queue."""
    condition, params = claimable_condition(max_retries, min_score, min_p_lore, exclude_channels)
    return _population(conn, condition, params)


def rejected_population(
    conn: sqlite3.Connection,
    *,
    max_retries: int,
    min_score: float,
    min_p_lore: float,
    exclude_channels: frozenset[str],
) -> dict[Bucket, list[int]]:
    """Claimable exchanges the gate turns away. The control for what the
    gate throws out, which the gate's own numbers cannot show."""
    condition, params = claimable_condition(max_retries, None, min_p_lore, exclude_channels)
    return _population(
        conn, f"{condition} AND NOT {GATE_SQL_CLAUSE}", [*params, min_p_lore, min_score]
    )


def _draw(
    rng: random.Random, pools: Mapping[Bucket, list[int]], counts: Mapping[Bucket, int]
) -> dict[Bucket, list[int]]:
    drawn: dict[Bucket, list[int]] = {}
    for bucket in SIZE_BUCKETS:
        want = counts.get(bucket, 0)
        pool = pools.get(bucket, [])
        if want > len(pool):
            raise SliceTooSmallError(
                f"bucket {bucket} needs {want} exchanges and the population has {len(pool)}"
            )
        drawn[bucket] = rng.sample(pool, want)
    return drawn


def _insert(
    conn: sqlite3.Connection,
    name: str,
    exchange_ids: Sequence[int],
    population: str,
    seed: int,
    at: datetime,
) -> None:
    conn.executemany(
        "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [
            (name, exchange_id, position, population, seed, to_db_time(at))
            for position, exchange_id in enumerate(exchange_ids, start=1)
        ],
    )


def freeze_slices(
    conn: sqlite3.Connection,
    *,
    max_retries: int,
    min_score: float,
    min_p_lore: float,
    exclude_channels: frozenset[str],
    at: datetime,
    seed: int = SLICE_SEED,
) -> dict[str, list[int]]:
    """Freeze the build slice, its held-out twin, the gate-rejected control,
    and the gold set Eric reads end to end (#190). Done once: a second call
    is refused, because every judgment is keyed to these exact exchanges."""
    existing = conn.execute("SELECT COUNT(*) AS n FROM eval_slices").fetchone()["n"]
    if existing:
        raise SliceExistsError(f"slices are already frozen ({existing} rows)")
    queue = queue_population(
        conn,
        max_retries=max_retries,
        min_score=min_score,
        min_p_lore=min_p_lore,
        exclude_channels=exclude_channels,
    )
    rejected = rejected_population(
        conn,
        max_retries=max_retries,
        min_score=min_score,
        min_p_lore=min_p_lore,
        exclude_channels=exclude_channels,
    )
    mix = {bucket: len(ids) for bucket, ids in queue.items()}
    rng = random.Random(seed)

    # S1 and S2 are drawn together and split, so they are disjoint by
    # construction and both match the queue's size mix.
    build_counts = allocate(mix, BUILD_SIZE)
    paired = _draw(rng, queue, {bucket: 2 * n for bucket, n in build_counts.items()})
    build = [i for b in SIZE_BUCKETS for i in paired[b][: build_counts[b]]]
    holdout = [i for b in SIZE_BUCKETS for i in paired[b][build_counts[b] :]]

    # The control uses the QUEUE's size mix, so it differs from S1 only in
    # what the gate decided, not in exchange size.
    control_draw = _draw(rng, rejected, allocate(mix, REJECTED_SIZE))
    control = [i for b in SIZE_BUCKETS for i in control_draw[b]]

    gold = rng.sample(build, GOLD_FROM_BUILD) + rng.sample(control, GOLD_FROM_REJECTED)
    rng.shuffle(gold)
    # Repeats come from the first half and are shown again after the rest,
    # unannounced, so the two judgments of one exchange are far apart.
    repeats = rng.sample(gold[: len(gold) // 2], GOLD_REPEAT_COUNT)

    plan = {
        BUILD: (build, "queue"),
        HOLDOUT: (holdout, "queue"),
        REJECTED: (control, "gate-rejected"),
        GOLD: (gold, "s1+c1"),
        GOLD_REPEATS: (repeats, "gold"),
    }
    with conn:
        for name, (ids, population) in plan.items():
            _insert(conn, name, ids, population, seed, at)
    return {name: ids for name, (ids, _) in plan.items()}


def slice_ids(conn: sqlite3.Connection, name: str) -> list[int]:
    return [
        row["exchange_id"]
        for row in conn.execute(
            "SELECT exchange_id FROM current_eval_slices WHERE name = ? ORDER BY position", (name,)
        )
    ]


def slice_summary(conn: sqlite3.Connection, name: str) -> list[BucketSummary]:
    counts: dict[Bucket, list[int]] = {bucket: [0, 0] for bucket in SIZE_BUCKETS}
    for row in conn.execute(
        "SELECT e.message_count AS n FROM current_eval_slices s"
        " JOIN exchanges e ON e.id = s.exchange_id"
        " WHERE s.name = ?",
        (name,),
    ):
        entry = counts[bucket_of(row["n"])]
        entry[0] += 1
        entry[1] += row["n"]
    return [
        BucketSummary(bucket, exchanges, messages)
        for bucket, (exchanges, messages) in counts.items()
    ]


def slice_names(conn: sqlite3.Connection) -> list[str]:
    names = (
        row["name"].split("@")[0]
        for row in conn.execute(
            "SELECT name, MIN(rowid) AS first FROM eval_slices GROUP BY name ORDER BY first"
        )
    )
    return list(dict.fromkeys(names))
