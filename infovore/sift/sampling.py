import random
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from infovore.db.channel_filter import exclude_channels_clause, include_channels_clause

DEFAULT_MIX_FRACTION_UNCERTAIN = 0.5


class SiftStrategy(StrEnum):
    RANDOM = "random"
    UNCERTAIN = "uncertain"
    MIXED = "mixed"


class NoScoredMessagesError(Exception):
    pass


@dataclass(frozen=True)
class SiftCandidate:
    id: int
    channel_id: int
    exchange_id: int
    p_trash: float | None


def eligible_message_pool(
    conn: sqlite3.Connection,
    exclude_channels: frozenset[str] = frozenset(),
    include_channels: frozenset[str] = frozenset(),
) -> list[SiftCandidate]:
    """Every message a sift batch is allowed to offer (issue #128): it must
    belong to an exchange (`exchange_messages` — a message never grouped
    isn't part of the corpus this loop is triaging), its author must not be
    opted out (their history is never surfaced, redacted or not), and it
    must not already carry a `human` label (a batch never re-offers a
    message the maintainer already sifted). `exclude_channels` (the
    denylist, issue #138) and `include_channels` (`--channels`) further
    restrict the pool by channel name, resolving a thread to its parent
    channel's name; the denylist always wins when a channel is named in
    both."""
    excl_clause, excl_params = exclude_channels_clause("m.channel_id", exclude_channels)
    incl_clause, incl_params = include_channels_clause("m.channel_id", include_channels)
    rows = conn.execute(
        "SELECT m.id AS id, m.channel_id AS channel_id, em.exchange_id AS exchange_id,"
        " m.p_trash AS p_trash"
        " FROM messages m"
        " JOIN exchange_messages em ON em.message_id = m.id"
        " WHERE m.author_id NOT IN (SELECT user_id FROM opt_outs)"
        " AND NOT EXISTS ("
        "   SELECT 1 FROM message_labels ml"
        "   WHERE ml.message_id = m.id AND ml.source = 'human'"
        f" ){excl_clause}{incl_clause}"
        " ORDER BY m.id",
        (*excl_params, *incl_params),
    ).fetchall()
    return [
        SiftCandidate(
            id=row["id"],
            channel_id=row["channel_id"],
            exchange_id=row["exchange_id"],
            p_trash=row["p_trash"],
        )
        for row in rows
    ]


def repeat_message_pool(
    conn: sqlite3.Connection,
    exclude_channels: frozenset[str] = frozenset(),
    include_channels: frozenset[str] = frozenset(),
) -> list[SiftCandidate]:
    """The mirror of `eligible_message_pool`: messages that ALREADY carry a
    human label, so a round can re-offer a handful of them unannounced and
    measure the maintainer's agreement with his own earlier judgment. That
    rate is the ceiling no technique can beat (issue #166)."""
    excl_clause, excl_params = exclude_channels_clause("m.channel_id", exclude_channels)
    incl_clause, incl_params = include_channels_clause("m.channel_id", include_channels)
    rows = conn.execute(
        "SELECT m.id AS id, m.channel_id AS channel_id, em.exchange_id AS exchange_id,"
        " m.p_trash AS p_trash"
        " FROM messages m"
        " JOIN exchange_messages em ON em.message_id = m.id"
        " WHERE m.author_id NOT IN (SELECT user_id FROM opt_outs)"
        " AND EXISTS ("
        "   SELECT 1 FROM message_labels ml"
        "   WHERE ml.message_id = m.id AND ml.source = 'human'"
        f" ){excl_clause}{incl_clause}"
        " ORDER BY m.id",
        (*excl_params, *incl_params),
    ).fetchall()
    return [
        SiftCandidate(
            id=row["id"],
            channel_id=row["channel_id"],
            exchange_id=row["exchange_id"],
            p_trash=row["p_trash"],
        )
        for row in rows
    ]


def _by_channel(rows: Sequence[SiftCandidate]) -> dict[int, list[SiftCandidate]]:
    strata: dict[int, list[SiftCandidate]] = {}
    for row in rows:
        strata.setdefault(row.channel_id, []).append(row)
    return strata


def _allocate_round_robin(strata: Mapping[int, Sequence[SiftCandidate]], n: int) -> dict[int, int]:
    """Split `n` across channel strata round-robin (mirrors the stratified
    trial sampler in `infovore.extract.runner.select_trial_sample`), so a
    batch spans channels instead of letting a chatty one like #general
    dominate it."""
    keys = sorted(strata)
    total = sum(len(pool) for pool in strata.values())
    if n >= total:
        return {key: len(strata[key]) for key in keys}
    allocation = {key: 0 for key in keys}
    remaining = n
    index = 0
    while remaining > 0:
        key = keys[index % len(keys)]
        if allocation[key] < len(strata[key]):
            allocation[key] += 1
            remaining -= 1
        index += 1
    return allocation


def _select_random_stratified(rows: Sequence[SiftCandidate], n: int, seed: int) -> list[int]:
    if not rows:
        return []
    strata = _by_channel(rows)
    allocation = _allocate_round_robin(strata, n)
    rng = random.Random(seed)
    selected: list[int] = []
    for channel_id, count in allocation.items():
        pool = strata[channel_id]
        ids = [row.id for row in pool]
        selected.extend(ids if count >= len(ids) else rng.sample(ids, count))
    return selected


def _uncertainty(row: SiftCandidate) -> tuple[float, int]:
    assert row.p_trash is not None
    return (abs(row.p_trash - 0.5), row.id)


def _select_uncertain_stratified(rows: Sequence[SiftCandidate], n: int) -> list[int]:
    scored = [row for row in rows if row.p_trash is not None]
    if not scored:
        raise NoScoredMessagesError
    strata = _by_channel(scored)
    allocation = _allocate_round_robin(strata, n)
    selected: list[int] = []
    for channel_id, count in allocation.items():
        pool = sorted(strata[channel_id], key=_uncertainty)
        selected.extend(row.id for row in pool[:count])
    return selected


def select_sift_sample(
    conn: sqlite3.Connection,
    n: int,
    seed: int,
    strategy: SiftStrategy,
    mix: float = DEFAULT_MIX_FRACTION_UNCERTAIN,
    exclude_channels: frozenset[str] = frozenset(),
    include_channels: frozenset[str] = frozenset(),
    repeat: int = 0,
) -> list[int]:
    """Pick `n` message ids from `eligible_message_pool`, channel-stratified
    (issue #128): `random` draws uniformly within each channel's
    round-robin allocation; `uncertain` takes, within that same
    allocation, the messages whose `p_trash` is closest to 0.5 (least sure)
    and raises `NoScoredMessagesError` if nothing has been scored yet
    (mirrors `--strategy uncertain` in `infovore.extract.runner`, which the
    caller maps to a `ConfigError`/exit `2`); `mixed` splits `n` between the
    two (`mix`, default 50/50) but — unlike plain `uncertain` — falls back
    to an all-`random` split when nothing anywhere has a `p_trash` yet, so a
    mixed round never has to wait on a trained classifier."""
    repeats = _select_repeats(conn, repeat, n, seed, exclude_channels, include_channels)
    n -= len(repeats)
    rows = eligible_message_pool(conn, exclude_channels, include_channels)
    if strategy is SiftStrategy.RANDOM:
        return sorted(repeats | set(_select_random_stratified(rows, n, seed)))
    if strategy is SiftStrategy.UNCERTAIN:
        return sorted(repeats | set(_select_uncertain_stratified(rows, n)))

    scored = [row for row in rows if row.p_trash is not None]
    if not scored:
        return sorted(repeats | set(_select_random_stratified(rows, n, seed)))

    n_uncertain = max(0, min(round(n * mix), len(scored), n))
    uncertain_ids = _select_uncertain_stratified(rows, n_uncertain) if n_uncertain else []
    uncertain_id_set = set(uncertain_ids)
    remaining_rows = [row for row in rows if row.id not in uncertain_id_set]
    random_ids = _select_random_stratified(remaining_rows, n - len(uncertain_ids), seed)
    return sorted(repeats | uncertain_id_set | set(random_ids))


def _select_repeats(
    conn: sqlite3.Connection,
    repeat: int,
    n: int,
    seed: int,
    exclude_channels: frozenset[str],
    include_channels: frozenset[str],
) -> set[int]:
    wanted = min(repeat, n)
    if wanted <= 0:
        return set()
    pool = repeat_message_pool(conn, exclude_channels, include_channels)
    return set(_select_random_stratified(pool, wanted, seed))
