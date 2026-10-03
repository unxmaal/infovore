import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from infovore.db.batch import BATCH_SIZE, ExchangeInputs, exchange_inputs_for_ids
from infovore.db.connection import transaction
from infovore.triage.parallel import ChunkPool
from infovore.triage.rules import DEFAULT_RULES, TriageRules
from infovore.triage.score import TriageResult, score_exchange
from infovore.triage.train import load_latest_model, score_stale

CHANNEL_PRIOR_WEIGHT = 0.3
CHANNEL_PRIOR_CAP = 0.1
CHANNEL_PRIOR_REASON = "channel_prior"


@dataclass(frozen=True)
class TriageStarted:
    total: int


@dataclass(frozen=True)
class TriageExchangeScored:
    exchange_id: int
    score: float
    index: int
    total: int


@dataclass(frozen=True)
class TriagePriorsApplied:
    channels: int


TriageEvent = TriageStarted | TriageExchangeScored | TriagePriorsApplied
TriageProgress = Callable[[TriageEvent], None]


def _ignore_triage_progress(event: TriageEvent) -> None:
    return None


@dataclass(frozen=True)
class TriageReport:
    candidates: int
    scored: int
    channels_adjusted: int
    channel_means: dict[int, float]
    channel_priors: dict[int, float]
    global_mean: float


def _raw_score(reasons_json: str) -> float:
    reasons: list[tuple[str, float]] = json.loads(reasons_json)
    raw = sum(weight for name, weight in reasons if name != CHANNEL_PRIOR_REASON)
    return round(min(1.0, max(0.0, raw)), 4)


# Module-level (spawn-safe) worker state for `ChunkPool`: `rules` is sent to
# each worker once, via the pool's `initializer`, not per task (issue #113).
# `_init_rule_score_worker` also runs directly, in-process, for the
# `workers <= 1` path -- see `infovore.triage.parallel`'s module docstring.
_worker_rules: TriageRules = DEFAULT_RULES


def _init_rule_score_worker(rules: TriageRules) -> None:
    global _worker_rules
    _worker_rules = rules


def _rule_score_worker_chunk(
    items: list[tuple[int, ExchangeInputs]],
) -> list[tuple[int, TriageResult]]:
    """`items`: `(exchange_id, inputs)` for a chunk of exchanges. Returns
    `(exchange_id, TriageResult)` for each, in the same order."""
    rules = _worker_rules
    return [
        (exchange_id, score_exchange(inputs.messages, inputs.reactions, inputs.attachments, rules))
        for exchange_id, inputs in items
    ]


def _score_batch(
    conn: sqlite3.Connection,
    exchange_ids: Sequence[int],
    rules: TriageRules,
    progress: TriageProgress,
    total: int,
    scored_before_this_batch: int,
    pool: ChunkPool[tuple[int, ExchangeInputs], list[tuple[int, TriageResult]]],
) -> None:
    inputs = exchange_inputs_for_ids(conn, exchange_ids)
    items = [(exchange_id, inputs[exchange_id]) for exchange_id in exchange_ids]
    scored = [result for chunk_result in pool.map_chunks(items) for result in chunk_result]

    updates: list[tuple[float, str, str, int]] = []
    for offset, (exchange_id, result) in enumerate(scored):
        updates.append(
            (
                result.score,
                json.dumps([list(pair) for pair in result.reasons]),
                rules.version,
                exchange_id,
            )
        )
        progress(
            TriageExchangeScored(
                exchange_id=exchange_id,
                score=result.score,
                index=scored_before_this_batch + offset + 1,
                total=total,
            )
        )
    with transaction(conn):
        conn.executemany(
            "UPDATE exchanges SET triage_score = ?, triage_reasons = ?, triage_version = ?"
            " WHERE id = ?",
            updates,
        )


def _channel_means(conn: sqlite3.Connection, rules: TriageRules) -> tuple[dict[int, float], float]:
    rows = conn.execute(
        "SELECT channel_id, triage_reasons FROM current_exchanges WHERE triage_version = ?",
        (rules.version,),
    ).fetchall()
    raw_by_channel: dict[int, list[float]] = {}
    all_raw: list[float] = []
    for row in rows:
        raw = _raw_score(row["triage_reasons"])
        raw_by_channel.setdefault(row["channel_id"], []).append(raw)
        all_raw.append(raw)
    channel_means = {
        channel_id: sum(values) / len(values) for channel_id, values in raw_by_channel.items()
    }
    global_mean = sum(all_raw) / len(all_raw) if all_raw else 0.0
    return channel_means, global_mean


def _clip_delta(delta: float) -> float:
    return max(-CHANNEL_PRIOR_CAP, min(CHANNEL_PRIOR_CAP, delta))


def _apply_channel_priors(
    conn: sqlite3.Connection,
    channel_means: dict[int, float],
    global_mean: float,
    rules: TriageRules,
) -> dict[int, float]:
    priors: dict[int, float] = {}
    rows = conn.execute(
        "SELECT id, channel_id, triage_score, triage_reasons FROM current_exchanges"
        " WHERE triage_version = ?",
        (rules.version,),
    ).fetchall()
    updates: list[tuple[float, str, int]] = []
    for row in rows:
        channel_id = row["channel_id"]
        channel_mean = channel_means.get(channel_id, global_mean)
        delta = round(_clip_delta(CHANNEL_PRIOR_WEIGHT * (channel_mean - global_mean)), 4)
        priors[channel_id] = delta
        raw = _raw_score(row["triage_reasons"])
        adjusted = round(min(1.0, max(0.0, raw + delta)), 4)
        reasons = [
            pair for pair in json.loads(row["triage_reasons"]) if pair[0] != CHANNEL_PRIOR_REASON
        ]
        reasons.append([CHANNEL_PRIOR_REASON, delta])
        reasons_json = json.dumps(reasons)
        if adjusted == row["triage_score"] and reasons_json == row["triage_reasons"]:
            continue
        updates.append((adjusted, reasons_json, row["id"]))

    for start in range(0, len(updates), BATCH_SIZE):
        batch = updates[start : start + BATCH_SIZE]
        with transaction(conn):
            conn.executemany(
                "UPDATE exchanges SET triage_score = ?, triage_reasons = ? WHERE id = ?", batch
            )
    return priors


def triage_pending(
    conn: sqlite3.Connection,
    progress: TriageProgress = _ignore_triage_progress,
    rules: TriageRules = DEFAULT_RULES,
    workers: int = 1,
) -> TriageReport:
    candidate_rows = conn.execute(
        "SELECT id FROM current_exchanges WHERE triage_version IS NULL OR triage_version != ?"
        " ORDER BY id",
        (rules.version,),
    ).fetchall()
    candidate_ids = [row["id"] for row in candidate_rows]
    progress(TriageStarted(total=len(candidate_ids)))
    with ChunkPool(_rule_score_worker_chunk, workers, _init_rule_score_worker, (rules,)) as pool:
        for start in range(0, len(candidate_ids), BATCH_SIZE):
            batch = candidate_ids[start : start + BATCH_SIZE]
            _score_batch(conn, batch, rules, progress, len(candidate_ids), start, pool)

    channel_means, global_mean = _channel_means(conn, rules)
    channel_priors = _apply_channel_priors(conn, channel_means, global_mean, rules)
    progress(TriagePriorsApplied(channels=len(channel_priors)))

    loaded_model = load_latest_model(conn)
    if loaded_model is not None:
        model_version, model = loaded_model
        score_stale(conn, model, model_version, rules, workers=workers)

    return TriageReport(
        candidates=len(candidate_ids),
        scored=len(candidate_ids),
        channels_adjusted=len(channel_priors),
        channel_means=channel_means,
        channel_priors=channel_priors,
        global_mean=global_mean,
    )
