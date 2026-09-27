import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from infovore.db.connection import transaction
from infovore.db.exchanges import exchange_message_ids
from infovore.db.raw import attachments_for_messages, messages_by_ids, reactions_for_messages
from infovore.triage.rules import DEFAULT_RULES, TriageRules
from infovore.triage.score import score_exchange
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


def _score_one(conn: sqlite3.Connection, exchange_id: int, rules: TriageRules) -> float:
    message_ids = exchange_message_ids(conn, exchange_id)
    messages = messages_by_ids(conn, message_ids)
    reactions = reactions_for_messages(conn, message_ids)
    attachments = attachments_for_messages(conn, message_ids)
    result = score_exchange(messages, reactions, attachments, rules)
    with transaction(conn):
        conn.execute(
            "UPDATE exchanges SET triage_score = ?, triage_reasons = ?, triage_version = ?"
            " WHERE id = ?",
            (
                result.score,
                json.dumps([list(pair) for pair in result.reasons]),
                rules.version,
                exchange_id,
            ),
        )
    return result.score


def _channel_means(conn: sqlite3.Connection, rules: TriageRules) -> tuple[dict[int, float], float]:
    rows = conn.execute(
        "SELECT channel_id, triage_reasons FROM exchanges WHERE triage_version = ?",
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
        "SELECT id, channel_id, triage_reasons FROM exchanges WHERE triage_version = ?",
        (rules.version,),
    ).fetchall()
    for row in rows:
        channel_id = row["channel_id"]
        channel_mean = channel_means.get(channel_id, global_mean)
        delta = round(_clip_delta(CHANNEL_PRIOR_WEIGHT * (channel_mean - global_mean)), 4)
        raw = _raw_score(row["triage_reasons"])
        adjusted = round(min(1.0, max(0.0, raw + delta)), 4)
        reasons = [
            pair for pair in json.loads(row["triage_reasons"]) if pair[0] != CHANNEL_PRIOR_REASON
        ]
        reasons.append([CHANNEL_PRIOR_REASON, delta])
        with transaction(conn):
            conn.execute(
                "UPDATE exchanges SET triage_score = ?, triage_reasons = ? WHERE id = ?",
                (adjusted, json.dumps(reasons), row["id"]),
            )
        priors[channel_id] = delta
    return priors


def triage_pending(
    conn: sqlite3.Connection,
    progress: TriageProgress = _ignore_triage_progress,
    rules: TriageRules = DEFAULT_RULES,
) -> TriageReport:
    candidate_rows = conn.execute(
        "SELECT id FROM exchanges WHERE triage_version IS NULL OR triage_version != ? ORDER BY id",
        (rules.version,),
    ).fetchall()
    candidate_ids = [row["id"] for row in candidate_rows]
    progress(TriageStarted(total=len(candidate_ids)))
    for index, exchange_id in enumerate(candidate_ids, start=1):
        score = _score_one(conn, exchange_id, rules)
        progress(
            TriageExchangeScored(
                exchange_id=exchange_id, score=score, index=index, total=len(candidate_ids)
            )
        )

    channel_means, global_mean = _channel_means(conn, rules)
    channel_priors = _apply_channel_priors(conn, channel_means, global_mean, rules)
    progress(TriagePriorsApplied(channels=len(channel_priors)))

    loaded_model = load_latest_model(conn)
    if loaded_model is not None:
        model_version, model = loaded_model
        score_stale(conn, model, model_version)

    return TriageReport(
        candidates=len(candidate_ids),
        scored=len(candidate_ids),
        channels_adjusted=len(channel_priors),
        channel_means=channel_means,
        channel_priors=channel_priors,
        global_mean=global_mean,
    )
