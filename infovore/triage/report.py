import json
import sqlite3
from dataclasses import dataclass

from infovore.triage.rules import DEFAULT_RULES, TriageRules

TOP_REASONS_LIMIT = 10


@dataclass(frozen=True)
class TriageStats:
    histogram: dict[str, int]
    channel_stats: dict[int, tuple[float, int]]
    above_threshold: int
    below_threshold: int
    top_reasons: list[tuple[str, int]]


def _bucket_label(score: float) -> str:
    index = min(9, int(score * 10))
    return f"{index / 10:.1f}-{(index + 1) / 10:.1f}"


def compute_triage_stats(
    conn: sqlite3.Connection, min_score: float, rules: TriageRules = DEFAULT_RULES
) -> TriageStats:
    rows = conn.execute(
        "SELECT channel_id, triage_score, triage_reasons FROM current_exchanges WHERE"
        " triage_version = ?",
        (rules.version,),
    ).fetchall()
    histogram: dict[str, int] = {}
    channel_scores: dict[int, list[float]] = {}
    above = 0
    below = 0
    reason_counts: dict[str, int] = {}
    for row in rows:
        score = row["triage_score"]
        bucket = _bucket_label(score)
        histogram[bucket] = histogram.get(bucket, 0) + 1
        channel_scores.setdefault(row["channel_id"], []).append(score)
        if score >= min_score:
            above += 1
        else:
            below += 1
        for name, _ in json.loads(row["triage_reasons"] or "[]"):
            reason_counts[name] = reason_counts.get(name, 0) + 1
    channel_stats = {
        channel_id: (sum(values) / len(values), len(values))
        for channel_id, values in channel_scores.items()
    }
    top_reasons = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[
        :TOP_REASONS_LIMIT
    ]
    return TriageStats(
        histogram=histogram,
        channel_stats=channel_stats,
        above_threshold=above,
        below_threshold=below,
        top_reasons=top_reasons,
    )
