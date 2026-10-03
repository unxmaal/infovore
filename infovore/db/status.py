import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from infovore.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TRIAGE_MIN_P_LORE,
    DEFAULT_TRIAGE_MIN_SCORE,
)
from infovore.db.channel_filter import include_channels_clause
from infovore.db.codec import from_db_time, to_db_time
from infovore.db.exchanges import claimable_condition
from infovore.db.labels import label_counts
from infovore.triage.gate import gate_sql
from infovore.triage.rules import DEFAULT_RULES, TriageRules


@dataclass(frozen=True)
class StatusReport:
    channels: int
    messages: int
    deleted_messages: int
    exchanges_by_status: dict[str, int]
    claims_by_novelty: dict[str, int]
    retracted_claims: int
    runs_by_outcome: dict[str, int]
    last_extraction_at: datetime | None
    last_probe_at: datetime | None
    live_prompt_version: str | None
    triaged_exchanges: int
    above_threshold_exchanges: int
    labels_by_source: dict[str, dict[str, int]]
    labels_effective: dict[str, int]
    latest_model_version: int | None
    latest_model_labels_used: int | None
    p_lore_scored: int
    passing_gate: int
    excluded_by_denylist: int
    pending_exchanges: int
    pending_gated: int
    extraction_per_hour: float | None
    probe_per_hour: float | None
    extraction_eta_hours: float | None
    extraction_input_tokens: int
    extraction_output_tokens: int
    extraction_cost_usd: float | None
    probe_input_tokens: int
    probe_output_tokens: int
    probe_cost_usd: float | None
    probe_claims: int
    throughput_window_hours: int


def _count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def _count_params(conn: sqlite3.Connection, sql: str, params: tuple[object, ...]) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def _counts(conn: sqlite3.Connection, sql: str) -> dict[str, int]:
    return {str(row[0]): int(row[1]) for row in conn.execute(sql)}


def _latest_time(conn: sqlite3.Connection, sql: str) -> datetime | None:
    value = conn.execute(sql).fetchone()[0]
    return from_db_time(value) if value is not None else None


def _live_prompt_version(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT version FROM prompt_versions WHERE promoted_at IS NOT NULL"
        " ORDER BY promoted_at DESC LIMIT 1"
    ).fetchone()
    return str(row[0]) if row is not None else None


def _latest_model(conn: sqlite3.Connection) -> tuple[int | None, int | None]:
    row = conn.execute(
        "SELECT version, labels_used FROM triage_model ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None, None
    return int(row["version"]), int(row["labels_used"])


def _excluded_by_denylist(
    conn: sqlite3.Connection,
    gate_clause: str,
    gate_params: tuple[float, float],
    exclude_channels: frozenset[str],
) -> int:
    if not exclude_channels:
        return 0
    denylist_clause, denylist_params = include_channels_clause(
        "exchanges.channel_id", exclude_channels
    )
    return _count_params(
        conn,
        f"SELECT COUNT(*) FROM current_exchanges AS exchanges WHERE {gate_clause}{denylist_clause}",
        (*gate_params, *denylist_params),
    )


DEFAULT_THROUGHPUT_WINDOW_HOURS = 6


def _sum(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    value = conn.execute(sql, params).fetchone()[0]
    return int(value) if value is not None else 0


def _sum_optional(
    conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()
) -> float | None:
    value = conn.execute(sql, params).fetchone()[0]
    return float(value) if value is not None else None


def _rate_per_hour(count: int, window_hours: int) -> float | None:
    if count == 0:
        return None
    return count / window_hours


def _eta_hours(pending: int, rate: float | None) -> float | None:
    if pending == 0:
        return 0.0
    if rate is None or rate <= 0:
        return None
    return pending / rate


def collect_status(
    conn: sqlite3.Connection,
    triage_min_score: float = DEFAULT_TRIAGE_MIN_SCORE,
    triage_min_p_lore: float = DEFAULT_TRIAGE_MIN_P_LORE,
    rules: TriageRules = DEFAULT_RULES,
    exclude_channels: frozenset[str] = frozenset(),
    *,
    now: datetime | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    throughput_window_hours: int = DEFAULT_THROUGHPUT_WINDOW_HOURS,
) -> StatusReport:
    counts = label_counts(conn)
    latest_model_version, latest_model_labels_used = _latest_model(conn)
    gate_clause, gate_params = gate_sql(triage_min_score, triage_min_p_lore)

    pending_clause, pending_params = claimable_condition(max_retries)
    gated_clause, gated_params = claimable_condition(
        max_retries, triage_min_score, triage_min_p_lore, exclude_channels
    )
    pending_gated = _count_params(
        conn,
        f"SELECT COUNT(*) FROM current_exchanges AS exchanges WHERE {gated_clause}",
        tuple(gated_params),
    )

    since = to_db_time(now - timedelta(hours=throughput_window_hours)) if now else None
    recent_runs = (
        _count_params(
            conn,
            "SELECT COUNT(*) FROM extraction_runs WHERE started_at >= ?",
            (since,),
        )
        if since is not None
        else 0
    )
    recent_probe_claims = (
        _sum(
            conn,
            "SELECT SUM(claim_count) FROM probe_runs WHERE started_at >= ?",
            (since,),
        )
        if since is not None
        else 0
    )
    extraction_per_hour = _rate_per_hour(recent_runs, throughput_window_hours)

    return StatusReport(
        channels=_count(conn, "SELECT COUNT(*) FROM channels"),
        messages=_count(conn, "SELECT COUNT(*) FROM messages"),
        deleted_messages=_count(conn, "SELECT COUNT(*) FROM messages WHERE deleted_at IS NOT NULL"),
        exchanges_by_status=_counts(
            conn, "SELECT extraction_status, COUNT(*) FROM current_exchanges GROUP BY 1 ORDER BY 1"
        ),
        claims_by_novelty=_counts(
            conn,
            "SELECT novelty, COUNT(*) FROM claims WHERE retracted_at IS NULL GROUP BY 1 ORDER BY 1",
        ),
        retracted_claims=_count(conn, "SELECT COUNT(*) FROM claims WHERE retracted_at IS NOT NULL"),
        runs_by_outcome=_counts(
            conn, "SELECT outcome, COUNT(*) FROM extraction_runs GROUP BY 1 ORDER BY 1"
        ),
        last_extraction_at=_latest_time(conn, "SELECT MAX(started_at) FROM extraction_runs"),
        last_probe_at=_latest_time(conn, "SELECT MAX(probed_at) FROM claims"),
        live_prompt_version=_live_prompt_version(conn),
        triaged_exchanges=_count_params(
            conn,
            "SELECT COUNT(*) FROM current_exchanges WHERE triage_version = ?",
            (rules.version,),
        ),
        above_threshold_exchanges=_count_params(
            conn,
            "SELECT COUNT(*) FROM current_exchanges WHERE triage_version = ? AND triage_score >= ?",
            (rules.version, triage_min_score),
        ),
        labels_by_source=counts.by_source,
        labels_effective=counts.effective,
        latest_model_version=latest_model_version,
        latest_model_labels_used=latest_model_labels_used,
        p_lore_scored=_count(
            conn, "SELECT COUNT(*) FROM current_exchanges WHERE p_lore IS NOT NULL"
        ),
        passing_gate=_count_params(
            conn,
            f"SELECT COUNT(*) FROM current_exchanges AS exchanges WHERE {gate_clause}",
            gate_params,
        ),
        excluded_by_denylist=_excluded_by_denylist(
            conn, gate_clause, gate_params, exclude_channels
        ),
        pending_exchanges=_count_params(
            conn,
            f"SELECT COUNT(*) FROM current_exchanges AS exchanges WHERE {pending_clause}",
            tuple(pending_params),
        ),
        pending_gated=pending_gated,
        extraction_per_hour=extraction_per_hour,
        probe_per_hour=_rate_per_hour(recent_probe_claims, throughput_window_hours),
        extraction_eta_hours=_eta_hours(pending_gated, extraction_per_hour),
        extraction_input_tokens=_sum(conn, "SELECT SUM(input_tokens) FROM extraction_runs"),
        extraction_output_tokens=_sum(conn, "SELECT SUM(output_tokens) FROM extraction_runs"),
        extraction_cost_usd=_sum_optional(conn, "SELECT SUM(cost_usd) FROM extraction_runs"),
        probe_input_tokens=_sum(
            conn,
            "SELECT COALESCE(SUM(recall_input_tokens), 0) + COALESCE(SUM(judge_input_tokens), 0)"
            " FROM probe_runs",
        ),
        probe_output_tokens=_sum(
            conn,
            "SELECT COALESCE(SUM(recall_output_tokens), 0) + COALESCE(SUM(judge_output_tokens), 0)"
            " FROM probe_runs",
        ),
        probe_cost_usd=_sum_optional(conn, "SELECT SUM(cost_usd) FROM probe_runs"),
        probe_claims=_sum(conn, "SELECT SUM(claim_count) FROM probe_runs"),
        throughput_window_hours=throughput_window_hours,
    )
