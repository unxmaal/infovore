import math
import random
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from infovore.extract.prompt import system_prompt

# A call cannot be billed less input than the system prompt alone occupies,
# before any exchange text or harness overhead. Prompt v1/v2-era trial runs
# recorded as little as 2 input tokens; at 2 tokens a run is FREE, so it
# sorts to the front of any claims-per-token ordering and distorts every arm.
# Derived from the prompt rather than hard-coded, so it cannot rot.
MIN_PLAUSIBLE_INPUT_TOKENS = math.ceil(len(system_prompt("v5")) / 4)

ORACLE = "oracle (true claims/token)"
RANDOM = "random order (control)"


@dataclass(frozen=True)
class Candidate:
    exchange_id: int
    tokens: int
    claims: int
    p_lore: float


@dataclass(frozen=True)
class GainPoint:
    label: str
    claims_at_10_percent: float
    claims_at_25_percent: float
    budget_for_90_percent: float
    budget_for_95_percent: float


@dataclass(frozen=True)
class GainReport:
    mode: str
    exchanges: int
    total_tokens: int
    total_claims: int
    points: tuple[GainPoint, ...]
    excluded_implausible: int = 0
    min_input_tokens: int = MIN_PLAUSIBLE_INPUT_TOKENS


def _claims_at_budget(
    ordered: Sequence[Candidate], fraction: float, totals: tuple[int, int]
) -> float:
    total_tokens, total_claims = totals
    if total_claims == 0:
        return 0.0
    ceiling = total_tokens * fraction
    spent = recovered = 0
    for candidate in ordered:
        if spent + candidate.tokens > ceiling:
            break
        spent += candidate.tokens
        recovered += candidate.claims
    return 100.0 * recovered / total_claims


def _budget_for_recall(
    ordered: Sequence[Candidate], fraction: float, totals: tuple[int, int]
) -> float:
    total_tokens, total_claims = totals
    if total_claims == 0 or total_tokens == 0:
        return 100.0
    spent = recovered = 0
    for candidate in ordered:
        spent += candidate.tokens
        recovered += candidate.claims
        if recovered >= total_claims * fraction:
            break
    else:  # pragma: no cover - unreachable, the full set always reaches any fraction <= 1
        pass
    return 100.0 * spent / total_tokens


def _point(label: str, ordered: Sequence[Candidate], totals: tuple[int, int]) -> GainPoint:
    return GainPoint(
        label=label,
        claims_at_10_percent=_claims_at_budget(ordered, 0.10, totals),
        claims_at_25_percent=_claims_at_budget(ordered, 0.25, totals),
        budget_for_90_percent=_budget_for_recall(ordered, 0.90, totals),
        budget_for_95_percent=_budget_for_recall(ordered, 0.95, totals),
    )


def gain_points(candidates: Sequence[Candidate], seed: int) -> tuple[GainPoint, ...]:
    """What a ranking buys per token spent, which is what AUC cannot see: AUC
    is unweighted and unpriced, so a gate can score 0.963 while moving almost
    no value forward (issue #151). The random control is returned
    unconditionally, because the gate's number is uninterpretable without it."""
    if not candidates:
        return ()
    totals = (
        sum(candidate.tokens for candidate in candidates),
        sum(candidate.claims for candidate in candidates),
    )
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    orderings: tuple[tuple[str, Callable[[Candidate], float] | None], ...] = (
        ("p_lore", lambda candidate: -candidate.p_lore),
        (ORACLE, lambda candidate: -candidate.claims / max(candidate.tokens, 1)),
        (RANDOM, None),
    )
    return tuple(
        _point(label, shuffled if key is None else sorted(candidates, key=key), totals)
        for label, key in orderings
    )


def compute_gain_curve(
    conn: sqlite3.Connection,
    mode: str,
    seed: int = 0,
    min_input_tokens: int = MIN_PLAUSIBLE_INPUT_TOKENS,
) -> GainReport:
    rows = conn.execute(
        "SELECT e.id AS exchange_id, e.p_lore AS p_lore,"
        " r.input_tokens + r.output_tokens AS tokens,"
        " (SELECT COUNT(*) FROM claims c WHERE c.extraction_run_id = r.id) AS claims"
        ", r.input_tokens AS input_tokens"
        " FROM extraction_runs r JOIN exchanges e ON e.id = r.exchange_id"
        " WHERE r.outcome = 'ok' AND r.mode = ?"
        "   AND r.input_tokens IS NOT NULL AND r.output_tokens IS NOT NULL"
        "   AND e.p_lore IS NOT NULL",
        (mode,),
    ).fetchall()
    credible = [row for row in rows if row["input_tokens"] >= min_input_tokens]
    candidates = [
        Candidate(
            exchange_id=row["exchange_id"],
            tokens=row["tokens"],
            claims=row["claims"],
            p_lore=row["p_lore"],
        )
        for row in credible
    ]
    return GainReport(
        mode=mode,
        exchanges=len(candidates),
        total_tokens=sum(candidate.tokens for candidate in candidates),
        total_claims=sum(candidate.claims for candidate in candidates),
        points=gain_points(candidates, seed),
        excluded_implausible=len(rows) - len(credible),
        min_input_tokens=min_input_tokens,
    )


def format_gain_report(report: GainReport) -> list[str]:
    if not report.points:
        return [
            f"gain curve, {report.mode} runs: no scored runs with recorded tokens yet,"
            " so there is nothing to rank"
        ]
    excluded = (
        f", {report.excluded_implausible} excluded as billed under"
        f" {report.min_input_tokens} input tokens"
        if report.excluded_implausible
        else ""
    )
    lines = [
        f"gain curve, {report.mode} runs: {report.exchanges} exchanges,"
        f" {report.total_claims} claims, {report.total_tokens} tokens{excluded}",
        f"  {'ordering':<30} {'@10%':>7} {'@25%':>7} {'for 90%':>9} {'for 95%':>9}",
    ]
    for point in report.points:
        lines.append(
            f"  {point.label:<30} {point.claims_at_10_percent:>6.1f}%"
            f" {point.claims_at_25_percent:>6.1f}%"
            f" {point.budget_for_90_percent:>8.1f}%"
            f" {point.budget_for_95_percent:>8.1f}%"
        )
    return lines
