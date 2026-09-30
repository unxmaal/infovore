"""Re-probe a sample and diff it against the stored verdicts (see README)."""

import asyncio
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field

from infovore.db.claims import record_probe_run
from infovore.extract.probe_runs import probe_run_row
from infovore.extract.protocol import (
    BatchNoveltyProbe,
    NoveltyProbe,
    ProbeOutcome,
    ProbeUsage,
)
from infovore.rows import ClaimRow, Novelty
from infovore.timing import Clock

DEFAULT_COMPARE_LIMIT = 50


@dataclass(frozen=True)
class ComparisonReport:
    compared: int = 0
    agreed: int = 0
    failed: int = 0
    calls: int = 0
    cost_usd: float | None = None
    confusion: dict[tuple[Novelty, Novelty], int] = field(default_factory=dict)

    @property
    def agreement_rate(self) -> float | None:
        if self.compared == 0:
            return None
        return self.agreed / self.compared

    @property
    def known_to_unknown(self) -> int:
        return self.confusion.get((Novelty.KNOWN, Novelty.UNKNOWN), 0)


def _sample(conn: sqlite3.Connection, limit: int) -> list[ClaimRow]:
    from infovore.db.claims import _row_to_claim

    rows = conn.execute(
        "SELECT * FROM claims WHERE novelty != 'unprobed' AND probe_error IS NULL"
        " AND retracted_at IS NULL ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_claim(row) for row in rows]


async def run_comparison(
    conn: sqlite3.Connection,
    probe: NoveltyProbe,
    clock: Clock,
    *,
    limit: int = DEFAULT_COMPARE_LIMIT,
    batch_size: int = 1,
    concurrency: int = 1,
) -> ComparisonReport:
    claims = _sample(conn, limit)
    if not claims:
        return ComparisonReport()

    confusion: dict[tuple[Novelty, Novelty], int] = {}
    agreed = 0
    compared = 0
    failed = 0
    calls = 0
    cost: float | None = None
    semaphore = asyncio.Semaphore(concurrency)
    batching = batch_size > 1 and isinstance(probe, BatchNoveltyProbe)

    def record(claim: ClaimRow, outcome: ProbeOutcome) -> None:
        nonlocal agreed, compared, failed
        if not outcome.succeeded:
            failed += 1
            return
        assert outcome.verdict is not None
        pair = (claim.novelty, outcome.verdict)
        confusion[pair] = confusion.get(pair, 0) + 1
        compared += 1
        if claim.novelty is outcome.verdict:
            agreed += 1

    def charge(usage: ProbeUsage, count: int, batched: bool, error: str | None) -> None:
        nonlocal calls, cost
        calls += 1
        record_probe_run(
            conn,
            probe_run_row(
                usage,
                started_at=clock.now(),
                finished_at=clock.now(),
                claim_count=count,
                batched=batched,
                error=error,
            ),
        )
        if usage.cost_usd is not None:
            cost = (cost or 0.0) + usage.cost_usd

    async def one(claim: ClaimRow) -> None:
        async with semaphore:
            outcome = await probe.probe(claim)
            failure = outcome.failure
            charge(outcome.usage, 1, False, failure.message if failure else None)
            record(claim, outcome)

    async def group(batch: Sequence[ClaimRow]) -> None:
        nonlocal failed
        async with semaphore:
            assert isinstance(probe, BatchNoveltyProbe)
            result = await probe.probe_batch(batch)
            failure = result.failure
            charge(result.usage, len(batch), True, failure.message if failure else None)
            if failure is not None:
                # A comparison is a one-shot diagnostic: a failed batch is
                # reported, not retried, so the numbers describe one pass.
                failed += len(batch)
                return
            assert result.outcomes is not None
            for claim, outcome in zip(batch, result.outcomes, strict=True):
                record(claim, outcome)

    if batching:
        groups = [claims[start : start + batch_size] for start in range(0, len(claims), batch_size)]
        # A group of one is a single probe: batching it would send the batch
        # prompt for one question and compare against a different code path
        # from the one it is standing in for.
        await asyncio.gather(
            *(group(batch) if len(batch) > 1 else one(batch[0]) for batch in groups)
        )
    else:
        await asyncio.gather(*(one(claim) for claim in claims))

    return ComparisonReport(
        compared=compared,
        agreed=agreed,
        failed=failed,
        calls=calls,
        cost_usd=cost,
        confusion=confusion,
    )
