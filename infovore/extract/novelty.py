import asyncio
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field

from infovore.db.claims import (
    claims_for_runs_needing_probe,
    claims_needing_probe,
    set_novelty,
    set_probe_error,
    unprobed_claims,
)
from infovore.extract.protocol import FailureKind, NoveltyProbe
from infovore.rows import ClaimRow, Novelty
from infovore.timing import Clock, Sleeper

DEFAULT_USAGE_LIMIT_RETRY_SECONDS = 300.0


@dataclass(frozen=True)
class ProbeReport:
    probed: int = 0
    by_verdict: dict[Novelty, int] = field(default_factory=dict)
    failed: int = 0
    pauses: int = 0


def _fetch_candidates(
    conn: sqlite3.Connection,
    probe_model: str | None,
    limit: int,
    run_ids: Sequence[int] | None,
) -> list[ClaimRow]:
    if run_ids is not None:
        return claims_for_runs_needing_probe(conn, run_ids, probe_model, limit)
    if probe_model is not None:
        return claims_needing_probe(conn, probe_model, limit)
    return unprobed_claims(conn, limit)


async def run_probe(
    conn: sqlite3.Connection,
    probe: NoveltyProbe,
    clock: Clock,
    sleeper: Sleeper,
    *,
    probe_model: str | None,
    limit: int,
    concurrency: int,
    run_ids: Sequence[int] | None = None,
) -> ProbeReport:
    probed = 0
    failed = 0
    pauses = 0
    by_verdict: dict[Novelty, int] = {}
    gave_up: set[int] = set()
    semaphore = asyncio.Semaphore(concurrency)

    async def handle(claim: ClaimRow) -> None:
        nonlocal probed, failed, pauses
        assert claim.id is not None
        claim_id = claim.id
        async with semaphore:
            while True:
                outcome = await probe.probe(claim)
                if outcome.succeeded:
                    assert outcome.verdict is not None
                    assert outcome.model is not None
                    assert outcome.answer is not None
                    set_novelty(
                        conn, claim_id, outcome.verdict, outcome.model, outcome.answer, clock.now()
                    )
                    probed += 1
                    by_verdict[outcome.verdict] = by_verdict.get(outcome.verdict, 0) + 1
                    return
                failure = outcome.failure
                assert failure is not None
                if failure.kind is FailureKind.USAGE_LIMIT:
                    pauses += 1
                    await sleeper.sleep(failure.retry_after or DEFAULT_USAGE_LIMIT_RETRY_SECONDS)
                    continue
                set_probe_error(conn, claim_id, failure.message)
                failed += 1
                gave_up.add(claim_id)
                return

    while True:
        candidates = [
            claim
            for claim in _fetch_candidates(conn, probe_model, limit, run_ids)
            if claim.id not in gave_up
        ]
        if not candidates:
            break
        await asyncio.gather(*(handle(claim) for claim in candidates))

    return ProbeReport(probed=probed, by_verdict=by_verdict, failed=failed, pauses=pauses)
