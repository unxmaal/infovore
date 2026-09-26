import argparse
import asyncio
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from infovore.cli import ExitCode, _say, stage_backend
from infovore.config import Stage
from infovore.db.claims import (
    claims_for_runs_needing_probe,
    claims_needing_probe,
    set_novelty,
    set_probe_error,
    unprobed_claims,
)
from infovore.extract.llm_extractor import LLMNoveltyProbe
from infovore.extract.protocol import FailureKind, NoveltyProbe
from infovore.rows import ClaimRow, Novelty
from infovore.timing import Clock, Sleeper

if TYPE_CHECKING:
    from infovore.cli import AppContext

DEFAULT_USAGE_LIMIT_RETRY_SECONDS = 300.0


@dataclass(frozen=True)
class ProbeReport:
    probed: int = 0
    by_verdict: dict[Novelty, int] = field(default_factory=dict)
    failed: int = 0
    pauses: int = 0


@dataclass(frozen=True)
class ProbeStarted:
    candidates: int


@dataclass(frozen=True)
class ClaimProbed:
    claim_id: int
    verdict: str


@dataclass(frozen=True)
class ClaimFailed:
    claim_id: int


@dataclass(frozen=True)
class ClaimPaused:
    claim_id: int


ProbeEvent = ProbeStarted | ClaimProbed | ClaimFailed | ClaimPaused
ProbeProgress = Callable[[ProbeEvent], None]


def _ignore_probe_progress(event: ProbeEvent) -> None:
    return None


def _fetch_candidates(
    conn: sqlite3.Connection,
    probe_model: str | None,
    limit: int,
    run_ids: Sequence[int] | None,
    include_failed: bool,
) -> list[ClaimRow]:
    if run_ids is not None:
        return claims_for_runs_needing_probe(
            conn, run_ids, probe_model, limit, include_failed=include_failed
        )
    if probe_model is not None:
        return claims_needing_probe(conn, probe_model, limit, include_failed=include_failed)
    return unprobed_claims(conn, limit, include_failed=include_failed)


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
    retry_failed: bool = False,
    progress: ProbeProgress = _ignore_probe_progress,
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
                    progress(ClaimProbed(claim_id=claim_id, verdict=outcome.verdict.value))
                    return
                failure = outcome.failure
                assert failure is not None
                if failure.kind is FailureKind.USAGE_LIMIT:
                    pauses += 1
                    progress(ClaimPaused(claim_id=claim_id))
                    await sleeper.sleep(failure.retry_after or DEFAULT_USAGE_LIMIT_RETRY_SECONDS)
                    continue
                set_probe_error(conn, claim_id, failure.message)
                failed += 1
                gave_up.add(claim_id)
                progress(ClaimFailed(claim_id=claim_id))
                return

    started = False
    while True:
        candidates = [
            claim
            for claim in _fetch_candidates(conn, probe_model, limit, run_ids, retry_failed)
            if claim.id not in gave_up
        ]
        if not started:
            progress(ProbeStarted(candidates=len(candidates)))
            started = True
        if not candidates:
            break
        await asyncio.gather(*(handle(claim) for claim in candidates))

    return ProbeReport(probed=probed, by_verdict=by_verdict, failed=failed, pauses=pauses)


def _format_by_verdict(by_verdict: dict[Novelty, int]) -> str:
    return " ".join(f"{verdict.value}={count}" for verdict, count in by_verdict.items()) or "none"


def _describe_probe_event(event: ProbeEvent) -> str:
    match event:
        case ProbeStarted(candidates=candidates):
            return f"probe: {candidates} candidates"
        case ClaimProbed(claim_id=claim_id, verdict=verdict):
            return f"claim {claim_id}: {verdict}"
        case ClaimFailed(claim_id=claim_id):
            return f"claim {claim_id}: failed"
        case _:
            return f"claim {event.claim_id}: paused"


class ProbeCommand:
    name = "probe"
    help = "closed-book novelty probe over unprobed claims"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--run-id", type=int, action="append", dest="run_ids", default=None)
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--probe-model", type=str, default=None)
        parser.add_argument("--retry-failed", action="store_true")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        probe_backend = await stage_backend(context, Stage.PROBE)
        judge_backend = await stage_backend(context, Stage.JUDGE)
        probe = LLMNoveltyProbe(probe_backend, judge_backend)
        limit = args.limit if args.limit is not None else context.settings.batch_size
        concurrency = context.settings.stages[Stage.PROBE].concurrency
        run_ids = tuple(args.run_ids) if args.run_ids else None

        report = await run_probe(
            context.conn,
            probe,
            context.clock,
            context.sleeper,
            probe_model=args.probe_model,
            limit=limit,
            concurrency=concurrency,
            run_ids=run_ids,
            retry_failed=args.retry_failed,
            progress=lambda event: _say(context.stdout, _describe_probe_event(event)),
        )
        context.stdout.write(
            f"probed: {report.probed}\n"
            f"by verdict: {_format_by_verdict(report.by_verdict)}\n"
            f"failed: {report.failed}\n"
            f"pauses: {report.pauses}\n"
        )
        return ExitCode.FAILURE if report.failed else ExitCode.OK
