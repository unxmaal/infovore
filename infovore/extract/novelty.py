import argparse
import asyncio
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from infovore.cli import ExitCode, _say, stage_backend
from infovore.config import DEFAULT_PROBE_BATCH_SIZE, ConfigError, Stage
from infovore.db.claims import (
    claims_for_runs_needing_probe,
    claims_needing_probe,
    set_novelty,
    set_probe_error,
    unprobed_claims,
)
from infovore.db.run_selection import (
    InvalidRunSelectorError,
    NoTrialBatchError,
    resolve_run_selector,
)
from infovore.extract.llm_extractor import BatchedLLMNoveltyProbe
from infovore.extract.protocol import (
    BatchNoveltyProbe,
    Failure,
    FailureKind,
    NoveltyProbe,
    ProbeOutcome,
)
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


def _interleave_by_exchange(claims: Sequence[ClaimRow]) -> list[ClaimRow]:
    """Reorder so consecutive claims come from different exchanges.

    Batched probing puts several probe questions in one call, and two claims
    from the same conversation can hint at each other's answers (issue #126).
    Dealing one claim per exchange per pass keeps same-exchange claims apart
    for as long as there are other exchanges left to deal from.
    """
    by_exchange: dict[int, list[ClaimRow]] = {}
    for claim in claims:
        by_exchange.setdefault(claim.exchange_id, []).append(claim)
    queues = list(by_exchange.values())
    ordered: list[ClaimRow] = []
    while queues:
        remaining: list[list[ClaimRow]] = []
        for queue in queues:
            ordered.append(queue.pop(0))
            if queue:
                remaining.append(queue)
        queues = remaining
    return ordered


def _batches(claims: Sequence[ClaimRow], size: int) -> list[list[ClaimRow]]:
    if size <= 1:
        return [[claim] for claim in claims]
    ordered = _interleave_by_exchange(claims)
    return [ordered[start : start + size] for start in range(0, len(ordered), size)]


def _retry_after(failure: Failure) -> float:
    if failure.retry_after is None:
        return DEFAULT_USAGE_LIMIT_RETRY_SECONDS
    return failure.retry_after


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
    batch_size: int = 1,
    progress: ProbeProgress = _ignore_probe_progress,
) -> ProbeReport:
    probed = 0
    failed = 0
    pauses = 0
    by_verdict: dict[Novelty, int] = {}
    gave_up: set[int] = set()
    semaphore = asyncio.Semaphore(concurrency)
    batching = batch_size > 1 and isinstance(probe, BatchNoveltyProbe)

    def record_success(claim_id: int, outcome: ProbeOutcome) -> None:
        nonlocal probed
        assert outcome.verdict is not None
        assert outcome.model is not None
        assert outcome.answer is not None
        set_novelty(conn, claim_id, outcome.verdict, outcome.model, outcome.answer, clock.now())
        probed += 1
        by_verdict[outcome.verdict] = by_verdict.get(outcome.verdict, 0) + 1
        progress(ClaimProbed(claim_id=claim_id, verdict=outcome.verdict.value))

    def record_failure(claim_id: int, failure: Failure) -> None:
        nonlocal failed
        set_probe_error(conn, claim_id, failure.message)
        failed += 1
        gave_up.add(claim_id)
        progress(ClaimFailed(claim_id=claim_id))

    async def probe_one(claim: ClaimRow) -> None:
        nonlocal pauses
        assert claim.id is not None
        claim_id = claim.id
        while True:
            outcome = await probe.probe(claim)
            if outcome.succeeded:
                record_success(claim_id, outcome)
                return
            failure = outcome.failure
            assert failure is not None
            if failure.kind is FailureKind.USAGE_LIMIT:
                pauses += 1
                progress(ClaimPaused(claim_id=claim_id))
                await sleeper.sleep(_retry_after(failure))
                continue
            record_failure(claim_id, failure)
            return

    async def probe_group(batch: Sequence[ClaimRow]) -> None:
        nonlocal pauses
        assert isinstance(probe, BatchNoveltyProbe)
        while True:
            result = await probe.probe_batch(batch)
            if result.succeeded:
                assert result.outcomes is not None
                for claim, outcome in zip(batch, result.outcomes, strict=True):
                    assert claim.id is not None
                    record_success(claim.id, outcome)
                return
            failure = result.failure
            assert failure is not None
            if failure.kind is FailureKind.USAGE_LIMIT:
                pauses += 1
                for claim in batch:
                    assert claim.id is not None
                    progress(ClaimPaused(claim_id=claim.id))
                await sleeper.sleep(_retry_after(failure))
                continue
            # The batch as a whole could not be mapped onto its claims. Re-probe
            # singly so each claim gets its own verdict or its own error, rather
            # than charging one call's failure to all of them.
            for claim in batch:
                await probe_one(claim)
            return

    async def handle(batch: Sequence[ClaimRow]) -> None:
        async with semaphore:
            if batching and len(batch) > 1:
                await probe_group(batch)
                return
            for claim in batch:
                await probe_one(claim)

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
        groups = _batches(candidates, batch_size) if batching else [[c] for c in candidates]
        await asyncio.gather(*(handle(group) for group in groups))

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
        parser.add_argument(
            "--run-id", nargs="*", dest="run_ids", default=None, metavar="RUN_ID_OR_RANGE"
        )
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--probe-model", type=str, default=None)
        parser.add_argument("--retry-failed", action="store_true")
        parser.add_argument(
            "--batch-size",
            type=int,
            default=DEFAULT_PROBE_BATCH_SIZE,
            help="claims per pair of LLM calls; 1 probes one claim at a time",
        )

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        probe_backend = await stage_backend(context, Stage.PROBE)
        judge_backend = await stage_backend(context, Stage.JUDGE)
        probe = BatchedLLMNoveltyProbe(probe_backend, judge_backend)
        limit = args.limit if args.limit is not None else context.settings.batch_size
        concurrency = context.settings.stages[Stage.PROBE].concurrency

        run_ids: tuple[int, ...] | None = None
        if args.run_ids is not None:
            try:
                run_ids = tuple(resolve_run_selector(context.conn, args.run_ids))
            except NoTrialBatchError as error:
                raise ConfigError(
                    "no trial batch found; pass --run-id explicitly, or run"
                    " `infovore extract --mode trial` first"
                ) from error
            except InvalidRunSelectorError as error:
                raise ConfigError(str(error)) from error

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
            batch_size=args.batch_size,
            progress=lambda event: _say(context.stdout, _describe_probe_event(event)),
        )
        context.stdout.write(
            f"probed: {report.probed}\n"
            f"by verdict: {_format_by_verdict(report.by_verdict)}\n"
            f"failed: {report.failed}\n"
            f"pauses: {report.pauses}\n"
        )
        return ExitCode.FAILURE if report.failed else ExitCode.OK
