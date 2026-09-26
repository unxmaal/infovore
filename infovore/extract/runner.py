import asyncio
import random
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field

from infovore.db.claims import NewClaim, record_run, register_prompt_version
from infovore.db.claims import live_prompt_version as db_live_prompt_version
from infovore.db.claims import retract_claim as db_retract_claim
from infovore.db.exchanges import claimable_exchanges, get_exchange, set_status
from infovore.db.exchanges import record_failure as db_record_failure
from infovore.extract.prompt import PROMPT_SHA256, PROMPT_VERSION, permalink
from infovore.extract.protocol import ClaimExtractor, ExtractedClaim, Failure, FailureKind
from infovore.extract.request import build_request
from infovore.rows import ExchangeRow, ExtractionRunRow, ExtractionStatus, RunMode, RunOutcome
from infovore.timing import Clock, Sleeper

DEFAULT_USAGE_LIMIT_RETRY_AFTER = 300.0


class PromptNotPromotedError(Exception):
    def __init__(self, version: str) -> None:
        super().__init__(version)
        self.version = version


@dataclass(frozen=True)
class ExtractionReport:
    processed: int
    succeeded: int
    failed: int
    skipped: int
    claims_recorded: int
    pauses: int
    run_ids: tuple[int, ...]


@dataclass
class _Accumulator:
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    claims_recorded: int = 0
    pauses: int = 0
    run_ids: list[int] = field(default_factory=list)

    def to_report(self) -> ExtractionReport:
        return ExtractionReport(
            processed=self.processed,
            succeeded=self.succeeded,
            failed=self.failed,
            skipped=self.skipped,
            claims_recorded=self.claims_recorded,
            pauses=self.pauses,
            run_ids=tuple(self.run_ids),
        )


def _size_bucket(message_count: int) -> str:
    if message_count == 1:
        return "1"
    if message_count <= 5:
        return "2-5"
    if message_count <= 20:
        return "6-20"
    return "21+"


def select_trial_sample(conn: sqlite3.Connection, n: int, seed: int) -> list[int]:
    rows = conn.execute(
        "SELECT id, channel_id, message_count FROM exchanges ORDER BY id"
    ).fetchall()
    if n >= len(rows):
        return [row["id"] for row in rows]

    strata: dict[tuple[int, str], list[int]] = {}
    for row in rows:
        key = (row["channel_id"], _size_bucket(row["message_count"]))
        strata.setdefault(key, []).append(row["id"])

    keys = sorted(strata)
    allocation = {key: 0 for key in keys}
    remaining = n
    index = 0
    while remaining > 0:
        key = keys[index % len(keys)]
        if allocation[key] < len(strata[key]):
            allocation[key] += 1
            remaining -= 1
        index += 1

    rng = random.Random(seed)
    selected: list[int] = []
    for key in keys:
        selected.extend(rng.sample(strata[key], allocation[key]))
    return sorted(selected)


def _previous_live_claim_ids(conn: sqlite3.Connection, exchange_id: int) -> list[int]:
    rows = conn.execute(
        "SELECT c.id FROM claims c JOIN extraction_runs r ON r.id = c.extraction_run_id"
        " WHERE c.exchange_id = ? AND r.mode = ? AND c.retracted_at IS NULL",
        (exchange_id, RunMode.LIVE.value),
    ).fetchall()
    return [row["id"] for row in rows]


def _finish_success(
    conn: sqlite3.Connection,
    clock: Clock,
    mode: RunMode,
    model_label: str,
    exchange: ExchangeRow,
    request_guild_id: int,
    outcome_model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    claims: Sequence[ExtractedClaim],
    accumulator: _Accumulator,
) -> None:
    assert exchange.id is not None
    now = clock.now()
    link = permalink(request_guild_id, exchange.channel_id, exchange.first_message_id)
    new_claims = [
        NewClaim(
            exchange_id=exchange.id,
            statement=claim.statement,
            subject=claim.subject,
            kind=claim.kind,
            confidence=claim.confidence,
            probe_question=claim.probe_question,
            permalink=link,
            supersedes_claim_id=claim.supersedes_claim_id,
            source_message_ids=claim.source_message_ids,
        )
        for claim in claims
    ]
    run_row = ExtractionRunRow(
        id=None,
        exchange_id=exchange.id,
        model=outcome_model or model_label,
        prompt_version=PROMPT_VERSION,
        started_at=now,
        finished_at=now,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        mode=mode,
        outcome=RunOutcome.OK,
        error=None,
    )

    previous_claim_ids: list[int] = []
    if mode is RunMode.LIVE and exchange.extraction_status is ExtractionStatus.STALE:
        previous_claim_ids = _previous_live_claim_ids(conn, exchange.id)

    recorded = record_run(conn, run_row, new_claims)
    accumulator.run_ids.append(recorded.run_id)
    accumulator.claims_recorded += len(recorded.claim_ids)
    accumulator.succeeded += 1

    if mode is RunMode.LIVE:
        for claim_id in previous_claim_ids:
            db_retract_claim(conn, claim_id, "reextracted", now)
        set_status(conn, exchange.id, ExtractionStatus.DONE)


def _finish_failure(
    conn: sqlite3.Connection,
    clock: Clock,
    mode: RunMode,
    model_label: str,
    max_retries: int,
    exchange: ExchangeRow,
    failure: Failure,
    input_tokens: int | None,
    output_tokens: int | None,
    accumulator: _Accumulator,
) -> None:
    assert exchange.id is not None
    now = clock.now()
    run_row = ExtractionRunRow(
        id=None,
        exchange_id=exchange.id,
        model=model_label,
        prompt_version=PROMPT_VERSION,
        started_at=now,
        finished_at=now,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        mode=mode,
        outcome=RunOutcome.FAILED,
        error=failure.message,
    )
    recorded = record_run(conn, run_row, [])
    accumulator.run_ids.append(recorded.run_id)
    accumulator.failed += 1

    if mode is RunMode.LIVE:
        db_record_failure(conn, exchange.id, failure.message, max_retries)


async def _handle_exchange(
    conn: sqlite3.Connection,
    extractor: ClaimExtractor,
    clock: Clock,
    sleeper: Sleeper,
    semaphore: asyncio.Semaphore,
    mode: RunMode,
    model_label: str,
    max_retries: int,
    exchange: ExchangeRow,
    accumulator: _Accumulator,
) -> None:
    async with semaphore:
        request = build_request(conn, exchange)
        accumulator.processed += 1

        if mode is RunMode.LIVE and all(
            message.author_id in request.opted_out_user_ids for message in request.messages
        ):
            assert exchange.id is not None
            set_status(conn, exchange.id, ExtractionStatus.SKIPPED)
            accumulator.skipped += 1
            return

        guild_id = request.messages[0].guild_id
        while True:
            outcome = await extractor.extract(request)
            if outcome.succeeded:
                _finish_success(
                    conn,
                    clock,
                    mode,
                    model_label,
                    exchange,
                    guild_id,
                    outcome.model,
                    outcome.input_tokens,
                    outcome.output_tokens,
                    outcome.claims,
                    accumulator,
                )
                return

            failure = outcome.failure
            assert failure is not None
            if failure.kind is FailureKind.USAGE_LIMIT:
                accumulator.pauses += 1
                await sleeper.sleep(failure.retry_after or DEFAULT_USAGE_LIMIT_RETRY_AFTER)
                continue

            _finish_failure(
                conn,
                clock,
                mode,
                model_label,
                max_retries,
                exchange,
                failure,
                outcome.input_tokens,
                outcome.output_tokens,
                accumulator,
            )
            return


async def _process_batch(
    conn: sqlite3.Connection,
    extractor: ClaimExtractor,
    clock: Clock,
    sleeper: Sleeper,
    semaphore: asyncio.Semaphore,
    mode: RunMode,
    model_label: str,
    max_retries: int,
    batch: Sequence[ExchangeRow],
    accumulator: _Accumulator,
) -> None:
    await asyncio.gather(
        *(
            _handle_exchange(
                conn,
                extractor,
                clock,
                sleeper,
                semaphore,
                mode,
                model_label,
                max_retries,
                exchange,
                accumulator,
            )
            for exchange in batch
        )
    )


async def run_extraction(
    conn: sqlite3.Connection,
    extractor: ClaimExtractor,
    clock: Clock,
    sleeper: Sleeper,
    *,
    mode: RunMode,
    model_label: str,
    batch_size: int,
    max_retries: int,
    concurrency: int,
    exchange_ids: Sequence[int] | None = None,
) -> ExtractionReport:
    register_prompt_version(conn, PROMPT_VERSION, PROMPT_SHA256, clock.now())
    if mode is RunMode.LIVE and db_live_prompt_version(conn) != PROMPT_VERSION:
        raise PromptNotPromotedError(PROMPT_VERSION)

    accumulator = _Accumulator()
    semaphore = asyncio.Semaphore(concurrency)

    if mode is RunMode.LIVE:
        while True:
            batch = claimable_exchanges(conn, batch_size, max_retries)
            if not batch:
                break
            await _process_batch(
                conn,
                extractor,
                clock,
                sleeper,
                semaphore,
                mode,
                model_label,
                max_retries,
                batch,
                accumulator,
            )
    else:
        batch = [
            exchange
            for exchange in (get_exchange(conn, id) for id in exchange_ids or ())
            if exchange is not None
        ]
        await _process_batch(
            conn,
            extractor,
            clock,
            sleeper,
            semaphore,
            mode,
            model_label,
            max_retries,
            batch,
            accumulator,
        )

    return accumulator.to_report()
