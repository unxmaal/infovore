import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.claims import (
    NewClaim,
    claim_source_ids,
    claims_for_run,
    get_claim,
    live_prompt_version,
    promote_prompt_version,
    record_run,
    register_prompt_version,
)
from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import get_exchange, insert_exchange
from infovore.extract.fake import MarkerExtractor
from infovore.extract.prompt import PROMPT_SHA256, PROMPT_VERSION, permalink
from infovore.extract.protocol import (
    ExtractionOutcome,
    ExtractionRequest,
    Failure,
    FailureKind,
)
from infovore.extract.runner import (
    ExchangeClaimed,
    ExchangeFailed,
    ExchangePaused,
    ExchangeSkipped,
    ExtractionEvent,
    ExtractionReport,
    ExtractionStarted,
    PromptNotPromotedError,
    UntriagedExchangesError,
    run_extraction,
    select_trial_sample,
)
from infovore.rows import (
    ClaimKind,
    ExchangeRow,
    ExtractionRunRow,
    ExtractionStatus,
    GroupingRule,
    MessageRow,
    RunMode,
    RunOutcome,
)
from infovore.timing import FixedClock, RecordingSleeper
from infovore.triage.score import TRIAGE_VERSION

GUILD_ID = 500
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def a_message(
    message_id: int,
    channel_id: int = 1,
    author_id: int = 1,
    content: str = "chatter",
    created_at: datetime = NOW,
) -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=channel_id,
        guild_id=GUILD_ID,
        author_id=author_id,
        author_name_at_time="alice",
        author_is_bot=False,
        created_at=created_at,
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        deleted_at=None,
        ingested_at=NOW,
        raw_json="{}",
    )


def seed_exchange(
    conn: sqlite3.Connection,
    messages: list[MessageRow],
    status: ExtractionStatus = ExtractionStatus.PENDING,
    retry_count: int = 0,
    parent_exchange_id: int | None = None,
    triage_score: float | None = 1.0,
    triage_version: str | None = TRIAGE_VERSION,
) -> ExchangeRow:
    for message in messages:
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " author_is_bot, created_at, content, ingested_at, raw_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                message.id,
                message.channel_id,
                message.guild_id,
                message.author_id,
                message.author_name_at_time,
                int(message.author_is_bot),
                message.created_at.isoformat(),
                message.content,
                message.ingested_at.isoformat(),
                message.raw_json,
            ),
        )
    first = messages[0].id
    last = messages[-1].id
    row = ExchangeRow(
        id=None,
        channel_id=messages[0].channel_id,
        thread_id=None,
        first_message_id=first,
        last_message_id=last,
        started_at=NOW,
        ended_at=NOW,
        message_count=len(messages),
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=f"hash-{first}-{last}-{status.value}-{retry_count}",
        parent_exchange_id=parent_exchange_id,
        extraction_status=status,
        retry_count=retry_count,
        last_error=None,
    )
    exchange_id = insert_exchange(conn, row, [m.id for m in messages])
    conn.execute(
        "UPDATE exchanges SET triage_score = ?, triage_reasons = ?, triage_version = ?"
        " WHERE id = ?",
        (
            triage_score,
            "[]" if triage_score is not None else None,
            triage_version,
            exchange_id,
        ),
    )
    exchange = get_exchange(conn, exchange_id)
    assert exchange is not None
    return exchange


def opt_out(conn: sqlite3.Connection, user_id: int) -> None:
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (?, ?)", (user_id, NOW.isoformat()))


class SequencedExtractor:
    def __init__(self, outcomes: list[ExtractionOutcome]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    async def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        return outcome


class ExplodingExtractor:
    async def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        raise AssertionError("extractor should not be called")


class ConcurrencyTrackingExtractor:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return ExtractionOutcome(
            claims=(), model="concurrent", input_tokens=None, output_tokens=None, failure=None
        )


async def promote(conn: sqlite3.Connection) -> None:
    register_prompt_version(conn, PROMPT_VERSION, PROMPT_SHA256, NOW)
    promote_prompt_version(conn, PROMPT_VERSION, NOW)


def test_live_requires_promotion(tmp_path: Path) -> None:
    conn = db(tmp_path)
    extractor = MarkerExtractor()

    async def go() -> None:
        with pytest.raises(PromptNotPromotedError):
            await run_extraction(
                conn,
                extractor,
                FixedClock(NOW),
                RecordingSleeper(),
                mode=RunMode.LIVE,
                model_label="model-x",
                batch_size=10,
                max_retries=3,
                concurrency=2,
            )

    asyncio.run(go())
    assert live_prompt_version(conn) is None


def test_success_records_claims_with_provenance_and_permalink(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = seed_exchange(conn, [a_message(1, content="FACT: Octane2 :: needs a jumper")])
    assert exchange.id is not None

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
        )

    report = asyncio.run(go())

    assert report.processed == 1
    assert report.succeeded == 1
    assert report.claims_recorded == 1
    assert len(report.run_ids) == 1
    claims = claims_for_run(conn, report.run_ids[0])
    assert len(claims) == 1
    claim = claims[0]
    assert claim.subject == "Octane2"
    assert claim.statement == "needs a jumper"
    assert claim.permalink == permalink(GUILD_ID, exchange.channel_id, exchange.first_message_id)
    assert claim.id is not None
    assert claim_source_ids(conn, claim.id) == [1]
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.extraction_status is ExtractionStatus.DONE


def test_zero_claim_exchange_is_done(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = seed_exchange(conn, [a_message(1, content="just chatting, nothing to see")])
    assert exchange.id is not None

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
        )

    report = asyncio.run(go())

    assert report.claims_recorded == 0
    assert report.succeeded == 1
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.extraction_status is ExtractionStatus.DONE


@pytest.mark.parametrize(
    "kind", [FailureKind.TRANSIENT, FailureKind.FATAL, FailureKind.INVALID_OUTPUT]
)
def test_each_failure_kind_records_failed_run_and_increments_retry(
    tmp_path: Path, kind: FailureKind
) -> None:
    conn = db(tmp_path)
    exchange = seed_exchange(conn, [a_message(1, content=f"FAIL: {kind.value}")])
    assert exchange.id is not None

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=1,
            concurrency=2,
        )

    report = asyncio.run(go())

    assert report.failed == 1
    assert report.succeeded == 0
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.extraction_status is ExtractionStatus.FAILED
    assert updated.retry_count == 1
    assert updated.last_error is not None
    row = conn.execute(
        "SELECT model, outcome, error FROM extraction_runs WHERE id = ?", (report.run_ids[0],)
    ).fetchone()
    assert row["model"] == "model-x"
    assert row["outcome"] == RunOutcome.FAILED.value
    assert kind.value in row["error"]


def test_usage_limit_pauses_then_succeeds(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, [a_message(1)])
    extractor = SequencedExtractor(
        [
            ExtractionOutcome(
                claims=(),
                model=None,
                input_tokens=None,
                output_tokens=None,
                failure=Failure(FailureKind.USAGE_LIMIT, "slow down", 45.0),
            ),
            ExtractionOutcome(
                claims=(), model="model-y", input_tokens=1, output_tokens=1, failure=None
            ),
        ]
    )
    sleeper = RecordingSleeper()

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            extractor,
            FixedClock(NOW),
            sleeper,
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
        )

    report = asyncio.run(go())

    assert extractor.calls == 2
    assert sleeper.slept == [45.0]
    assert report.pauses == 1
    assert report.succeeded == 1
    assert report.failed == 0


def test_usage_limit_defaults_retry_after_to_300(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, [a_message(1)])
    extractor = SequencedExtractor(
        [
            ExtractionOutcome(
                claims=(),
                model=None,
                input_tokens=None,
                output_tokens=None,
                failure=Failure(FailureKind.USAGE_LIMIT, "slow down", None),
            ),
            ExtractionOutcome(
                claims=(), model="model-y", input_tokens=None, output_tokens=None, failure=None
            ),
        ]
    )
    sleeper = RecordingSleeper()

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            extractor,
            FixedClock(NOW),
            sleeper,
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
        )

    asyncio.run(go())

    assert sleeper.slept == [300.0]


def test_usage_limit_zero_retry_after_sleeps_zero_not_default(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, [a_message(1)])
    extractor = SequencedExtractor(
        [
            ExtractionOutcome(
                claims=(),
                model=None,
                input_tokens=None,
                output_tokens=None,
                failure=Failure(FailureKind.USAGE_LIMIT, "slow down", 0.0),
            ),
            ExtractionOutcome(
                claims=(), model="model-y", input_tokens=None, output_tokens=None, failure=None
            ),
        ]
    )
    sleeper = RecordingSleeper()
    events: list[ExtractionEvent] = []

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            extractor,
            FixedClock(NOW),
            sleeper,
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
            progress=events.append,
        )

    asyncio.run(go())

    assert sleeper.slept == [0.0]
    paused = events[1]
    assert isinstance(paused, ExchangePaused)
    assert paused.retry_after == 0.0


def test_retries_until_failed_after_max_retries(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = seed_exchange(conn, [a_message(1, content="FAIL: transient")])
    assert exchange.id is not None

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=2,
            concurrency=2,
        )

    report = asyncio.run(go())

    assert report.processed == 2
    assert report.failed == 2
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.extraction_status is ExtractionStatus.FAILED
    assert updated.retry_count == 2


def test_stale_reextraction_retracts_previous_live_claims(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = seed_exchange(
        conn, [a_message(1, content="FACT: Octane2 :: new fact")], status=ExtractionStatus.STALE
    )
    assert exchange.id is not None
    register_prompt_version(conn, PROMPT_VERSION, PROMPT_SHA256, NOW)
    recorded = record_run(
        conn,
        ExtractionRunRow(
            id=None,
            exchange_id=exchange.id,
            model="old-model",
            prompt_version=PROMPT_VERSION,
            started_at=NOW,
            finished_at=NOW,
            input_tokens=1,
            output_tokens=1,
            mode=RunMode.LIVE,
            outcome=RunOutcome.OK,
            error=None,
        ),
        [
            NewClaim(
                exchange_id=exchange.id,
                statement="old fact",
                subject="Octane2",
                kind=ClaimKind.FACT,
                confidence=0.9,
                probe_question="what?",
                permalink="https://discord.com/channels/1/1/1",
                supersedes_claim_id=None,
                source_message_ids=(1,),
            )
        ],
    )
    old_claim_id = recorded.claim_ids[0]

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
        )

    report = asyncio.run(go())

    old_claim = get_claim(conn, old_claim_id)
    assert old_claim is not None
    assert old_claim.retracted_at is not None
    assert old_claim.retraction_reason == "reextracted"
    new_claims = claims_for_run(conn, report.run_ids[0])
    assert len(new_claims) == 1
    assert new_claims[0].retracted_at is None
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.extraction_status is ExtractionStatus.DONE


def test_trial_mode_success_never_mutates_status_or_retries(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = seed_exchange(conn, [a_message(1, content="FACT: Octane2 :: a fact")])
    assert exchange.id is not None
    exchange_id = exchange.id

    async def go() -> ExtractionReport:
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.TRIAL,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
            exchange_ids=[exchange_id],
        )

    report = asyncio.run(go())

    assert report.claims_recorded == 1
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.extraction_status is ExtractionStatus.PENDING
    assert updated.retry_count == 0


def test_trial_mode_failure_never_mutates_status_or_retries(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = seed_exchange(conn, [a_message(1, content="FAIL: transient")])
    assert exchange.id is not None
    exchange_id = exchange.id

    async def go() -> ExtractionReport:
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.TRIAL,
            model_label="model-x",
            batch_size=10,
            max_retries=1,
            concurrency=2,
            exchange_ids=[exchange_id],
        )

    report = asyncio.run(go())

    assert report.failed == 1
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.extraction_status is ExtractionStatus.PENDING
    assert updated.retry_count == 0


def test_trial_mode_never_retracts_stale_exchange_claims(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = seed_exchange(
        conn, [a_message(1, content="FACT: Octane2 :: new fact")], status=ExtractionStatus.STALE
    )
    assert exchange.id is not None
    exchange_id = exchange.id
    register_prompt_version(conn, PROMPT_VERSION, PROMPT_SHA256, NOW)
    recorded = record_run(
        conn,
        ExtractionRunRow(
            id=None,
            exchange_id=exchange.id,
            model="old-model",
            prompt_version=PROMPT_VERSION,
            started_at=NOW,
            finished_at=NOW,
            input_tokens=1,
            output_tokens=1,
            mode=RunMode.LIVE,
            outcome=RunOutcome.OK,
            error=None,
        ),
        [
            NewClaim(
                exchange_id=exchange.id,
                statement="old fact",
                subject="Octane2",
                kind=ClaimKind.FACT,
                confidence=0.9,
                probe_question="what?",
                permalink="https://discord.com/channels/1/1/1",
                supersedes_claim_id=None,
                source_message_ids=(1,),
            )
        ],
    )
    old_claim_id = recorded.claim_ids[0]

    async def go() -> ExtractionReport:
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.TRIAL,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
            exchange_ids=[exchange_id],
        )

    asyncio.run(go())

    old_claim = get_claim(conn, old_claim_id)
    assert old_claim is not None
    assert old_claim.retracted_at is None
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.extraction_status is ExtractionStatus.STALE


def test_opted_out_only_exchange_is_skipped_without_calling_extractor_in_live_mode(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    opt_out(conn, 99)
    exchange = seed_exchange(conn, [a_message(1, author_id=99, content="FACT: X :: y")])
    assert exchange.id is not None

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
        )

    report = asyncio.run(go())

    assert report.skipped == 1
    assert report.succeeded == 0
    assert report.run_ids == ()
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.extraction_status is ExtractionStatus.SKIPPED
    count = conn.execute("SELECT COUNT(*) AS n FROM extraction_runs").fetchone()["n"]
    assert count == 0


def test_opted_out_only_exchange_is_skipped_in_trial_mode_without_status_change(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    opt_out(conn, 99)
    exchange = seed_exchange(conn, [a_message(1, author_id=99, content="FACT: X :: y")])
    assert exchange.id is not None
    exchange_id = exchange.id

    async def go() -> ExtractionReport:
        return await run_extraction(
            conn,
            ExplodingExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.TRIAL,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
            exchange_ids=[exchange_id],
        )

    report = asyncio.run(go())

    assert report.skipped == 1
    assert report.succeeded == 0
    assert report.claims_recorded == 0
    assert report.run_ids == ()
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.extraction_status is ExtractionStatus.PENDING
    assert updated.retry_count == 0
    count = conn.execute("SELECT COUNT(*) AS n FROM extraction_runs").fetchone()["n"]
    assert count == 0


def test_batch_loop_drains_across_multiple_batches(tmp_path: Path) -> None:
    conn = db(tmp_path)
    for i in range(1, 4):
        seed_exchange(conn, [a_message(i, channel_id=i, content=f"FACT: s{i} :: v{i}")])

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=1,
            max_retries=3,
            concurrency=1,
        )

    report = asyncio.run(go())

    assert report.processed == 3
    assert report.succeeded == 3
    assert report.claims_recorded == 3


def test_concurrency_never_exceeds_the_limit(tmp_path: Path) -> None:
    conn = db(tmp_path)
    for i in range(1, 7):
        seed_exchange(conn, [a_message(i, channel_id=i)])
    extractor = ConcurrencyTrackingExtractor()

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            extractor,
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
        )

    report = asyncio.run(go())

    assert report.processed == 6
    assert extractor.max_active <= 2
    assert extractor.max_active >= 2


def test_select_trial_sample_returns_all_when_n_ge_total(tmp_path: Path) -> None:
    conn = db(tmp_path)
    for i in range(1, 4):
        seed_exchange(conn, [a_message(i, channel_id=1)])

    assert select_trial_sample(conn, 10, seed=0) == [1, 2, 3]
    assert select_trial_sample(conn, 3, seed=0) == [1, 2, 3]


def test_select_trial_sample_empty_db(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert select_trial_sample(conn, 5, seed=0) == []


def test_select_trial_sample_is_reproducible_for_same_seed(tmp_path: Path) -> None:
    conn = db(tmp_path)
    message_id = 1
    for channel_id in (1, 2):
        for _ in range(10):
            seed_exchange(conn, [a_message(message_id, channel_id=channel_id)])
            message_id += 1

    first = select_trial_sample(conn, 6, seed=7)
    second = select_trial_sample(conn, 6, seed=7)

    assert first == second
    assert len(first) == 6


def test_select_trial_sample_stratifies_across_channels(tmp_path: Path) -> None:
    conn = db(tmp_path)
    message_id = 1
    for _ in range(10):
        seed_exchange(conn, [a_message(message_id, channel_id=1)])
        message_id += 1
    for _ in range(2):
        seed_exchange(
            conn,
            [a_message(message_id + offset, channel_id=2) for offset in range(25)],
        )
        message_id += 25

    sample = select_trial_sample(conn, 4, seed=0)

    exchanges = {
        row["id"]: row["channel_id"] for row in conn.execute("SELECT id, channel_id FROM exchanges")
    }
    sampled_channels = {exchanges[i] for i in sample}
    assert sampled_channels == {1, 2}
    assert len(sample) == 4


def test_select_trial_sample_covers_all_size_buckets(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, [a_message(1, channel_id=1)])
    seed_exchange(conn, [a_message(i, channel_id=2) for i in range(10, 13)])
    seed_exchange(conn, [a_message(i, channel_id=3) for i in range(20, 30)])
    seed_exchange(conn, [a_message(i, channel_id=4) for i in range(40, 62)])

    sample = select_trial_sample(conn, 3, seed=0)

    assert len(sample) == 3


def test_select_trial_sample_skips_an_exhausted_stratum_in_round_robin(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, [a_message(1, channel_id=1)])
    for message_id in range(2, 7):
        seed_exchange(conn, [a_message(message_id, channel_id=2)])

    sample = select_trial_sample(conn, 3, seed=0)

    exchanges = {
        row["id"]: row["channel_id"] for row in conn.execute("SELECT id, channel_id FROM exchanges")
    }
    channel_counts: dict[int, int] = {}
    for exchange_id in sample:
        channel_counts[exchanges[exchange_id]] = channel_counts.get(exchanges[exchange_id], 0) + 1
    assert channel_counts == {1: 1, 2: 2}


def test_progress_events_trial_mode_known_total(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = seed_exchange(conn, [a_message(1, content="FACT: Octane2 :: needs a jumper")])
    assert exchange.id is not None
    exchange_id = exchange.id
    events: list[ExtractionEvent] = []

    async def go() -> ExtractionReport:
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.TRIAL,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=1,
            exchange_ids=[exchange_id],
            progress=events.append,
        )

    asyncio.run(go())

    assert events == [
        ExtractionStarted(mode=RunMode.TRIAL, total=1),
        ExchangeClaimed(exchange_id=exchange_id, claims=1, index=1, total=1),
    ]


def test_progress_events_live_mode_unknown_total_and_skip(tmp_path: Path) -> None:
    conn = db(tmp_path)
    opt_out(conn, 99)
    exchange = seed_exchange(conn, [a_message(1, author_id=99, content="FACT: X :: y")])
    assert exchange.id is not None
    exchange_id = exchange.id
    events: list[ExtractionEvent] = []

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=1,
            progress=events.append,
        )

    asyncio.run(go())

    assert events == [
        ExtractionStarted(mode=RunMode.LIVE, total=None),
        ExchangeSkipped(exchange_id=exchange_id, index=1, total=None),
    ]


def test_progress_events_pause_then_failure_in_order(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, [a_message(1)])
    extractor = SequencedExtractor(
        [
            ExtractionOutcome(
                claims=(),
                model=None,
                input_tokens=None,
                output_tokens=None,
                failure=Failure(FailureKind.USAGE_LIMIT, "slow down", 5.0),
            ),
            ExtractionOutcome(
                claims=(),
                model=None,
                input_tokens=None,
                output_tokens=None,
                failure=Failure(FailureKind.TRANSIENT, "boom", None),
            ),
        ]
    )
    events: list[ExtractionEvent] = []

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            extractor,
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=1,
            concurrency=1,
            progress=events.append,
        )

    asyncio.run(go())

    assert events == [
        ExtractionStarted(mode=RunMode.LIVE, total=None),
        ExchangePaused(exchange_id=1, retry_after=5.0, index=0, total=None),
        ExchangeFailed(exchange_id=1, kind="transient", index=1, total=None),
    ]


def test_progress_defaults_to_noop(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = seed_exchange(conn, [a_message(1, content="FACT: Octane2 :: needs a jumper")])
    assert exchange.id is not None

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
        )

    report = asyncio.run(go())
    assert report.succeeded == 1


def test_select_trial_sample_different_seeds_can_differ(tmp_path: Path) -> None:
    conn = db(tmp_path)
    message_id = 1
    for _ in range(10):
        seed_exchange(conn, [a_message(message_id, channel_id=1)])
        message_id += 1

    results = {tuple(select_trial_sample(conn, 3, seed=s)) for s in range(5)}
    assert len(results) > 1


def test_select_trial_sample_min_score_filters_low_scores(tmp_path: Path) -> None:
    conn = db(tmp_path)
    high = seed_exchange(conn, [a_message(1, channel_id=1)], triage_score=0.9)
    seed_exchange(conn, [a_message(2, channel_id=1)], triage_score=0.1)
    assert high.id is not None

    sample = select_trial_sample(conn, 10, seed=0, min_score=0.5)

    assert sample == [high.id]


def test_select_trial_sample_max_score_filters_high_scores(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, [a_message(1, channel_id=1)], triage_score=0.9)
    low = seed_exchange(conn, [a_message(2, channel_id=1)], triage_score=0.1)
    assert low.id is not None

    sample = select_trial_sample(conn, 10, seed=0, max_score=0.5)

    assert sample == [low.id]


def test_select_trial_sample_min_and_max_score_bound_a_range(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, [a_message(1, channel_id=1)], triage_score=0.9)
    middle = seed_exchange(conn, [a_message(2, channel_id=1)], triage_score=0.5)
    seed_exchange(conn, [a_message(3, channel_id=1)], triage_score=0.1)
    assert middle.id is not None

    sample = select_trial_sample(conn, 10, seed=0, min_score=0.3, max_score=0.7)

    assert sample == [middle.id]


def test_select_trial_sample_without_filters_ignores_triage(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, [a_message(1, channel_id=1)], triage_score=None, triage_version=None)

    assert select_trial_sample(conn, 10, seed=0) == [1]


def test_live_refuses_when_pending_exchange_untriaged(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, [a_message(1)], triage_score=None, triage_version=None)

    async def go() -> None:
        await promote(conn)
        with pytest.raises(UntriagedExchangesError):
            await run_extraction(
                conn,
                MarkerExtractor(),
                FixedClock(NOW),
                RecordingSleeper(),
                mode=RunMode.LIVE,
                model_label="model-x",
                batch_size=10,
                max_retries=3,
                concurrency=2,
            )

    asyncio.run(go())


def test_live_refuses_when_pending_exchange_has_stale_triage_version(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, [a_message(1)], triage_score=0.9, triage_version="t0")

    async def go() -> None:
        await promote(conn)
        with pytest.raises(UntriagedExchangesError):
            await run_extraction(
                conn,
                MarkerExtractor(),
                FixedClock(NOW),
                RecordingSleeper(),
                mode=RunMode.LIVE,
                model_label="model-x",
                batch_size=10,
                max_retries=3,
                concurrency=2,
            )

    asyncio.run(go())


def test_trial_mode_never_checked_for_untriaged_exchanges(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = seed_exchange(
        conn,
        [a_message(1, content="FACT: Octane2 :: a fact")],
        triage_score=None,
        triage_version=None,
    )
    assert exchange.id is not None
    exchange_id = exchange.id

    async def go() -> ExtractionReport:
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.TRIAL,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
            exchange_ids=[exchange_id],
        )

    report = asyncio.run(go())

    assert report.claims_recorded == 1


def test_live_only_claims_exchanges_at_or_above_min_score(tmp_path: Path) -> None:
    conn = db(tmp_path)
    high = seed_exchange(
        conn, [a_message(1, content="FACT: Octane2 :: needs a jumper")], triage_score=0.9
    )
    low = seed_exchange(
        conn, [a_message(2, channel_id=2, content="FACT: Fuel :: needs a fan")], triage_score=0.1
    )
    assert high.id is not None
    assert low.id is not None

    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            MarkerExtractor(),
            FixedClock(NOW),
            RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
            min_score=0.3,
        )

    report = asyncio.run(go())

    assert report.processed == 1
    assert report.succeeded == 1
    updated_high = get_exchange(conn, high.id)
    assert updated_high is not None
    assert updated_high.extraction_status is ExtractionStatus.DONE
    updated_low = get_exchange(conn, low.id)
    assert updated_low is not None
    assert updated_low.extraction_status is ExtractionStatus.PENDING
    assert updated_low.retry_count == 0
