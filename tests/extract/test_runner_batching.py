import asyncio
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from infovore.extract.protocol import (
    BatchExtractionOutcome,
    ExtractionOutcome,
    ExtractionRequest,
    Failure,
    FailureKind,
)
from infovore.extract.runner import ExtractionReport, run_extraction
from infovore.rows import RunMode
from infovore.timing import FixedClock, RecordingSleeper
from tests.extract.test_runner import NOW, a_message, db, promote, seed_exchange


class RecordingBatchExtractor:
    """Answers batches with a fixed claim count, recording how it was called."""

    def __init__(self, *, batch_failures: Sequence[Failure] = ()) -> None:
        self._batch_failures = list(batch_failures)
        self.batch_sizes: list[int] = []
        self.single_calls = 0

    def _outcome(self, tokens: int = 10) -> ExtractionOutcome:
        return ExtractionOutcome(
            claims=(),
            model="batch-model",
            input_tokens=tokens,
            output_tokens=tokens,
            failure=None,
            cost_usd=0.001,
        )

    async def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        self.single_calls += 1
        return ExtractionOutcome(
            claims=(),
            model="single-model",
            input_tokens=5,
            output_tokens=5,
            failure=None,
            cost_usd=0.002,
        )

    async def extract_batch(self, requests: Sequence[ExtractionRequest]) -> BatchExtractionOutcome:
        self.batch_sizes.append(len(requests))
        if self._batch_failures:
            return BatchExtractionOutcome(None, self._batch_failures.pop(0))
        return BatchExtractionOutcome(
            outcomes=tuple(self._outcome() for _ in requests), failure=None
        )


def seed(conn: sqlite3.Connection, count: int) -> None:
    for index in range(count):
        seed_exchange(conn, [a_message(index + 1)])


def run(
    conn: sqlite3.Connection,
    extractor: object,
    *,
    extract_batch_size: int,
    sleeper: RecordingSleeper | None = None,
) -> ExtractionReport:
    async def go() -> ExtractionReport:
        await promote(conn)
        return await run_extraction(
            conn,
            extractor,  # type: ignore[arg-type]
            FixedClock(NOW),
            sleeper or RecordingSleeper(),
            mode=RunMode.LIVE,
            model_label="model-x",
            batch_size=10,
            max_retries=3,
            concurrency=2,
            extract_batch_size=extract_batch_size,
        )

    return asyncio.run(go())


def test_exchanges_are_extracted_in_groups(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 6)
    extractor = RecordingBatchExtractor()

    report = run(conn, extractor, extract_batch_size=3)

    assert report.succeeded == 6
    assert sorted(extractor.batch_sizes) == [3, 3]
    assert extractor.single_calls == 0
    runs = conn.execute("SELECT model FROM extraction_runs").fetchall()
    assert [row["model"] for row in runs] == ["batch-model"] * 6


def test_batch_size_one_uses_the_per_exchange_path(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 3)
    extractor = RecordingBatchExtractor()

    report = run(conn, extractor, extract_batch_size=1)

    assert report.succeeded == 3
    assert extractor.batch_sizes == []
    assert extractor.single_calls == 3


def test_an_extractor_without_batching_still_works(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 2)

    class SingleOnly:
        def __init__(self) -> None:
            self.calls = 0

        async def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
            self.calls += 1
            return ExtractionOutcome((), "m", 1, 1, None)

    extractor = SingleOnly()
    report = run(conn, extractor, extract_batch_size=5)

    assert report.succeeded == 2
    assert extractor.calls == 2


def test_a_failed_batch_falls_back_to_one_exchange_at_a_time(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 3)
    extractor = RecordingBatchExtractor(
        batch_failures=[Failure(FailureKind.INVALID_OUTPUT, "uncitable sources", None)]
    )

    report = run(conn, extractor, extract_batch_size=3)

    assert report.succeeded == 3
    assert report.failed == 0
    assert extractor.batch_sizes == [3]
    assert extractor.single_calls == 3
    runs = conn.execute("SELECT model FROM extraction_runs").fetchall()
    assert [row["model"] for row in runs] == ["single-model"] * 3


def test_a_usage_limit_pauses_and_retries_the_whole_group(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 2)
    extractor = RecordingBatchExtractor(
        batch_failures=[Failure(FailureKind.USAGE_LIMIT, "limit", 12.0)]
    )
    sleeper = RecordingSleeper()

    report = run(conn, extractor, extract_batch_size=2, sleeper=sleeper)

    assert report.pauses == 1
    assert report.succeeded == 2
    assert sleeper.slept == [12.0]
    assert extractor.batch_sizes == [2, 2]
    assert extractor.single_calls == 0
