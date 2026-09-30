import asyncio
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

from infovore.db.claims import get_claim, get_probe_run
from infovore.extract.llm_extractor import BatchedLLMNoveltyProbe, _sum_optional_float
from infovore.extract.novelty import run_probe
from infovore.extract.protocol import (
    BatchProbeOutcome,
    Failure,
    FailureKind,
    NoveltyProbe,
    ProbeOutcome,
    ProbeUsage,
)
from infovore.llm.fake import FakeBackend
from infovore.llm.protocol import ErrorKind, LLMResult, Usage
from infovore.rows import ClaimRow, Novelty, RunOutcome
from infovore.timing import Clock, FixedClock, RecordingSleeper
from tests.extract.test_probe_batching import NOW, db, seed


class UsageProbe:
    """Answers with a fixed usage so the persisted row can be asserted."""

    def __init__(self, usage: ProbeUsage, *, batch_failure: Failure | None = None) -> None:
        self._usage = usage
        self._batch_failure = batch_failure
        self.batch_calls = 0

    async def probe(self, claim: ClaimRow) -> ProbeOutcome:
        return ProbeOutcome(Novelty.UNKNOWN, "single-model", "a", None, self._usage)

    async def probe_batch(self, claims: Sequence[ClaimRow]) -> BatchProbeOutcome:
        self.batch_calls += 1
        if self._batch_failure is not None:
            return BatchProbeOutcome(None, self._batch_failure, self._usage)
        return BatchProbeOutcome(
            outcomes=tuple(ProbeOutcome(Novelty.KNOWN, "batch-model", "a", None) for _ in claims),
            failure=None,
            usage=self._usage,
        )


class TickingClock(Clock):
    """Advances one second per read, so a start and a finish differ."""

    def __init__(self) -> None:
        self._ticks = 0

    def now(self) -> datetime:
        self._ticks += 1
        return NOW + timedelta(seconds=self._ticks)


A_USAGE = ProbeUsage(
    probe_model="claude-sonnet-5",
    judge_model="claude-haiku-4-5",
    recall_input_tokens=700,
    recall_output_tokens=120,
    judge_input_tokens=400,
    judge_output_tokens=60,
    cost_usd=0.0042,
)


def run(
    conn: sqlite3.Connection,
    probe: NoveltyProbe,
    *,
    batch_size: int,
) -> None:
    asyncio.run(
        run_probe(
            conn,
            probe,
            FixedClock(NOW),
            RecordingSleeper(),
            probe_model=None,
            limit=50,
            concurrency=2,
            batch_size=batch_size,
        )
    )


def probe_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM probe_runs ORDER BY id"))


def test_a_batch_records_one_probe_run_for_the_whole_pair(tmp_path: Path) -> None:
    conn = db(tmp_path)
    claim_ids = seed(conn, 4, exchanges=4)

    run(conn, UsageProbe(A_USAGE), batch_size=4)

    rows = probe_runs(conn)
    assert len(rows) == 1
    row = rows[0]
    assert row["claim_count"] == 4
    assert row["batched"] == 1
    assert row["recall_input_tokens"] == 700
    assert row["judge_output_tokens"] == 60
    assert row["cost_usd"] == 0.0042
    assert row["outcome"] == "ok"
    assert row["probe_model"] == "claude-sonnet-5"
    assert row["judge_model"] == "claude-haiku-4-5"
    # every claim points at that one pair
    for claim_id in claim_ids:
        claim = get_claim(conn, claim_id)
        assert claim is not None
        assert claim.probe_run_id == row["id"]


def test_summing_probe_runs_does_not_multiply_a_batch_by_its_claims(tmp_path: Path) -> None:
    # The reason probe_runs is its own table: the same tokens recorded on each
    # claim would be counted claim_count times over.
    conn = db(tmp_path)
    seed(conn, 6, exchanges=6)

    run(conn, UsageProbe(A_USAGE), batch_size=3)

    total_cost = conn.execute("SELECT SUM(cost_usd) FROM probe_runs").fetchone()[0]
    total_claims = conn.execute("SELECT SUM(claim_count) FROM probe_runs").fetchone()[0]
    assert total_claims == 6
    assert total_cost == 2 * 0.0042  # two batches, not six claims' worth


def test_a_single_claim_probe_records_an_unbatched_run(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 2, exchanges=2)

    run(conn, UsageProbe(A_USAGE), batch_size=1)

    rows = probe_runs(conn)
    assert len(rows) == 2
    assert [row["claim_count"] for row in rows] == [1, 1]
    assert [row["batched"] for row in rows] == [0, 0]


def test_a_failed_probe_still_records_what_it_spent(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 1, exchanges=1)

    class FailingProbe:
        async def probe(self, claim: ClaimRow) -> ProbeOutcome:
            return ProbeOutcome(
                None,
                None,
                None,
                Failure(FailureKind.FATAL, "refused", None),
                A_USAGE,
            )

    run(conn, FailingProbe(), batch_size=1)

    rows = probe_runs(conn)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "failed"
    assert rows[0]["error"] == "refused"
    assert rows[0]["cost_usd"] == 0.0042


def test_a_usage_limit_records_no_probe_run(tmp_path: Path) -> None:
    # Nothing was charged and the claim is retried, so a row would be a
    # phantom cost.
    conn = db(tmp_path)
    seed(conn, 1, exchanges=1)

    class PausingThenOk:
        def __init__(self) -> None:
            self.calls = 0

        async def probe(self, claim: ClaimRow) -> ProbeOutcome:
            self.calls += 1
            if self.calls == 1:
                return ProbeOutcome(
                    None, None, None, Failure(FailureKind.USAGE_LIMIT, "limit", 1.0)
                )
            return ProbeOutcome(Novelty.UNKNOWN, "m", "a", None, A_USAGE)

    run(conn, PausingThenOk(), batch_size=1)

    rows = probe_runs(conn)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "ok"


def test_a_failed_batch_records_the_batch_and_then_each_fallback(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 2, exchanges=2)
    probe = UsageProbe(
        A_USAGE,
        batch_failure=Failure(FailureKind.INVALID_OUTPUT, "missing indices [1]", None),
    )

    run(conn, probe, batch_size=2)

    rows = probe_runs(conn)
    # the failed pair, then one pair per claim from the fallback
    assert [(row["batched"], row["outcome"]) for row in rows] == [
        (1, "failed"),
        (0, "ok"),
        (0, "ok"),
    ]
    assert rows[0]["claim_count"] == 2


def test_get_probe_run_hydrates_the_row(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 2, exchanges=2)

    run(conn, UsageProbe(A_USAGE), batch_size=2)

    run_id = probe_runs(conn)[0]["id"]
    row = get_probe_run(conn, run_id)
    assert row is not None
    assert row.claim_count == 2
    assert row.batched is True
    assert row.outcome is RunOutcome.OK
    assert row.cost_usd == 0.0042
    assert row.started_at == NOW
    assert row.finished_at == NOW
    assert get_probe_run(conn, 9999) is None


def test_the_batched_probe_reports_the_cost_of_both_calls() -> None:
    probe_backend = FakeBackend.scripted(
        [
            LLMResult.ok_structured(
                {"answers": [{"index": 0, "answer": "a0"}]},
                "claude-sonnet-5",
                Usage(700, 120, 0.003),
            )
        ],
    )
    judge_backend = FakeBackend.scripted(
        [
            LLMResult.ok_structured(
                {"verdicts": [{"index": 0, "verdict": "known", "reason": "r"}]},
                "claude-haiku-4-5",
                Usage(400, 60, 0.0012),
            )
        ],
    )
    probe = BatchedLLMNoveltyProbe(probe_backend, judge_backend)

    from tests.extract.test_batched_probe import a_claim

    result = asyncio.run(probe.probe_batch([a_claim(1)]))

    assert result.usage.probe_model == "claude-sonnet-5"
    assert result.usage.judge_model == "claude-haiku-4-5"
    assert result.usage.recall_input_tokens == 700
    assert result.usage.judge_input_tokens == 400
    assert result.usage.cost_usd == 0.003 + 0.0012


def test_a_batch_that_dies_at_the_judge_still_reports_the_recall_cost() -> None:
    probe_backend = FakeBackend.scripted(
        [
            LLMResult.ok_structured(
                {"answers": [{"index": 0, "answer": "a0"}]},
                "claude-sonnet-5",
                Usage(700, 120, 0.003),
            )
        ],
    )
    judge_backend = FakeBackend.scripted([LLMResult.failed(ErrorKind.TRANSIENT, "boom", None)])
    probe = BatchedLLMNoveltyProbe(probe_backend, judge_backend)

    from tests.extract.test_batched_probe import a_claim

    result = asyncio.run(probe.probe_batch([a_claim(1)]))

    assert result.failure is not None
    assert result.usage.recall_input_tokens == 700
    assert result.usage.cost_usd == 0.003
    assert result.usage.judge_input_tokens is None


def test_an_unreported_cost_stays_none_rather_than_becoming_zero() -> None:
    # A backend that reports nothing must not be recorded as free.
    assert _sum_optional_float(None, None) is None
    assert _sum_optional_float(None, 1.5) == 1.5
    assert _sum_optional_float(2.0, None) == 2.0


def test_a_probe_run_brackets_the_call_rather_than_one_instant(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 1, exchanges=1)

    asyncio.run(
        run_probe(
            conn,
            UsageProbe(A_USAGE),
            TickingClock(),
            RecordingSleeper(),
            probe_model=None,
            limit=50,
            concurrency=1,
            batch_size=1,
        )
    )

    row = probe_runs(conn)[0]
    assert row["started_at"] < row["finished_at"]
