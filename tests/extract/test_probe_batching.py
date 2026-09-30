import asyncio
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.claims import NewClaim, get_claim, record_run, register_prompt_version
from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database
from infovore.extract.novelty import (
    ClaimFailed,
    ClaimPaused,
    ProbeEvent,
    ProbeProgress,
    ProbeReport,
    _batches,
    _ignore_probe_progress,
    _interleave_by_exchange,
    run_probe,
)
from infovore.extract.protocol import (
    BatchProbeOutcome,
    Failure,
    FailureKind,
    NoveltyProbe,
    ProbeOutcome,
)
from infovore.rows import ClaimKind, ClaimRow, ExtractionRunRow, Novelty, RunMode, RunOutcome
from infovore.timing import FixedClock, RecordingSleeper

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def seed(conn: sqlite3.Connection, claims: int, exchanges: int = 1) -> list[int]:
    register_prompt_version(conn, "v1", "sha", NOW)
    for exchange_id in range(1, exchanges + 1):
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 1, 1, 'a', ?, 'x', ?, '{}')",
            (exchange_id, to_db_time(NOW), to_db_time(NOW)),
        )
        conn.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
            " started_at, ended_at, message_count, grouping_rule, content_hash)"
            " VALUES (?, 1, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
            (
                exchange_id,
                exchange_id,
                exchange_id,
                to_db_time(NOW),
                to_db_time(NOW),
                f"h{exchange_id}",
            ),
        )
    claim_ids: list[int] = []
    for index in range(claims):
        exchange_id = (index % exchanges) + 1
        recorded = record_run(
            conn,
            ExtractionRunRow(
                id=None,
                exchange_id=exchange_id,
                model="m",
                prompt_version="v1",
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
                    exchange_id=exchange_id,
                    statement=f"statement {index}",
                    subject="s",
                    kind=ClaimKind.FACT,
                    confidence=0.9,
                    probe_question=f"question {index}?",
                    permalink="p",
                    supersedes_claim_id=None,
                    source_message_ids=(exchange_id,),
                )
            ],
        )
        claim_ids.extend(recorded.claim_ids)
    conn.commit()
    return claim_ids


class RecordingBatchProbe:
    """A probe that answers batches from a verdict-by-probe-question map."""

    def __init__(
        self,
        verdicts: dict[str, Novelty],
        *,
        batch_failures: Sequence[Failure] = (),
    ) -> None:
        self._verdicts = verdicts
        self._batch_failures = list(batch_failures)
        self.batch_sizes: list[int] = []
        self.single_calls: list[int] = []

    async def probe(self, claim: ClaimRow) -> ProbeOutcome:
        assert claim.id is not None
        self.single_calls.append(claim.id)
        return ProbeOutcome(
            verdict=self._verdicts[claim.probe_question],
            model="single-model",
            answer=f"answer for {claim.probe_question}",
            failure=None,
        )

    async def probe_batch(self, claims: Sequence[ClaimRow]) -> BatchProbeOutcome:
        self.batch_sizes.append(len(claims))
        if self._batch_failures:
            return BatchProbeOutcome(None, self._batch_failures.pop(0))
        return BatchProbeOutcome(
            outcomes=tuple(
                ProbeOutcome(
                    verdict=self._verdicts[claim.probe_question],
                    model="batch-model",
                    answer=f"answer for {claim.probe_question}",
                    failure=None,
                )
                for claim in claims
            ),
            failure=None,
        )


def verdict_map(count: int, verdict: Novelty = Novelty.UNKNOWN) -> dict[str, Novelty]:
    return {f"question {index}?": verdict for index in range(count)}


def run(
    conn: sqlite3.Connection,
    probe: NoveltyProbe,
    *,
    batch_size: int,
    sleeper: RecordingSleeper | None = None,
    progress: ProbeProgress = _ignore_probe_progress,
) -> ProbeReport:
    return asyncio.run(
        run_probe(
            conn,
            probe,
            FixedClock(NOW),
            sleeper or RecordingSleeper(),
            probe_model=None,
            limit=50,
            concurrency=2,
            batch_size=batch_size,
            progress=progress,
        )
    )


def test_batches_claims_and_records_every_verdict(tmp_path: Path) -> None:
    conn = db(tmp_path)
    claim_ids = seed(conn, 6, exchanges=6)
    probe = RecordingBatchProbe(verdict_map(6, Novelty.KNOWN))

    report = run(conn, probe, batch_size=3)

    assert report.probed == 6
    assert sorted(probe.batch_sizes) == [3, 3]
    assert probe.single_calls == []
    for claim_id in claim_ids:
        claim = get_claim(conn, claim_id)
        assert claim is not None
        assert claim.novelty is Novelty.KNOWN
        assert claim.probe_model == "batch-model"


def test_batch_size_one_uses_the_per_claim_path(tmp_path: Path) -> None:
    conn = db(tmp_path)
    claim_ids = seed(conn, 3, exchanges=3)
    probe = RecordingBatchProbe(verdict_map(3))

    report = run(conn, probe, batch_size=1)

    assert report.probed == 3
    assert probe.batch_sizes == []
    assert sorted(probe.single_calls) == sorted(claim_ids)


def test_a_probe_without_batching_still_works(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 2, exchanges=2)

    class SingleOnly:
        def __init__(self) -> None:
            self.calls = 0

        async def probe(self, claim: ClaimRow) -> ProbeOutcome:
            self.calls += 1
            return ProbeOutcome(Novelty.UNKNOWN, "m", "a", None)

    probe = SingleOnly()
    report = run(conn, probe, batch_size=10)

    assert report.probed == 2
    assert probe.calls == 2


def test_a_malformed_batch_falls_back_to_per_claim(tmp_path: Path) -> None:
    conn = db(tmp_path)
    claim_ids = seed(conn, 3, exchanges=3)
    probe = RecordingBatchProbe(
        verdict_map(3, Novelty.PARTIAL),
        batch_failures=[Failure(FailureKind.INVALID_OUTPUT, "missing indices [1]", None)],
    )

    report = run(conn, probe, batch_size=3)

    assert report.probed == 3
    assert report.failed == 0
    assert probe.batch_sizes == [3]
    assert sorted(probe.single_calls) == sorted(claim_ids)
    for claim_id in claim_ids:
        claim = get_claim(conn, claim_id)
        assert claim is not None
        # each claim got its own verdict, from the per-claim path
        assert claim.novelty is Novelty.PARTIAL
        assert claim.probe_model == "single-model"


def test_a_usage_limit_pauses_and_retries_the_whole_batch(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 2, exchanges=2)
    probe = RecordingBatchProbe(
        verdict_map(2),
        batch_failures=[Failure(FailureKind.USAGE_LIMIT, "limit", 12.0)],
    )
    sleeper = RecordingSleeper()
    events: list[ProbeEvent] = []

    report = run(conn, probe, batch_size=2, sleeper=sleeper, progress=events.append)

    assert report.pauses == 1
    assert report.probed == 2
    assert sleeper.slept == [12.0]
    assert probe.batch_sizes == [2, 2]
    assert probe.single_calls == []
    assert len([event for event in events if isinstance(event, ClaimPaused)]) == 2


def test_a_usage_limit_without_a_retry_after_uses_the_default(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 2, exchanges=2)
    probe = RecordingBatchProbe(
        verdict_map(2),
        batch_failures=[Failure(FailureKind.USAGE_LIMIT, "limit", None)],
    )
    sleeper = RecordingSleeper()

    run(conn, probe, batch_size=2, sleeper=sleeper)

    assert sleeper.slept == [300.0]


def test_fallback_records_a_per_claim_failure(tmp_path: Path) -> None:
    conn = db(tmp_path)
    claim_ids = seed(conn, 2, exchanges=2)

    class FailingSingle(RecordingBatchProbe):
        async def probe(self, claim: ClaimRow) -> ProbeOutcome:
            assert claim.id is not None
            self.single_calls.append(claim.id)
            return ProbeOutcome(None, None, None, Failure(FailureKind.FATAL, "nope", None))

    probe = FailingSingle(
        verdict_map(2),
        batch_failures=[Failure(FailureKind.FATAL, "batch died", None)],
    )
    events: list[ProbeEvent] = []

    report = run(conn, probe, batch_size=2, progress=events.append)

    assert report.failed == 2
    assert report.probed == 0
    assert len([event for event in events if isinstance(event, ClaimFailed)]) == 2
    for claim_id in claim_ids:
        claim = get_claim(conn, claim_id)
        assert claim is not None
        assert claim.probe_error == "nope"


def a_claim(claim_id: int, exchange_id: int) -> ClaimRow:
    return ClaimRow(
        id=claim_id,
        exchange_id=exchange_id,
        extraction_run_id=1,
        statement="s",
        subject="s",
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question="q?",
        permalink="p",
        supersedes_claim_id=None,
        novelty=Novelty.UNPROBED,
        probe_model=None,
        probe_answer=None,
        probed_at=None,
        probe_error=None,
        retracted_at=None,
        retraction_reason=None,
    )


def test_interleaving_separates_claims_from_the_same_exchange() -> None:
    # Three exchanges, three claims each, all of one exchange listed together.
    claims = [a_claim(index, exchange_id) for exchange_id in (1, 2, 3) for index in range(3)]

    ordered = _interleave_by_exchange(claims)

    assert [claim.exchange_id for claim in ordered] == [1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_batches_of_three_hold_three_distinct_exchanges() -> None:
    claims = [a_claim(index, exchange_id) for exchange_id in (1, 2, 3) for index in range(3)]

    groups = _batches(claims, 3)

    assert len(groups) == 3
    for group in groups:
        assert len({claim.exchange_id for claim in group}) == 3


def test_interleaving_drains_an_uneven_exchange() -> None:
    claims = [a_claim(0, 1), a_claim(1, 1), a_claim(2, 1), a_claim(3, 2)]

    ordered = _interleave_by_exchange(claims)

    assert [claim.exchange_id for claim in ordered] == [1, 2, 1, 1]


def test_batch_size_one_yields_singleton_groups() -> None:
    claims = [a_claim(0, 1), a_claim(1, 2)]

    assert _batches(claims, 1) == [[claims[0]], [claims[1]]]
