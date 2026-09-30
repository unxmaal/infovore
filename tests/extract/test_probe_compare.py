import asyncio
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from infovore.db.claims import get_claim
from infovore.extract.compare import ComparisonReport, run_comparison
from infovore.extract.protocol import (
    BatchProbeOutcome,
    Failure,
    FailureKind,
    ProbeOutcome,
    ProbeUsage,
)
from infovore.rows import ClaimRow, Novelty
from infovore.timing import FixedClock
from tests.extract.test_probe_batching import NOW, db, seed


class VerdictProbe:
    """Returns a verdict chosen per claim, so a flip can be staged."""

    def __init__(self, verdicts: dict[str, Novelty]) -> None:
        self._verdicts = verdicts
        self.batch_sizes: list[int] = []

    async def probe(self, claim: ClaimRow) -> ProbeOutcome:
        return ProbeOutcome(self._verdicts[claim.probe_question], "m", "a", None)

    async def probe_batch(self, claims: Sequence[ClaimRow]) -> BatchProbeOutcome:
        self.batch_sizes.append(len(claims))
        return BatchProbeOutcome(
            outcomes=tuple(
                ProbeOutcome(self._verdicts[claim.probe_question], "m", "a", None)
                for claim in claims
            ),
            failure=None,
            usage=ProbeUsage(cost_usd=0.01),
        )


def label(conn: sqlite3.Connection, claim_id: int, verdict: Novelty) -> None:
    conn.execute(
        "UPDATE claims SET novelty = ?, probe_model = 'old', probe_answer = 'a',"
        " probed_at = ? WHERE id = ?",
        (verdict.value, NOW.isoformat(), claim_id),
    )
    conn.commit()


def compare(
    conn: sqlite3.Connection, probe: object, *, limit: int, batch_size: int = 2
) -> ComparisonReport:
    return asyncio.run(
        run_comparison(
            conn,
            probe,  # type: ignore[arg-type]
            FixedClock(NOW),
            limit=limit,
            batch_size=batch_size,
            concurrency=2,
        )
    )


def test_only_already_probed_claims_are_sampled(tmp_path: Path) -> None:
    conn = db(tmp_path)
    claim_ids = seed(conn, 3, exchanges=3)
    label(conn, claim_ids[0], Novelty.KNOWN)
    label(conn, claim_ids[1], Novelty.UNKNOWN)
    # claim_ids[2] stays unprobed: it has no reference to compare against

    report = compare(
        conn, VerdictProbe({f"question {i}?": Novelty.KNOWN for i in range(3)}), limit=10
    )

    assert report.compared == 2


def test_the_existing_verdicts_are_not_overwritten(tmp_path: Path) -> None:
    # The stored verdict is the reference. A comparison that destroys it
    # cannot be run twice.
    conn = db(tmp_path)
    claim_ids = seed(conn, 2, exchanges=2)
    label(conn, claim_ids[0], Novelty.KNOWN)
    label(conn, claim_ids[1], Novelty.KNOWN)

    compare(conn, VerdictProbe({f"question {i}?": Novelty.UNKNOWN for i in range(2)}), limit=10)

    for claim_id in claim_ids:
        claim = get_claim(conn, claim_id)
        assert claim is not None
        assert claim.novelty is Novelty.KNOWN
        assert claim.probe_model == "old"


def test_agreement_and_the_confusion_matrix(tmp_path: Path) -> None:
    conn = db(tmp_path)
    claim_ids = seed(conn, 3, exchanges=3)
    label(conn, claim_ids[0], Novelty.KNOWN)
    label(conn, claim_ids[1], Novelty.KNOWN)
    label(conn, claim_ids[2], Novelty.UNKNOWN)

    report = compare(
        conn,
        VerdictProbe(
            {
                "question 0?": Novelty.KNOWN,
                "question 1?": Novelty.UNKNOWN,
                "question 2?": Novelty.UNKNOWN,
            }
        ),
        limit=10,
    )

    assert report.compared == 3
    assert report.agreed == 2
    assert report.agreement_rate == 2 / 3
    assert report.confusion[(Novelty.KNOWN, Novelty.UNKNOWN)] == 1
    assert report.confusion[(Novelty.KNOWN, Novelty.KNOWN)] == 1


def test_known_turning_into_unknown_is_called_out(tmp_path: Path) -> None:
    # The direction that silently inflates the novelty rate.
    conn = db(tmp_path)
    claim_ids = seed(conn, 2, exchanges=2)
    label(conn, claim_ids[0], Novelty.KNOWN)
    label(conn, claim_ids[1], Novelty.PARTIAL)

    report = compare(
        conn,
        VerdictProbe({"question 0?": Novelty.UNKNOWN, "question 1?": Novelty.UNKNOWN}),
        limit=10,
    )

    assert report.known_to_unknown == 1


def test_the_sample_is_capped_by_limit(tmp_path: Path) -> None:
    conn = db(tmp_path)
    claim_ids = seed(conn, 5, exchanges=5)
    for claim_id in claim_ids:
        label(conn, claim_id, Novelty.KNOWN)

    report = compare(
        conn,
        VerdictProbe({f"question {i}?": Novelty.KNOWN for i in range(5)}),
        limit=2,
    )

    assert report.compared == 2


def test_the_comparison_reports_its_own_cost(tmp_path: Path) -> None:
    conn = db(tmp_path)
    claim_ids = seed(conn, 2, exchanges=2)
    for claim_id in claim_ids:
        label(conn, claim_id, Novelty.KNOWN)

    report = compare(
        conn,
        VerdictProbe({f"question {i}?": Novelty.KNOWN for i in range(2)}),
        limit=10,
        batch_size=2,
    )

    assert report.cost_usd == 0.01
    assert report.calls == 1


def test_nothing_to_compare_is_not_an_error(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed(conn, 1, exchanges=1)

    report = compare(conn, VerdictProbe({}), limit=10)

    assert report.compared == 0
    assert report.agreement_rate is None


def test_a_failed_batch_is_reported_rather_than_retried(tmp_path: Path) -> None:
    conn = db(tmp_path)
    claim_ids = seed(conn, 2, exchanges=2)
    for claim_id in claim_ids:
        label(conn, claim_id, Novelty.KNOWN)

    class FailingProbe:
        async def probe(self, claim: ClaimRow) -> ProbeOutcome:
            return ProbeOutcome(
                None, None, None, Failure(FailureKind.FATAL, "nope", None), ProbeUsage()
            )

        async def probe_batch(self, claims: Sequence[ClaimRow]) -> BatchProbeOutcome:
            return BatchProbeOutcome(
                None, Failure(FailureKind.INVALID_OUTPUT, "bad", None), ProbeUsage()
            )

    report = compare(conn, FailingProbe(), limit=10, batch_size=2)

    assert report.failed == 2
    assert report.compared == 0
    assert report.agreement_rate is None


def test_a_failed_single_probe_is_counted(tmp_path: Path) -> None:
    conn = db(tmp_path)
    claim_ids = seed(conn, 1, exchanges=1)
    label(conn, claim_ids[0], Novelty.KNOWN)

    class FailingProbe:
        async def probe(self, claim: ClaimRow) -> ProbeOutcome:
            return ProbeOutcome(
                None, None, None, Failure(FailureKind.FATAL, "nope", None), ProbeUsage()
            )

    report = compare(conn, FailingProbe(), limit=10, batch_size=1)

    assert report.failed == 1
    assert report.compared == 0
