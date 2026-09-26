from infovore.extract.protocol import (
    ExtractedClaim,
    ExtractionOutcome,
    Failure,
    FailureKind,
    ProbeOutcome,
)
from infovore.rows import ClaimKind, Novelty


def test_failure_kinds() -> None:
    assert [k.value for k in FailureKind] == [
        "transient",
        "usage_limit",
        "fatal",
        "invalid_output",
    ]


def test_outcomes_report_success_and_failure() -> None:
    claim = ExtractedClaim(
        statement="s",
        subject="subj",
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question="q?",
        source_message_ids=(1,),
        supersedes_claim_id=None,
    )
    ok = ExtractionOutcome(claims=(claim,), model="m", input_tokens=1, output_tokens=2, failure=None)
    assert ok.succeeded
    failed = ExtractionOutcome(
        claims=(),
        model=None,
        input_tokens=None,
        output_tokens=None,
        failure=Failure(FailureKind.TRANSIENT, "x", None),
    )
    assert not failed.succeeded
    probe = ProbeOutcome(verdict=Novelty.UNKNOWN, model="m", answer="I don't know", failure=None)
    assert probe.succeeded
    assert not ProbeOutcome(None, None, None, Failure(FailureKind.FATAL, "x", None)).succeeded
