from datetime import UTC, datetime

from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule
from infovore.triage.gate import passes_gate

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_exchange(triage_score: float | None) -> ExchangeRow:
    return ExchangeRow(
        id=1,
        channel_id=1,
        thread_id=None,
        first_message_id=1,
        last_message_id=1,
        started_at=NOW,
        ended_at=NOW,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash="h",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
        triage_score=triage_score,
        triage_reasons=None,
        triage_version="t1",
    )


def test_passes_gate_when_score_meets_threshold() -> None:
    assert passes_gate(make_exchange(0.5), min_score=0.3) is True


def test_passes_gate_when_score_equals_threshold() -> None:
    assert passes_gate(make_exchange(0.3), min_score=0.3) is True


def test_fails_gate_when_score_below_threshold() -> None:
    assert passes_gate(make_exchange(0.1), min_score=0.3) is False


def test_fails_gate_when_score_is_none() -> None:
    assert passes_gate(make_exchange(None), min_score=0.0) is False
