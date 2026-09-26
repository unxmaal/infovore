from datetime import UTC, datetime

import pytest

from infovore.extract.fake import MarkerExtractor, MarkerProbe
from infovore.extract.protocol import (
    ClaimExtractor,
    ExtractionRequest,
    FailureKind,
    NoveltyProbe,
)
from infovore.rows import (
    ClaimKind,
    ClaimRow,
    ExchangeRow,
    ExtractionStatus,
    GroupingRule,
    MessageRow,
    Novelty,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_message(id: int, content: str, author_id: int = 1) -> MessageRow:
    return MessageRow(
        id=id,
        channel_id=10,
        guild_id=100,
        author_id=author_id,
        author_name_at_time="alice",
        author_is_bot=False,
        created_at=NOW,
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        deleted_at=None,
        ingested_at=NOW,
        raw_json="{}",
    )


def make_exchange() -> ExchangeRow:
    return ExchangeRow(
        id=1,
        channel_id=10,
        thread_id=None,
        first_message_id=1,
        last_message_id=1,
        started_at=NOW,
        ended_at=NOW,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash="hash",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )


def make_claim_row(claim_id: int, statement: str = "s") -> ClaimRow:
    return ClaimRow(
        id=claim_id,
        exchange_id=1,
        extraction_run_id=1,
        statement=statement,
        subject="subj",
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question="q?",
        permalink="https://example/1",
        supersedes_claim_id=None,
        novelty=Novelty.UNPROBED,
        probe_model=None,
        probe_answer=None,
        probed_at=None,
        probe_error=None,
        retracted_at=None,
        retraction_reason=None,
    )


def make_request(
    messages: tuple[MessageRow, ...],
    context_messages: tuple[MessageRow, ...] = (),
    related_claims: tuple[ClaimRow, ...] = (),
    opted_out_user_ids: frozenset[int] = frozenset(),
) -> ExtractionRequest:
    return ExtractionRequest(
        exchange=make_exchange(),
        channel_name="general",
        messages=messages,
        context_messages=context_messages,
        attachments=(),
        reactions=(),
        related_claims=related_claims,
        opted_out_user_ids=opted_out_user_ids,
    )


def test_structural_typing() -> None:
    extractor: ClaimExtractor = MarkerExtractor()
    probe: NoveltyProbe = MarkerProbe()
    assert isinstance(extractor, MarkerExtractor)
    assert isinstance(probe, MarkerProbe)


async def test_fact_marker_yields_fact_claim() -> None:
    message = make_message(1, "FACT: widget :: the widget uses a 40-pin connector")
    outcome = await MarkerExtractor().extract(make_request((message,)))
    assert outcome.failure is None
    assert outcome.model == "fake-marker"
    assert outcome.input_tokens is None
    assert outcome.output_tokens is None
    assert len(outcome.claims) == 1
    claim = outcome.claims[0]
    assert claim.kind is ClaimKind.FACT
    assert claim.subject == "widget"
    assert claim.statement == "the widget uses a 40-pin connector"
    assert claim.confidence == 0.9
    assert claim.probe_question == "What is known about widget?"
    assert claim.statement not in claim.probe_question
    assert claim.source_message_ids == (1,)
    assert claim.supersedes_claim_id is None


async def test_procedure_marker_yields_procedure_claim() -> None:
    message = make_message(2, "PROCEDURE: jumper :: set JP3 to enable termination")
    outcome = await MarkerExtractor().extract(make_request((message,)))
    claim = outcome.claims[0]
    assert claim.kind is ClaimKind.PROCEDURE
    assert claim.subject == "jumper"
    assert claim.statement == "set JP3 to enable termination"


async def test_ref_marker_yields_reference_claim() -> None:
    message = make_message(3, "REF: manual :: see the IRIX admin guide")
    outcome = await MarkerExtractor().extract(make_request((message,)))
    claim = outcome.claims[0]
    assert claim.kind is ClaimKind.REFERENCE
    assert claim.subject == "manual"
    assert claim.statement == "see the IRIX admin guide"


async def test_correction_marker_without_supersedes() -> None:
    message = make_message(4, "CORRECTION: widget :: it is actually a 50-pin connector")
    outcome = await MarkerExtractor().extract(make_request((message,)))
    claim = outcome.claims[0]
    assert claim.kind is ClaimKind.CORRECTION
    assert claim.statement == "it is actually a 50-pin connector"
    assert claim.supersedes_claim_id is None


async def test_correction_marker_with_valid_supersedes() -> None:
    related = make_claim_row(7)
    message = make_message(
        5, "CORRECTION: widget :: it is a 50-pin connector [supersedes 7]"
    )
    outcome = await MarkerExtractor().extract(
        make_request((message,), related_claims=(related,))
    )
    claim = outcome.claims[0]
    assert claim.statement == "it is a 50-pin connector"
    assert claim.supersedes_claim_id == 7


async def test_correction_marker_with_unknown_supersedes_is_ignored() -> None:
    related = make_claim_row(7)
    message = make_message(
        6, "CORRECTION: widget :: it is a 50-pin connector [supersedes 99]"
    )
    outcome = await MarkerExtractor().extract(
        make_request((message,), related_claims=(related,))
    )
    claim = outcome.claims[0]
    assert claim.statement == "it is a 50-pin connector"
    assert claim.supersedes_claim_id is None


async def test_line_without_separator_is_ignored() -> None:
    message = make_message(8, "FACT: no separator here")
    outcome = await MarkerExtractor().extract(make_request((message,)))
    assert outcome.claims == ()


async def test_non_marker_lines_are_ignored() -> None:
    message = make_message(9, "just chatting about nothing in particular")
    outcome = await MarkerExtractor().extract(make_request((message,)))
    assert outcome.claims == ()


async def test_marker_is_case_sensitive() -> None:
    message = make_message(10, "fact: widget :: lowercase marker should not match")
    outcome = await MarkerExtractor().extract(make_request((message,)))
    assert outcome.claims == ()


async def test_marker_only_matches_at_line_start() -> None:
    message = make_message(11, "note: see FACT: widget :: not at line start")
    outcome = await MarkerExtractor().extract(make_request((message,)))
    assert outcome.claims == ()


async def test_multiple_lines_in_one_message_yield_multiple_claims() -> None:
    message = make_message(
        12,
        "FACT: widget :: uses a 40-pin connector\n"
        "chatter that should be ignored\n"
        "PROCEDURE: jumper :: set JP3 to enable termination",
    )
    outcome = await MarkerExtractor().extract(make_request((message,)))
    assert len(outcome.claims) == 2
    assert outcome.claims[0].kind is ClaimKind.FACT
    assert outcome.claims[1].kind is ClaimKind.PROCEDURE
    assert outcome.claims[0].source_message_ids == (12,)
    assert outcome.claims[1].source_message_ids == (12,)


async def test_context_messages_are_not_scanned() -> None:
    context_message = make_message(13, "FACT: ghost :: should never be extracted")
    real_message = make_message(14, "FACT: widget :: uses a 40-pin connector")
    outcome = await MarkerExtractor().extract(
        make_request((real_message,), context_messages=(context_message,))
    )
    assert len(outcome.claims) == 1
    assert outcome.claims[0].subject == "widget"


async def test_opted_out_author_messages_are_skipped_entirely() -> None:
    message = make_message(15, "FACT: widget :: uses a 40-pin connector", author_id=42)
    outcome = await MarkerExtractor().extract(
        make_request((message,), opted_out_user_ids=frozenset({42}))
    )
    assert outcome.claims == ()
    assert outcome.failure is None


@pytest.mark.parametrize(
    "kind",
    [FailureKind.TRANSIENT, FailureKind.USAGE_LIMIT, FailureKind.FATAL, FailureKind.INVALID_OUTPUT],
)
async def test_fail_marker_returns_matching_failure(kind: FailureKind) -> None:
    message = make_message(16, f"FAIL: {kind.value}")
    outcome = await MarkerExtractor().extract(make_request((message,)))
    assert outcome.claims == ()
    assert outcome.model is None
    assert outcome.input_tokens is None
    assert outcome.output_tokens is None
    assert outcome.failure is not None
    assert outcome.failure.kind is kind
    assert isinstance(outcome.failure.message, str) and outcome.failure.message != ""
    assert outcome.failure.retry_after == (60.0 if kind is FailureKind.USAGE_LIMIT else None)


async def test_fail_marker_short_circuits_and_discards_prior_claims() -> None:
    first = make_message(17, "FACT: widget :: uses a 40-pin connector")
    second = make_message(18, "FAIL: fatal")
    outcome = await MarkerExtractor().extract(make_request((first, second)))
    assert outcome.claims == ()
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.FATAL


async def test_fail_marker_in_opted_out_message_is_not_triggered() -> None:
    message = make_message(19, "FAIL: fatal", author_id=42)
    outcome = await MarkerExtractor().extract(
        make_request((message,), opted_out_user_ids=frozenset({42}))
    )
    assert outcome.claims == ()
    assert outcome.failure is None


async def test_probe_known_marker() -> None:
    claim = make_claim_row(1, statement="it is known [known] already")
    outcome = await MarkerProbe().probe(claim)
    assert outcome.failure is None
    assert outcome.verdict is Novelty.KNOWN
    assert outcome.model == "fake-probe"
    assert outcome.answer == "fake answer"


async def test_probe_partial_marker() -> None:
    claim = make_claim_row(2, statement="partially known [partial] here")
    outcome = await MarkerProbe().probe(claim)
    assert outcome.verdict is Novelty.PARTIAL


async def test_probe_contradicts_marker() -> None:
    claim = make_claim_row(3, statement="this contradicts belief [contradicts]")
    outcome = await MarkerProbe().probe(claim)
    assert outcome.verdict is Novelty.CONTRADICTS


async def test_probe_unknown_marker() -> None:
    claim = make_claim_row(4, statement="totally novel [unknown] fact")
    outcome = await MarkerProbe().probe(claim)
    assert outcome.verdict is Novelty.UNKNOWN


async def test_probe_defaults_to_unknown_without_marker() -> None:
    claim = make_claim_row(5, statement="no marker present at all")
    outcome = await MarkerProbe().probe(claim)
    assert outcome.verdict is Novelty.UNKNOWN
    assert outcome.failure is None


async def test_probe_fail_marker_returns_transient_failure() -> None:
    claim = make_claim_row(6, statement="something [probe-fail] happened")
    outcome = await MarkerProbe().probe(claim)
    assert outcome.verdict is None
    assert outcome.model is None
    assert outcome.answer is None
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.TRANSIENT
