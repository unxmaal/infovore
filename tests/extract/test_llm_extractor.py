import json
from datetime import UTC, datetime

from infovore.extract.llm_extractor import (
    JUDGE_SYSTEM_PROMPT,
    RECALL_SYSTEM_PROMPT,
    REPAIR_PREVIOUS_OUTPUT_HEADER,
    REPAIR_VALIDATION_ERROR_HEADER,
    LLMClaimExtractor,
    LLMNoveltyProbe,
)
from infovore.extract.protocol import (
    ClaimExtractor,
    ExtractionRequest,
    FailureKind,
    NoveltyProbe,
)
from infovore.extract.schema import ExtractionOut, JudgeOut, RecallOut, json_schema_for
from infovore.llm.fake import FakeBackend
from infovore.llm.protocol import Capabilities, ErrorKind, LLMResult, Usage
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
NATIVE_CAPABILITIES = Capabilities(native_json_schema=True, max_concurrency=4)
TEXT_ONLY_CAPABILITIES = Capabilities(native_json_schema=False, max_concurrency=4)


def a_message(message_id: int, content: str = "What PROM does an Octane2 need?") -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=10,
        guild_id=100,
        author_id=1,
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


def an_exchange() -> ExchangeRow:
    return ExchangeRow(
        id=1,
        channel_id=10,
        thread_id=None,
        first_message_id=1,
        last_message_id=2,
        started_at=NOW,
        ended_at=NOW,
        message_count=2,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash="hash",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )


def a_request(messages: tuple[MessageRow, ...] | None = None) -> ExtractionRequest:
    return ExtractionRequest(
        exchange=an_exchange(),
        channel_name="hardware",
        messages=messages if messages is not None else (a_message(1), a_message(2)),
        context_messages=(),
        attachments=(),
        reactions=(),
        related_claims=(),
        opted_out_user_ids=frozenset(),
    )


def a_claim(
    statement: str = "Octane2 needs PROM 6.5",
    probe_question: str = "What PROM version does an Octane2 need?",
) -> ClaimRow:
    return ClaimRow(
        id=5,
        exchange_id=1,
        extraction_run_id=1,
        statement=statement,
        subject="Octane2",
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question=probe_question,
        permalink="https://discord.com/channels/1/1/1",
        supersedes_claim_id=None,
        novelty=Novelty.UNPROBED,
        probe_model=None,
        probe_answer=None,
        probed_at=None,
        probe_error=None,
        retracted_at=None,
        retraction_reason=None,
    )


def _claim_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "statement": "IRIX 6.5.30 requires the November 2006 overlay",
        "subject": "IRIX 6.5.30",
        "kind": "fact",
        "confidence": 0.9,
        "probe_question": "What overlay version does IRIX 6.5.30 require?",
        "source_message_ids": [1],
        "supersedes": None,
    }
    base.update(overrides)
    return base


def _extraction_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"claims": [_claim_payload()]}
    base.update(overrides)
    return base


async def test_extract_uses_native_structured_output() -> None:
    backend = FakeBackend.scripted(
        [LLMResult.ok_structured(_extraction_payload(), "claude-sonnet-4-5-20250929")],
        capabilities=NATIVE_CAPABILITIES,
    )
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert outcome.succeeded
    assert len(outcome.claims) == 1
    assert outcome.claims[0].statement == "IRIX 6.5.30 requires the November 2006 overlay"
    assert outcome.model == "claude-sonnet-4-5-20250929"
    assert len(backend.requests) == 1
    assert backend.requests[0].json_schema == json_schema_for(ExtractionOut)


async def test_extract_text_json_path_when_not_native() -> None:
    backend = FakeBackend.scripted(
        [LLMResult.ok_text(json.dumps(_extraction_payload()), "local-model-7b")],
        capabilities=TEXT_ONLY_CAPABILITIES,
    )
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert outcome.succeeded
    assert outcome.model == "local-model-7b"
    assert len(outcome.claims) == 1


async def test_extract_falls_back_to_text_when_native_but_no_structured_present() -> None:
    backend = FakeBackend.scripted(
        [LLMResult.ok_text(json.dumps(_extraction_payload()), "claude-sonnet")],
        capabilities=NATIVE_CAPABILITIES,
    )
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert outcome.succeeded
    assert outcome.model == "claude-sonnet"


async def test_extract_repair_succeeds_after_invalid_output() -> None:
    bad = LLMResult.ok_structured({"claims": [_claim_payload(source_message_ids=[999])]}, "m1")
    good = LLMResult.ok_structured(_extraction_payload(), "m2")
    backend = FakeBackend.scripted([bad, good], capabilities=NATIVE_CAPABILITIES)
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert outcome.succeeded
    assert outcome.model == "m2"
    assert len(backend.requests) == 2
    repair_request = backend.requests[1]
    assert repair_request.system == backend.requests[0].system
    assert "uncitable" in repair_request.prompt
    assert "EXCHANGE:" in repair_request.prompt


async def test_extract_repair_fails_returns_invalid_output_failure() -> None:
    bad = LLMResult.ok_structured({"claims": [_claim_payload(source_message_ids=[999])]}, "m1")
    still_bad = LLMResult.ok_structured(
        {"claims": [_claim_payload(source_message_ids=[998])]}, "m2"
    )
    backend = FakeBackend.scripted([bad, still_bad], capabilities=NATIVE_CAPABILITIES)
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert not outcome.succeeded
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.INVALID_OUTPUT
    assert outcome.model is None
    assert outcome.claims == ()


async def test_extract_repair_quotes_previous_text_output() -> None:
    bad = LLMResult.ok_text("not json at all", "m1")
    good = LLMResult.ok_text(json.dumps(_extraction_payload()), "m2")
    backend = FakeBackend.scripted([bad, good], capabilities=TEXT_ONLY_CAPABILITIES)
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert outcome.succeeded
    repair_request = backend.requests[1]
    assert "not json at all" in repair_request.prompt


async def test_extract_repair_quotes_empty_previous_output_when_result_is_bare() -> None:
    bad = LLMResult(text=None, structured=None, model="m1")
    good = LLMResult.ok_structured(_extraction_payload(), "m2")
    backend = FakeBackend.scripted([bad, good], capabilities=NATIVE_CAPABILITIES)
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert outcome.succeeded
    repair_request = backend.requests[1]
    start = repair_request.prompt.index(REPAIR_PREVIOUS_OUTPUT_HEADER) + len(
        REPAIR_PREVIOUS_OUTPUT_HEADER
    )
    end = repair_request.prompt.index(REPAIR_VALIDATION_ERROR_HEADER)
    assert repair_request.prompt[start:end].strip() == ""


async def test_extract_initial_call_transient_error() -> None:
    backend = FakeBackend.scripted([LLMResult.failed(ErrorKind.TRANSIENT, "network blip", None)])
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert not outcome.succeeded
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.TRANSIENT
    assert outcome.failure.message == "network blip"
    assert outcome.model is None


async def test_extract_initial_call_fatal_error() -> None:
    backend = FakeBackend.scripted([LLMResult.failed(ErrorKind.FATAL, "bad request", None)])
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert not outcome.succeeded
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.FATAL


async def test_extract_initial_call_usage_limit_error_carries_retry_after() -> None:
    backend = FakeBackend.scripted([LLMResult.failed(ErrorKind.USAGE_LIMIT, "limit reached", 42.0)])
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert not outcome.succeeded
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.USAGE_LIMIT
    assert outcome.failure.retry_after == 42.0


async def test_extract_repair_call_backend_error_maps_error_kind() -> None:
    bad = LLMResult.ok_structured({"claims": [_claim_payload(source_message_ids=[999])]}, "m1")
    backend = FakeBackend.scripted(
        [bad, LLMResult.failed(ErrorKind.USAGE_LIMIT, "limit reached", 7.0)],
        capabilities=NATIVE_CAPABILITIES,
    )
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert not outcome.succeeded
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.USAGE_LIMIT
    assert outcome.failure.retry_after == 7.0
    assert outcome.model is None


async def test_extract_sums_tokens_across_initial_and_repair_calls() -> None:
    bad = LLMResult(
        text=None,
        structured={"claims": [_claim_payload(source_message_ids=[999])]},
        model="m1",
        usage=Usage(input_tokens=100, output_tokens=50, cost_usd=None),
    )
    good = LLMResult(
        text=None,
        structured=_extraction_payload(),
        model="m2",
        usage=Usage(input_tokens=20, output_tokens=10, cost_usd=None),
    )
    backend = FakeBackend.scripted([bad, good], capabilities=NATIVE_CAPABILITIES)
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert outcome.succeeded
    assert outcome.input_tokens == 120
    assert outcome.output_tokens == 60


async def test_extract_tokens_none_when_backend_never_reports_them() -> None:
    backend = FakeBackend.scripted(
        [LLMResult.ok_structured(_extraction_payload(), "m1")], capabilities=NATIVE_CAPABILITIES
    )
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert outcome.input_tokens is None
    assert outcome.output_tokens is None


async def test_extract_tokens_partial_reporting_treats_missing_as_zero() -> None:
    bad = LLMResult(
        text=None,
        structured={"claims": [_claim_payload(source_message_ids=[999])]},
        model="m1",
        usage=Usage(input_tokens=None, output_tokens=None, cost_usd=None),
    )
    good = LLMResult(
        text=None,
        structured=_extraction_payload(),
        model="m2",
        usage=Usage(input_tokens=20, output_tokens=10, cost_usd=None),
    )
    backend = FakeBackend.scripted([bad, good], capabilities=NATIVE_CAPABILITIES)
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert outcome.input_tokens == 20
    assert outcome.output_tokens == 10


async def test_extract_records_canonical_model_id_not_alias() -> None:
    backend = FakeBackend.scripted(
        [LLMResult.ok_structured(_extraction_payload(), "claude-sonnet-4-5-20250929")],
        capabilities=NATIVE_CAPABILITIES,
    )
    extractor = LLMClaimExtractor(backend)
    outcome = await extractor.extract(a_request())
    assert outcome.model == "claude-sonnet-4-5-20250929"


async def test_extract_uses_configured_max_output_tokens() -> None:
    backend = FakeBackend.scripted(
        [LLMResult.ok_structured(_extraction_payload(), "m1")], capabilities=NATIVE_CAPABILITIES
    )
    extractor = LLMClaimExtractor(backend, max_output_tokens=42)
    await extractor.extract(a_request())
    assert backend.requests[0].max_output_tokens == 42


async def test_llm_claim_extractor_conforms_to_claim_extractor_protocol() -> None:
    extractor: ClaimExtractor = LLMClaimExtractor(FakeBackend.scripted([]))
    assert isinstance(extractor, LLMClaimExtractor)


async def test_recall_prompt_never_includes_the_statement() -> None:
    claim = a_claim(statement="Octane2 needs PROM 6.5")
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "I don't know"}, "probe-model")],
        capabilities=NATIVE_CAPABILITIES,
    )
    judge_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"verdict": "unknown", "reason": "no idea"}, "judge-model")],
        capabilities=NATIVE_CAPABILITIES,
    )
    probe = LLMNoveltyProbe(recall_backend, judge_backend)
    await probe.probe(claim)
    recall_request = recall_backend.requests[0]
    assert recall_request.prompt == claim.probe_question
    assert "Octane2 needs PROM 6.5" not in recall_request.prompt
    assert "Octane2 needs PROM 6.5" not in recall_request.system
    assert recall_request.system == RECALL_SYSTEM_PROMPT


async def test_probe_records_recall_canonical_model_and_answer() -> None:
    claim = a_claim()
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "PROM 6.5"}, "probe-canonical-model")],
        capabilities=NATIVE_CAPABILITIES,
    )
    judge_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"verdict": "known", "reason": "matches"}, "judge-model")],
        capabilities=NATIVE_CAPABILITIES,
    )
    probe = LLMNoveltyProbe(recall_backend, judge_backend)
    outcome = await probe.probe(claim)
    assert outcome.succeeded
    assert outcome.model == "probe-canonical-model"
    assert outcome.answer == "PROM 6.5"
    assert outcome.verdict is Novelty.KNOWN


async def test_probe_judge_system_prompt_is_the_module_constant() -> None:
    claim = a_claim()
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "PROM 6.5"}, "probe-model")],
        capabilities=NATIVE_CAPABILITIES,
    )
    judge_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"verdict": "known", "reason": "matches"}, "judge-model")],
        capabilities=NATIVE_CAPABILITIES,
    )
    probe = LLMNoveltyProbe(recall_backend, judge_backend)
    await probe.probe(claim)
    judge_request = judge_backend.requests[0]
    assert judge_request.system == JUDGE_SYSTEM_PROMPT
    assert "Octane2" in judge_request.prompt
    assert "PROM 6.5" in judge_request.prompt


async def test_probe_verdict_unknown() -> None:
    claim = a_claim()
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "I don't know"}, "probe-model")]
    )
    judge_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"verdict": "unknown", "reason": "no idea"}, "judge-model")]
    )
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(claim)
    assert outcome.verdict is Novelty.UNKNOWN


async def test_probe_verdict_partial() -> None:
    claim = a_claim()
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "some IRIX version"}, "probe-model")]
    )
    judge_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"verdict": "partial", "reason": "partial match"}, "judge-model")]
    )
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(claim)
    assert outcome.verdict is Novelty.PARTIAL


async def test_probe_verdict_contradicts() -> None:
    claim = a_claim()
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "PROM 5.0 for sure"}, "probe-model")]
    )
    judge_backend = FakeBackend.scripted(
        [
            LLMResult.ok_structured(
                {"verdict": "contradicts", "reason": "wrong version"}, "judge-model"
            )
        ]
    )
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(claim)
    assert outcome.verdict is Novelty.CONTRADICTS


async def test_probe_verdict_known() -> None:
    claim = a_claim()
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "PROM 6.5"}, "probe-model")]
    )
    judge_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"verdict": "known", "reason": "matches"}, "judge-model")]
    )
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(claim)
    assert outcome.verdict is Novelty.KNOWN


async def test_probe_recall_call_transient_error() -> None:
    recall_backend = FakeBackend.scripted([LLMResult.failed(ErrorKind.TRANSIENT, "blip", None)])
    judge_backend = FakeBackend.scripted([])
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(a_claim())
    assert not outcome.succeeded
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.TRANSIENT
    assert outcome.model is None
    assert outcome.answer is None
    assert len(judge_backend.requests) == 0


async def test_probe_recall_call_fatal_error() -> None:
    recall_backend = FakeBackend.scripted([LLMResult.failed(ErrorKind.FATAL, "bad", None)])
    judge_backend = FakeBackend.scripted([])
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(a_claim())
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.FATAL


async def test_probe_recall_call_usage_limit_error() -> None:
    recall_backend = FakeBackend.scripted(
        [LLMResult.failed(ErrorKind.USAGE_LIMIT, "limit reached", 12.0)]
    )
    judge_backend = FakeBackend.scripted([])
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(a_claim())
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.USAGE_LIMIT
    assert outcome.failure.retry_after == 12.0


async def test_probe_judge_call_transient_error() -> None:
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "PROM 6.5"}, "probe-model")]
    )
    judge_backend = FakeBackend.scripted([LLMResult.failed(ErrorKind.TRANSIENT, "blip", None)])
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(a_claim())
    assert not outcome.succeeded
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.TRANSIENT
    assert outcome.model == "probe-model"
    assert outcome.answer == "PROM 6.5"


async def test_probe_judge_call_fatal_error() -> None:
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "PROM 6.5"}, "probe-model")]
    )
    judge_backend = FakeBackend.scripted([LLMResult.failed(ErrorKind.FATAL, "bad", None)])
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(a_claim())
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.FATAL


async def test_probe_judge_call_usage_limit_error() -> None:
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "PROM 6.5"}, "probe-model")]
    )
    judge_backend = FakeBackend.scripted(
        [LLMResult.failed(ErrorKind.USAGE_LIMIT, "limit reached", 5.0)]
    )
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(a_claim())
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.USAGE_LIMIT
    assert outcome.failure.retry_after == 5.0


async def test_probe_recall_invalid_json_is_invalid_output() -> None:
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_text("not json at all", "probe-model")],
        capabilities=TEXT_ONLY_CAPABILITIES,
    )
    judge_backend = FakeBackend.scripted([])
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(a_claim())
    assert not outcome.succeeded
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.INVALID_OUTPUT
    assert outcome.model == "probe-model"
    assert outcome.answer is None
    assert len(judge_backend.requests) == 0


async def test_probe_judge_invalid_json_is_invalid_output() -> None:
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "PROM 6.5"}, "probe-model")]
    )
    judge_backend = FakeBackend.scripted(
        [LLMResult.ok_text("not json at all", "judge-model")],
        capabilities=TEXT_ONLY_CAPABILITIES,
    )
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(a_claim())
    assert not outcome.succeeded
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.INVALID_OUTPUT
    assert outcome.model == "probe-model"
    assert outcome.answer == "PROM 6.5"


async def test_probe_text_json_path_when_not_native() -> None:
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_text(json.dumps({"answer": "PROM 6.5"}), "probe-model")],
        capabilities=TEXT_ONLY_CAPABILITIES,
    )
    judge_backend = FakeBackend.scripted(
        [LLMResult.ok_text(json.dumps({"verdict": "known", "reason": "ok"}), "judge-model")],
        capabilities=TEXT_ONLY_CAPABILITIES,
    )
    outcome = await LLMNoveltyProbe(recall_backend, judge_backend).probe(a_claim())
    assert outcome.succeeded
    assert outcome.verdict is Novelty.KNOWN
    assert outcome.answer == "PROM 6.5"


async def test_probe_uses_recall_and_judge_schemas() -> None:
    recall_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"answer": "PROM 6.5"}, "probe-model")]
    )
    judge_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"verdict": "known", "reason": "ok"}, "judge-model")]
    )
    await LLMNoveltyProbe(recall_backend, judge_backend).probe(a_claim())
    assert recall_backend.requests[0].json_schema == json_schema_for(RecallOut)
    assert judge_backend.requests[0].json_schema == json_schema_for(JudgeOut)


async def test_llm_novelty_probe_conforms_to_novelty_probe_protocol() -> None:
    probe: NoveltyProbe = LLMNoveltyProbe(FakeBackend.scripted([]), FakeBackend.scripted([]))
    assert isinstance(probe, LLMNoveltyProbe)
