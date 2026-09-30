import asyncio
from datetime import UTC, datetime

from infovore.extract.llm_extractor import (
    BATCH_JUDGE_MAX_OUTPUT_TOKENS_PER_CLAIM,
    BATCH_JUDGE_SYSTEM_PROMPT,
    BATCH_RECALL_MAX_OUTPUT_TOKENS_PER_CLAIM,
    BATCH_RECALL_SYSTEM_PROMPT,
    BatchedLLMNoveltyProbe,
    LLMNoveltyProbe,
)
from infovore.extract.protocol import BatchNoveltyProbe, FailureKind
from infovore.llm.fake import FakeBackend
from infovore.llm.protocol import Capabilities, ErrorKind, LLMResult
from infovore.rows import ClaimKind, ClaimRow, Novelty

NOW = datetime(2026, 1, 1, tzinfo=UTC)
TEXT_ONLY_CAPABILITIES = Capabilities(native_json_schema=False, max_concurrency=4)


def a_claim(claim_id: int, exchange_id: int = 1, subject: str = "Octane2") -> ClaimRow:
    return ClaimRow(
        id=claim_id,
        exchange_id=exchange_id,
        extraction_run_id=1,
        statement=f"{subject} fact {claim_id}",
        subject=subject,
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question=f"question {claim_id}?",
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


def recall_result(*pairs: tuple[int, str]) -> LLMResult:
    return LLMResult.ok_structured(
        {"answers": [{"index": i, "answer": a} for i, a in pairs]}, "probe-model"
    )


def judge_result(*pairs: tuple[int, str]) -> LLMResult:
    return LLMResult.ok_structured(
        {"verdicts": [{"index": i, "verdict": v, "reason": "r"} for i, v in pairs]},
        "judge-model",
    )


def batched(recall: LLMResult, judge: LLMResult) -> tuple[BatchedLLMNoveltyProbe, FakeBackend]:
    probe_backend = FakeBackend.scripted([recall])
    judge_backend = FakeBackend.scripted([judge])
    return BatchedLLMNoveltyProbe(probe_backend, judge_backend), probe_backend


def test_the_batch_protocol_check_discriminates() -> None:
    # run_probe switches on this isinstance, so it has to fire on the batched
    # probe and not on the per-claim one.
    backends = (FakeBackend.scripted([]), FakeBackend.scripted([]))

    assert isinstance(BatchedLLMNoveltyProbe(*backends), BatchNoveltyProbe)
    assert not isinstance(LLMNoveltyProbe(*backends), BatchNoveltyProbe)


def test_probes_a_batch_in_one_pair_of_calls() -> None:
    probe_backend = FakeBackend.scripted(
        [recall_result((0, "a0"), (1, "a1"), (2, "a2"))],
    )
    judge_backend = FakeBackend.scripted(
        [judge_result((0, "known"), (1, "unknown"), (2, "partial"))],
    )
    probe = BatchedLLMNoveltyProbe(probe_backend, judge_backend)
    claims = [a_claim(1), a_claim(2), a_claim(3)]

    result = asyncio.run(probe.probe_batch(claims))

    assert result.succeeded
    assert result.outcomes is not None
    assert [outcome.verdict for outcome in result.outcomes] == [
        Novelty.KNOWN,
        Novelty.UNKNOWN,
        Novelty.PARTIAL,
    ]
    assert [outcome.answer for outcome in result.outcomes] == ["a0", "a1", "a2"]
    assert len(probe_backend.requests) == 1
    assert len(judge_backend.requests) == 1


def test_out_of_order_responses_stay_attached_to_their_own_claim() -> None:
    probe, _ = batched(
        recall_result((2, "a2"), (0, "a0"), (1, "a1")),
        judge_result((1, "unknown"), (2, "partial"), (0, "known")),
    )
    claims = [a_claim(1), a_claim(2), a_claim(3)]

    result = asyncio.run(probe.probe_batch(claims))

    assert result.outcomes is not None
    assert [outcome.verdict for outcome in result.outcomes] == [
        Novelty.KNOWN,
        Novelty.UNKNOWN,
        Novelty.PARTIAL,
    ]
    assert [outcome.answer for outcome in result.outcomes] == ["a0", "a1", "a2"]


def test_batch_prompts_number_every_claim() -> None:
    probe_backend = FakeBackend.scripted([recall_result((0, "a0"), (1, "a1"))])
    judge_backend = FakeBackend.scripted([judge_result((0, "known"), (1, "known"))])
    probe = BatchedLLMNoveltyProbe(probe_backend, judge_backend)

    asyncio.run(probe.probe_batch([a_claim(1), a_claim(2)]))

    recall_request = probe_backend.requests[0]
    assert recall_request.system == BATCH_RECALL_SYSTEM_PROMPT
    assert "[0] question 1?" in recall_request.prompt
    assert "[1] question 2?" in recall_request.prompt

    judge_request = judge_backend.requests[0]
    assert judge_request.system == BATCH_JUDGE_SYSTEM_PROMPT
    assert "RECALL ANSWER: a0" in judge_request.prompt
    assert "RECALL ANSWER: a1" in judge_request.prompt


def test_output_caps_scale_with_the_batch() -> None:
    # The per-claim caps were sized for one answer. Reusing them for N would
    # truncate the response into a parse failure.
    probe_backend = FakeBackend.scripted([recall_result(*((i, f"a{i}") for i in range(4)))])
    judge_backend = FakeBackend.scripted([judge_result(*((i, "known") for i in range(4)))])
    probe = BatchedLLMNoveltyProbe(probe_backend, judge_backend)

    asyncio.run(probe.probe_batch([a_claim(i) for i in range(4)]))

    assert probe_backend.requests[0].max_output_tokens == (
        BATCH_RECALL_MAX_OUTPUT_TOKENS_PER_CLAIM * 4
    )
    assert judge_backend.requests[0].max_output_tokens == (
        BATCH_JUDGE_MAX_OUTPUT_TOKENS_PER_CLAIM * 4
    )


def test_an_empty_batch_makes_no_calls() -> None:
    probe_backend = FakeBackend.scripted([])
    judge_backend = FakeBackend.scripted([])
    probe = BatchedLLMNoveltyProbe(probe_backend, judge_backend)

    result = asyncio.run(probe.probe_batch([]))

    assert result.outcomes == ()
    assert result.failure is None
    assert probe_backend.requests == []


def test_recall_error_is_a_batch_level_failure() -> None:
    probe_backend = FakeBackend.scripted(
        [LLMResult.failed(ErrorKind.USAGE_LIMIT, "limit", 30.0)],
    )
    judge_backend = FakeBackend.scripted([])
    probe = BatchedLLMNoveltyProbe(probe_backend, judge_backend)

    result = asyncio.run(probe.probe_batch([a_claim(1), a_claim(2)]))

    assert result.outcomes is None
    assert result.failure is not None
    assert result.failure.kind is FailureKind.USAGE_LIMIT
    assert result.failure.retry_after == 30.0
    assert judge_backend.requests == []


def test_judge_error_is_a_batch_level_failure() -> None:
    probe, _ = batched(
        recall_result((0, "a0"), (1, "a1")),
        LLMResult.failed(ErrorKind.TRANSIENT, "boom", None),
    )

    result = asyncio.run(probe.probe_batch([a_claim(1), a_claim(2)]))

    assert result.outcomes is None
    assert result.failure is not None
    assert result.failure.kind is FailureKind.TRANSIENT


def test_a_short_recall_response_is_an_invalid_output_failure() -> None:
    probe, _ = batched(recall_result((0, "a0")), judge_result((0, "known")))

    result = asyncio.run(probe.probe_batch([a_claim(1), a_claim(2), a_claim(3)]))

    assert result.outcomes is None
    assert result.failure is not None
    assert result.failure.kind is FailureKind.INVALID_OUTPUT
    assert "missing indices" in result.failure.message


def test_a_malformed_judge_response_is_an_invalid_output_failure() -> None:
    probe, _ = batched(
        recall_result((0, "a0"), (1, "a1")),
        judge_result((0, "known"), (0, "unknown")),
    )

    result = asyncio.run(probe.probe_batch([a_claim(1), a_claim(2)]))

    assert result.outcomes is None
    assert result.failure is not None
    assert result.failure.kind is FailureKind.INVALID_OUTPUT
    assert "duplicate index" in result.failure.message


def test_text_only_backends_parse_json_out_of_the_text() -> None:
    probe_backend = FakeBackend.scripted(
        [LLMResult.ok_text('{"answers": [{"index": 0, "answer": "a0"}]}', "m")],
        TEXT_ONLY_CAPABILITIES,
    )
    judge_backend = FakeBackend.scripted(
        [LLMResult.ok_text('{"verdicts": [{"index": 0, "verdict": "known", "reason": "r"}]}', "m")],
        TEXT_ONLY_CAPABILITIES,
    )
    probe = BatchedLLMNoveltyProbe(probe_backend, judge_backend)

    result = asyncio.run(probe.probe_batch([a_claim(1)]))

    assert result.outcomes is not None
    assert result.outcomes[0].verdict is Novelty.KNOWN


def test_probe_delegates_to_the_single_claim_path() -> None:
    probe_backend = FakeBackend.scripted([LLMResult.ok_structured({"answer": "a"}, "m")])
    judge_backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"verdict": "known", "reason": "r"}, "m")],
    )
    probe = BatchedLLMNoveltyProbe(probe_backend, judge_backend)

    outcome = asyncio.run(probe.probe(a_claim(1)))

    assert outcome.succeeded
    assert outcome.verdict is Novelty.KNOWN
    # the per-claim prompts, not the batch ones
    assert probe_backend.requests[0].system != BATCH_RECALL_SYSTEM_PROMPT


def test_batched_agrees_with_per_claim_on_the_same_answers() -> None:
    # The negative control issue #126 asks for: batching must not change the
    # verdict a claim would have received one at a time.
    claims = [a_claim(1), a_claim(2), a_claim(3)]
    answers = ["a0", "a1", "a2"]
    verdicts = ["known", "unknown", "contradicts"]

    single = LLMNoveltyProbe(
        FakeBackend.scripted(
            [LLMResult.ok_structured({"answer": answer}, "m") for answer in answers],
        ),
        FakeBackend.scripted(
            [
                LLMResult.ok_structured({"verdict": verdict, "reason": "r"}, "m")
                for verdict in verdicts
            ],
        ),
    )
    one_at_a_time = [asyncio.run(single.probe(claim)) for claim in claims]

    probe, _ = batched(
        recall_result(*enumerate(answers)),
        judge_result(*enumerate(verdicts)),
    )
    as_a_batch = asyncio.run(probe.probe_batch(claims))

    assert as_a_batch.outcomes is not None
    assert [outcome.verdict for outcome in as_a_batch.outcomes] == [
        outcome.verdict for outcome in one_at_a_time
    ]
    assert [outcome.answer for outcome in as_a_batch.outcomes] == [
        outcome.answer for outcome in one_at_a_time
    ]
