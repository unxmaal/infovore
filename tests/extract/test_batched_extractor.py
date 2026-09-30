import asyncio
from datetime import UTC, datetime

from infovore.extract.llm_extractor import (
    BatchedLLMClaimExtractor,
    _split_tokens,
)
from infovore.extract.prompt import BATCH_SYSTEM_PROMPT
from infovore.extract.protocol import ExtractionRequest, FailureKind
from infovore.llm.fake import FakeBackend
from infovore.llm.protocol import ErrorKind, LLMResult, Usage
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule, MessageRow

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def a_message(message_id: int, content: str) -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=10,
        guild_id=100,
        author_id=message_id,
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


def a_request(exchange_id: int) -> ExtractionRequest:
    messages = (a_message(exchange_id * 100 + 1, "an Octane2 needs a PROM"),)
    return ExtractionRequest(
        exchange=ExchangeRow(
            id=exchange_id,
            channel_id=10,
            thread_id=None,
            parent_exchange_id=None,
            first_message_id=messages[0].id,
            last_message_id=messages[0].id,
            started_at=NOW,
            ended_at=NOW,
            message_count=1,
            grouping_rule=GroupingRule.QUIET_GAP,
            content_hash=f"h{exchange_id}",
            extraction_status=ExtractionStatus.PENDING,
            retry_count=0,
            last_error=None,
        ),
        channel_name="sgi-development",
        messages=messages,
        context_messages=(),
        attachments=(),
        reactions=(),
        related_claims=(),
        opted_out_user_ids=frozenset(),
    )


def claim(ref: str, statement: str) -> dict[str, object]:
    return {
        "statement": statement,
        "subject": "Octane2",
        "kind": "fact",
        "confidence": 0.9,
        "probe_question": "What does it need?",
        "sources": [ref],
        "supersedes": None,
    }


def batch_result(
    items: list[tuple[int, list[dict[str, object]]]],
    usage: Usage | None = None,
) -> LLMResult:
    return LLMResult.ok_structured(
        {"items": [{"index": index, "claims": claims} for index, claims in items]},
        "claude-sonnet-5",
        usage,
    )


def totals(parts: list[int | None]) -> int:
    return sum(part or 0 for part in parts)


def test_split_tokens_sums_to_the_total_exactly() -> None:
    # Attribution must not invent or lose tokens: SUM over the rows has to
    # equal what the one call actually billed.
    assert totals(_split_tokens(100, [1, 1, 1])) == 100
    assert totals(_split_tokens(7, [1, 1, 1, 1, 1])) == 7
    assert totals(_split_tokens(0, [1, 2])) == 0
    assert _split_tokens(None, [1, 1]) == [None, None]
    assert _split_tokens(10, []) == []


def test_split_tokens_weights_by_share() -> None:
    assert _split_tokens(100, [3, 1]) == [75, 25]
    assert _split_tokens(10, [0, 0]) == [5, 5]  # no weight: split evenly


def test_extracts_every_exchange_from_one_call() -> None:
    backend = FakeBackend.scripted(
        [
            batch_result(
                [
                    (0, [claim("e0m1", "first fact")]),
                    (1, []),
                    (2, [claim("e2m1", "third fact")]),
                ],
                Usage(900, 300, 0.03),
            )
        ]
    )
    extractor = BatchedLLMClaimExtractor(backend)
    requests = [a_request(1), a_request(2), a_request(3)]

    result = asyncio.run(extractor.extract_batch(requests))

    assert result.succeeded
    assert result.outcomes is not None
    assert [len(outcome.claims) for outcome in result.outcomes] == [1, 0, 1]
    assert result.outcomes[0].claims[0].statement == "first fact"
    assert result.outcomes[0].claims[0].source_message_ids == (101,)
    assert result.outcomes[2].claims[0].source_message_ids == (301,)
    assert len(backend.requests) == 1
    assert backend.requests[0].system == BATCH_SYSTEM_PROMPT


def test_tokens_and_cost_are_attributed_without_double_counting() -> None:
    backend = FakeBackend.scripted(
        [
            batch_result(
                [
                    (0, [claim("e0m1", "first fact"), claim("e0m1", "second fact")]),
                    (1, [claim("e1m1", "third fact")]),
                ],
                Usage(900, 300, 0.03),
            )
        ]
    )
    extractor = BatchedLLMClaimExtractor(backend)

    result = asyncio.run(extractor.extract_batch([a_request(1), a_request(2)]))

    assert result.outcomes is not None
    assert sum(o.input_tokens or 0 for o in result.outcomes) == 900
    assert sum(o.output_tokens or 0 for o in result.outcomes) == 300
    assert sum(o.cost_usd or 0.0 for o in result.outcomes) == 0.03
    # output follows the claims: two claims vs one
    assert [o.output_tokens for o in result.outcomes] == [200, 100]


def test_a_claim_citing_another_exchange_fails_the_batch() -> None:
    backend = FakeBackend.scripted([batch_result([(0, [claim("e1m1", "wrong")]), (1, [])])])
    extractor = BatchedLLMClaimExtractor(backend)

    result = asyncio.run(extractor.extract_batch([a_request(1), a_request(2)]))

    assert result.outcomes is None
    assert result.failure is not None
    assert result.failure.kind is FailureKind.INVALID_OUTPUT
    assert "uncitable sources" in result.failure.message


def test_a_short_response_fails_the_batch() -> None:
    backend = FakeBackend.scripted([batch_result([(0, [])])])
    extractor = BatchedLLMClaimExtractor(backend)

    result = asyncio.run(extractor.extract_batch([a_request(1), a_request(2)]))

    assert result.outcomes is None
    assert result.failure is not None
    assert "missing indices" in result.failure.message


def test_a_backend_error_is_a_batch_level_failure() -> None:
    backend = FakeBackend.scripted([LLMResult.failed(ErrorKind.USAGE_LIMIT, "limit", 30.0)])
    extractor = BatchedLLMClaimExtractor(backend)

    result = asyncio.run(extractor.extract_batch([a_request(1), a_request(2)]))

    assert result.outcomes is None
    assert result.failure is not None
    assert result.failure.kind is FailureKind.USAGE_LIMIT
    assert result.failure.retry_after == 30.0


def test_output_cap_scales_with_the_batch() -> None:
    backend = FakeBackend.scripted([batch_result([(0, []), (1, []), (2, [])])])
    extractor = BatchedLLMClaimExtractor(backend, max_output_tokens_per_exchange=4000)

    asyncio.run(extractor.extract_batch([a_request(1), a_request(2), a_request(3)]))

    assert backend.requests[0].max_output_tokens == 12000


def test_extract_delegates_to_the_single_exchange_path() -> None:
    backend = FakeBackend.scripted(
        [LLMResult.ok_structured({"claims": []}, "claude-sonnet-5", Usage(10, 5, 0.001))]
    )
    extractor = BatchedLLMClaimExtractor(backend)

    outcome = asyncio.run(extractor.extract(a_request(1)))

    assert outcome.succeeded
    assert outcome.input_tokens == 10
    assert backend.requests[0].system != BATCH_SYSTEM_PROMPT


def test_an_empty_batch_makes_no_call() -> None:
    backend = FakeBackend.scripted([])
    extractor = BatchedLLMClaimExtractor(backend)

    result = asyncio.run(extractor.extract_batch([]))

    assert result.outcomes == ()
    assert backend.requests == []
