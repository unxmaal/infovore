from datetime import UTC, datetime

from infovore.extract.prompt import BATCH_SYSTEM_PROMPT, render_batch_prompt
from infovore.extract.protocol import ExtractionRequest
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


def a_request(exchange_id: int, contents: list[str]) -> ExtractionRequest:
    messages = tuple(
        a_message(exchange_id * 100 + index, content)
        for index, content in enumerate(contents, start=1)
    )
    return ExtractionRequest(
        exchange=ExchangeRow(
            id=exchange_id,
            channel_id=10,
            thread_id=None,
            parent_exchange_id=None,
            first_message_id=messages[0].id,
            last_message_id=messages[-1].id,
            started_at=NOW,
            ended_at=NOW,
            message_count=len(messages),
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


def test_each_exchange_gets_its_own_ref_namespace() -> None:
    rendered = render_batch_prompt([a_request(1, ["a", "b"]), a_request(2, ["c"])])

    assert rendered.refs_by_index == [
        {"e0m1": 101, "e0m2": 102},
        {"e1m1": 201},
    ]


def test_no_ref_is_shared_between_exchanges() -> None:
    # Overlapping refs are what would let a claim land on the wrong
    # conversation without any validation noticing.
    rendered = render_batch_prompt([a_request(1, ["a"]), a_request(2, ["b"]), a_request(3, ["c"])])

    seen: set[str] = set()
    for refs in rendered.refs_by_index:
        assert not (seen & refs.keys())
        seen |= refs.keys()


def test_the_prompt_numbers_every_exchange_and_carries_its_refs() -> None:
    rendered = render_batch_prompt([a_request(1, ["first message"]), a_request(2, ["second"])])

    assert rendered.system == BATCH_SYSTEM_PROMPT
    assert "EXCHANGE 0" in rendered.prompt
    assert "EXCHANGE 1" in rendered.prompt
    assert "e0m1" in rendered.prompt
    assert "e1m1" in rendered.prompt
    assert "first message" in rendered.prompt
    assert "second" in rendered.prompt


def test_related_claim_ids_are_tracked_per_exchange() -> None:
    rendered = render_batch_prompt([a_request(1, ["a"]), a_request(2, ["b"])])

    assert rendered.related_by_index == [set(), set()]


def test_a_single_request_batch_still_namespaces() -> None:
    rendered = render_batch_prompt([a_request(1, ["a"])])

    assert rendered.refs_by_index == [{"e0m1": 101}]
