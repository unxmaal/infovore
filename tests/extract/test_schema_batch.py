import json

import pytest

from infovore.extract.schema import (
    InvalidExtractionError,
    parse_batch_judge,
    parse_batch_recall,
)
from infovore.rows import Novelty


def recall(*pairs: tuple[int, str]) -> dict[str, object]:
    return {"answers": [{"index": index, "answer": answer} for index, answer in pairs]}


def judge(*pairs: tuple[int, str]) -> dict[str, object]:
    return {
        "verdicts": [
            {"index": index, "verdict": verdict, "reason": "r"} for index, verdict in pairs
        ]
    }


def test_parse_batch_recall_returns_answers_in_index_order() -> None:
    parsed = parse_batch_recall(recall((0, "a"), (1, "b"), (2, "c")), 3)

    assert parsed == ["a", "b", "c"]


def test_parse_batch_recall_maps_by_index_not_by_position() -> None:
    # The response arrives out of order. Positional reading would attribute
    # each answer to the wrong claim, silently and permanently.
    parsed = parse_batch_recall(recall((2, "c"), (0, "a"), (1, "b")), 3)

    assert parsed == ["a", "b", "c"]


def test_parse_batch_recall_rejects_a_missing_index() -> None:
    with pytest.raises(InvalidExtractionError, match=r"missing indices \[1\]"):
        parse_batch_recall(recall((0, "a"), (2, "c")), 3)


def test_parse_batch_recall_rejects_a_duplicate_index() -> None:
    with pytest.raises(InvalidExtractionError, match="duplicate index 0"):
        parse_batch_recall(recall((0, "a"), (0, "b")), 2)


def test_parse_batch_recall_rejects_an_index_past_the_batch() -> None:
    with pytest.raises(InvalidExtractionError, match="index 5 out of range"):
        parse_batch_recall(recall((0, "a"), (5, "b")), 2)


def test_parse_batch_recall_rejects_a_short_response() -> None:
    # A response truncated by max_output_tokens is a partial read, not a
    # complete batch with fewer claims in it.
    with pytest.raises(InvalidExtractionError, match=r"missing indices \[2, 3\]"):
        parse_batch_recall(recall((0, "a"), (1, "b")), 4)


def test_parse_batch_recall_rejects_a_malformed_payload() -> None:
    with pytest.raises(InvalidExtractionError):
        parse_batch_recall({"answers": [{"index": -1, "answer": "a"}]}, 1)


def test_parse_batch_recall_accepts_a_json_string() -> None:
    assert parse_batch_recall(json.dumps(recall((0, "a"))), 1) == ["a"]


def test_parse_batch_recall_rejects_invalid_json() -> None:
    with pytest.raises(InvalidExtractionError, match="not valid JSON"):
        parse_batch_recall("{nope", 1)


def test_parse_batch_judge_returns_verdicts_in_index_order() -> None:
    parsed = parse_batch_judge(judge((0, "known"), (1, "unknown")), 2)

    assert parsed == [Novelty.KNOWN, Novelty.UNKNOWN]


def test_parse_batch_judge_maps_by_index_not_by_position() -> None:
    parsed = parse_batch_judge(judge((1, "contradicts"), (0, "partial")), 2)

    assert parsed == [Novelty.PARTIAL, Novelty.CONTRADICTS]


def test_parse_batch_judge_rejects_a_missing_index() -> None:
    with pytest.raises(InvalidExtractionError, match=r"missing indices \[1\]"):
        parse_batch_judge(judge((0, "known")), 2)


def test_parse_batch_judge_rejects_an_unknown_verdict() -> None:
    with pytest.raises(InvalidExtractionError):
        parse_batch_judge(judge((0, "maybe")), 1)
