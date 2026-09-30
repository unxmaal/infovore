import pytest

from infovore.extract.schema import InvalidExtractionError, parse_batch_extraction
from infovore.rows import ClaimKind


def a_claim(sources: list[str], statement: str = "Octane2 needs PROM 6.5") -> dict[str, object]:
    return {
        "statement": statement,
        "subject": "Octane2",
        "kind": "fact",
        "confidence": 0.9,
        "probe_question": "What PROM version does it need?",
        "sources": sources,
        "supersedes": None,
    }


def payload(*items: tuple[int, list[dict[str, object]]]) -> dict[str, object]:
    return {"items": [{"index": index, "claims": claims} for index, claims in items]}


REFS = [{"e0m1": 101, "e0m2": 102}, {"e1m1": 201}, {"e2m1": 301}]
RELATED: list[set[int]] = [set(), set(), set()]


def test_returns_one_claim_tuple_per_item_in_index_order() -> None:
    parsed = parse_batch_extraction(
        payload(
            (0, [a_claim(["e0m1"])]),
            (1, []),
            (2, [a_claim(["e2m1"], "Indy takes a 4600SC")]),
        ),
        REFS,
        RELATED,
    )

    assert [len(claims) for claims in parsed] == [1, 0, 1]
    assert parsed[0][0].source_message_ids == (101,)
    assert parsed[2][0].kind is ClaimKind.FACT


def test_maps_by_index_not_by_position() -> None:
    parsed = parse_batch_extraction(
        payload(
            (2, [a_claim(["e2m1"], "third")]),
            (0, [a_claim(["e0m1"], "first")]),
            (1, [a_claim(["e1m1"], "second")]),
        ),
        REFS,
        RELATED,
    )

    assert [claims[0].statement for claims in parsed] == ["first", "second", "third"]


def test_a_claim_may_not_cite_another_exchange(tmp_path: object = None) -> None:
    # The defect that makes batching dangerous: a claim attributed to the
    # wrong conversation gets the wrong permalink and the wrong sources, and
    # nothing downstream would ever notice.
    with pytest.raises(InvalidExtractionError, match="uncitable sources"):
        parse_batch_extraction(payload((0, [a_claim(["e1m1"])]), (1, []), (2, [])), REFS, RELATED)


def test_a_claim_may_not_cite_a_ref_that_exists_nowhere() -> None:
    with pytest.raises(InvalidExtractionError, match="uncitable sources"):
        parse_batch_extraction(payload((0, [a_claim(["e9m9"])]), (1, []), (2, [])), REFS, RELATED)


def test_a_missing_item_is_rejected() -> None:
    with pytest.raises(InvalidExtractionError, match=r"missing indices \[2\]"):
        parse_batch_extraction(payload((0, []), (1, [])), REFS, RELATED)


def test_a_duplicate_item_is_rejected() -> None:
    with pytest.raises(InvalidExtractionError, match="duplicate index 0"):
        parse_batch_extraction(payload((0, []), (0, []), (1, []), (2, [])), REFS, RELATED)


def test_an_item_past_the_batch_is_rejected() -> None:
    with pytest.raises(InvalidExtractionError, match="index 7 out of range"):
        parse_batch_extraction(payload((0, []), (1, []), (2, []), (7, [])), REFS, RELATED)


def test_supersedes_is_checked_against_that_item_only() -> None:
    related: list[set[int]] = [{55}, set(), set()]
    claim = a_claim(["e0m1"])
    claim["supersedes"] = 55
    parsed = parse_batch_extraction(payload((0, [claim]), (1, []), (2, [])), REFS, related)
    assert parsed[0][0].supersedes_claim_id == 55

    other = a_claim(["e1m1"])
    other["supersedes"] = 55
    with pytest.raises(InvalidExtractionError, match="is not a related claim id"):
        parse_batch_extraction(payload((0, []), (1, [other]), (2, [])), REFS, related)


def test_a_malformed_payload_is_rejected() -> None:
    with pytest.raises(InvalidExtractionError):
        parse_batch_extraction({"items": [{"index": 0}]}, REFS, RELATED)
