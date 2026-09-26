import contextlib
import json
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from infovore.extract.protocol import ExtractedClaim
from infovore.extract.schema import (
    ClaimOut,
    ExtractionOut,
    InvalidExtractionError,
    JudgeOut,
    RecallOut,
    first_json_object,
    json_schema_for,
    parse_extraction,
    parse_judge,
    parse_recall,
)
from infovore.rows import ClaimKind, Novelty


def _claim(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "statement": "IRIX 6.5.30 requires the November 2006 overlay",
        "subject": "IRIX 6.5.30",
        "kind": "fact",
        "confidence": 0.9,
        "probe_question": "What overlay version does IRIX 6.5.30 require?",
        "source_message_ids": [1, 2],
        "supersedes": None,
    }
    base.update(overrides)
    return base


def test_claim_out_accepts_valid_payload() -> None:
    claim = ClaimOut.model_validate(_claim())
    assert claim.statement == "IRIX 6.5.30 requires the November 2006 overlay"
    assert claim.kind is ClaimKind.FACT
    assert claim.source_message_ids == [1, 2]
    assert claim.supersedes is None


def test_claim_out_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ClaimOut.model_validate(_claim(bogus="x"))


def test_claim_out_strips_and_requires_statement() -> None:
    claim = ClaimOut.model_validate(_claim(statement="  padded  "))
    assert claim.statement == "padded"
    with pytest.raises(ValidationError):
        ClaimOut.model_validate(_claim(statement="   "))


def test_claim_out_requires_subject() -> None:
    with pytest.raises(ValidationError):
        ClaimOut.model_validate(_claim(subject=""))


def test_claim_out_requires_probe_question() -> None:
    with pytest.raises(ValidationError):
        ClaimOut.model_validate(_claim(probe_question="  "))


def test_claim_out_rejects_invalid_kind() -> None:
    with pytest.raises(ValidationError):
        ClaimOut.model_validate(_claim(kind="opinion"))


def test_claim_out_confidence_bounds() -> None:
    ClaimOut.model_validate(_claim(confidence=0.0))
    ClaimOut.model_validate(_claim(confidence=1.0))
    with pytest.raises(ValidationError):
        ClaimOut.model_validate(_claim(confidence=-0.0001))
    with pytest.raises(ValidationError):
        ClaimOut.model_validate(_claim(confidence=1.0001))


def test_claim_out_rejects_empty_source_ids() -> None:
    with pytest.raises(ValidationError):
        ClaimOut.model_validate(_claim(source_message_ids=[]))


def test_claim_out_rejects_duplicate_source_ids() -> None:
    with pytest.raises(ValidationError):
        ClaimOut.model_validate(_claim(source_message_ids=[1, 1]))


def test_claim_out_accepts_supersedes_int() -> None:
    claim = ClaimOut.model_validate(_claim(supersedes=7))
    assert claim.supersedes == 7


def test_claim_out_rejects_probe_question_containing_statement() -> None:
    with pytest.raises(ValidationError):
        ClaimOut.model_validate(
            _claim(
                statement="the overlay is required",
                probe_question="Isn't it true that the overlay is required?",
            )
        )


def test_claim_out_rejects_probe_question_containing_statement_case_insensitive() -> None:
    with pytest.raises(ValidationError):
        ClaimOut.model_validate(
            _claim(
                statement="the overlay is required",
                probe_question="THE OVERLAY IS REQUIRED, right?",
            )
        )


def test_claim_out_is_frozen() -> None:
    claim = ClaimOut.model_validate(_claim())
    with pytest.raises(ValidationError):
        claim.statement = "changed"  # type: ignore[misc]


def test_extraction_out_accepts_empty_claims() -> None:
    parsed = ExtractionOut.model_validate({"claims": []})
    assert parsed.claims == []


def test_extraction_out_accepts_claims() -> None:
    parsed = ExtractionOut.model_validate({"claims": [_claim()]})
    assert len(parsed.claims) == 1


def test_extraction_out_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ExtractionOut.model_validate({"claims": [], "bogus": 1})


def test_extraction_out_rejects_invalid_claim() -> None:
    with pytest.raises(ValidationError):
        ExtractionOut.model_validate({"claims": [_claim(confidence=2.0)]})


def test_judge_out_accepts_each_valid_verdict() -> None:
    for verdict in ("unknown", "partial", "contradicts", "known"):
        judge = JudgeOut.model_validate({"verdict": verdict, "reason": "because"})
        assert judge.verdict == verdict


def test_judge_out_rejects_unprobed() -> None:
    with pytest.raises(ValidationError):
        JudgeOut.model_validate({"verdict": "unprobed", "reason": "because"})


def test_judge_out_rejects_unknown_verdict_string() -> None:
    with pytest.raises(ValidationError):
        JudgeOut.model_validate({"verdict": "maybe", "reason": "because"})


def test_judge_out_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        JudgeOut.model_validate({"verdict": "known", "reason": "because", "bogus": 1})


def test_judge_out_requires_reason() -> None:
    with pytest.raises(ValidationError):
        JudgeOut.model_validate({"verdict": "known"})


def test_recall_out_accepts_answer() -> None:
    recall = RecallOut.model_validate({"answer": "yes"})
    assert recall.answer == "yes"


def test_recall_out_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        RecallOut.model_validate({"answer": "yes", "bogus": 1})


def test_recall_out_requires_answer() -> None:
    with pytest.raises(ValidationError):
        RecallOut.model_validate({})


def test_json_schema_for_claim_out_forbids_additional_properties() -> None:
    schema = json_schema_for(ClaimOut)
    assert schema["additionalProperties"] is False
    required = cast(list[str], schema["required"])
    assert set(required) == {
        "statement",
        "subject",
        "kind",
        "confidence",
        "probe_question",
        "source_message_ids",
        "supersedes",
    }


def test_json_schema_for_extraction_out_nested_defs_forbid_additional_properties() -> None:
    schema = json_schema_for(ExtractionOut)
    assert schema["additionalProperties"] is False
    defs = cast(dict[str, dict[str, object]], schema["$defs"])
    assert defs["ClaimOut"]["additionalProperties"] is False


def test_json_schema_for_judge_out_and_recall_out() -> None:
    judge_schema = json_schema_for(JudgeOut)
    recall_schema = json_schema_for(RecallOut)
    assert judge_schema["additionalProperties"] is False
    assert recall_schema["additionalProperties"] is False


def test_first_json_object_extracts_from_surrounding_prose() -> None:
    text = 'Here is the answer:\n{"a": 1, "b": [1, 2]}\nThanks.'
    assert first_json_object(text) == {"a": 1, "b": [1, 2]}


def test_first_json_object_handles_code_fences() -> None:
    text = '```json\n{"a": 1}\n```'
    assert first_json_object(text) == {"a": 1}


def test_first_json_object_ignores_braces_inside_strings() -> None:
    text = '{"note": "a { brace and \\" quote } inside"}'
    assert first_json_object(text) == {"note": 'a { brace and " quote } inside'}


def test_first_json_object_handles_nested_objects() -> None:
    text = 'prefix {"a": {"b": 1}} suffix'
    assert first_json_object(text) == {"a": {"b": 1}}


def test_first_json_object_ignores_stray_closing_brace() -> None:
    text = 'junk } prose {"a": 1}'
    assert first_json_object(text) == {"a": 1}


def test_first_json_object_skips_unparsable_span_for_next_one() -> None:
    text = '{not json} then {"ok": true}'
    assert first_json_object(text) == {"ok": True}


def test_first_json_object_raises_when_no_brace_present() -> None:
    with pytest.raises(InvalidExtractionError):
        first_json_object("just prose, no braces here")


def test_first_json_object_raises_when_never_balanced() -> None:
    with pytest.raises(InvalidExtractionError):
        first_json_object("{ this never closes")


@given(st.text(max_size=200))
def test_first_json_object_never_raises_anything_but_invalid_extraction_error(
    text: str,
) -> None:
    with contextlib.suppress(InvalidExtractionError):
        first_json_object(text)


def test_parse_extraction_returns_extracted_claims() -> None:
    payload = {"claims": [_claim()]}
    result = parse_extraction(payload, citable_ids={1, 2}, related_claim_ids=set())
    assert result == (
        ExtractedClaim(
            statement="IRIX 6.5.30 requires the November 2006 overlay",
            subject="IRIX 6.5.30",
            kind=ClaimKind.FACT,
            confidence=0.9,
            probe_question="What overlay version does IRIX 6.5.30 require?",
            source_message_ids=(1, 2),
            supersedes_claim_id=None,
        ),
    )


def test_parse_extraction_accepts_json_string_payload() -> None:
    payload = json.dumps({"claims": [_claim()]})
    result = parse_extraction(payload, citable_ids={1, 2}, related_claim_ids=set())
    assert len(result) == 1


def test_parse_extraction_rejects_non_json_string() -> None:
    with pytest.raises(InvalidExtractionError):
        parse_extraction("not json {{{", citable_ids=set(), related_claim_ids=set())


def test_parse_extraction_rejects_structurally_invalid_payload() -> None:
    with pytest.raises(InvalidExtractionError):
        parse_extraction({"bogus": True}, citable_ids=set(), related_claim_ids=set())


def test_parse_extraction_returns_empty_tuple_for_no_claims() -> None:
    result = parse_extraction({"claims": []}, citable_ids=set(), related_claim_ids=set())
    assert result == ()


def test_parse_extraction_rejects_uncitable_source_id() -> None:
    payload = {"claims": [_claim(source_message_ids=[1, 3])]}
    with pytest.raises(InvalidExtractionError, match="3"):
        parse_extraction(payload, citable_ids={1, 2}, related_claim_ids=set())


def test_parse_extraction_rejects_unrelated_supersedes() -> None:
    payload = {"claims": [_claim(supersedes=5)]}
    with pytest.raises(InvalidExtractionError, match="5"):
        parse_extraction(payload, citable_ids={1, 2}, related_claim_ids=set())


def test_parse_extraction_accepts_related_supersedes() -> None:
    payload = {"claims": [_claim(supersedes=5)]}
    result = parse_extraction(payload, citable_ids={1, 2}, related_claim_ids={5})
    assert result[0].supersedes_claim_id == 5


def test_parse_extraction_reports_all_problems_across_claims() -> None:
    payload = {
        "claims": [
            _claim(source_message_ids=[9]),
            _claim(supersedes=42),
        ]
    }
    with pytest.raises(InvalidExtractionError) as excinfo:
        parse_extraction(payload, citable_ids={1, 2}, related_claim_ids=set())
    message = str(excinfo.value)
    assert "claim 0" in message
    assert "claim 1" in message


def test_parse_judge_maps_each_verdict() -> None:
    assert parse_judge({"verdict": "unknown", "reason": "r"}) is Novelty.UNKNOWN
    assert parse_judge({"verdict": "partial", "reason": "r"}) is Novelty.PARTIAL
    assert parse_judge({"verdict": "contradicts", "reason": "r"}) is Novelty.CONTRADICTS
    assert parse_judge({"verdict": "known", "reason": "r"}) is Novelty.KNOWN


def test_parse_judge_rejects_unprobed() -> None:
    with pytest.raises(InvalidExtractionError):
        parse_judge({"verdict": "unprobed", "reason": "r"})


def test_parse_judge_accepts_json_string() -> None:
    payload = json.dumps({"verdict": "known", "reason": "r"})
    assert parse_judge(payload) is Novelty.KNOWN


def test_parse_judge_rejects_non_json_string() -> None:
    with pytest.raises(InvalidExtractionError):
        parse_judge("not json {{{")


def test_parse_recall_returns_answer() -> None:
    assert parse_recall({"answer": "yes it does"}) == "yes it does"


def test_parse_recall_accepts_json_string() -> None:
    assert parse_recall(json.dumps({"answer": "yes"})) == "yes"


def test_parse_recall_rejects_non_json_string() -> None:
    with pytest.raises(InvalidExtractionError):
        parse_recall("not json {{{")


def test_parse_recall_rejects_structurally_invalid_payload() -> None:
    with pytest.raises(InvalidExtractionError):
        parse_recall({"bogus": True})


def test_invalid_extraction_error_is_value_error() -> None:
    assert issubclass(InvalidExtractionError, ValueError)
