import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from infovore.extract.protocol import ExtractedClaim
from infovore.rows import ClaimKind, Novelty


class InvalidExtractionError(ValueError):
    pass


class ClaimOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: str
    subject: str
    kind: ClaimKind
    confidence: float = Field(ge=0.0, le=1.0)
    probe_question: str
    sources: list[str] = Field(min_length=1)
    supersedes: int | None

    @field_validator("statement", "subject", "probe_question")
    @classmethod
    def _strip_and_require(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("sources")
    @classmethod
    def _no_duplicate_sources(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("sources must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _probe_question_must_not_leak(self) -> "ClaimOut":
        if self.statement.lower() in self.probe_question.lower():
            raise ValueError("probe_question must not contain the statement verbatim")
        return self


class ExtractionOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: list[ClaimOut]


class JudgeOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: Literal["unknown", "partial", "contradicts", "known"]
    reason: str


class RecallOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str


def json_schema_for(model: type[BaseModel]) -> dict[str, object]:
    return model.model_json_schema()


def first_json_object(text: str) -> object:
    spans: list[tuple[int, int]] = []
    depth = 0
    start = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, index))
    for span_start, span_end in spans:
        try:
            return json.loads(text[span_start : span_end + 1])
        except Exception:
            continue
    raise InvalidExtractionError("no JSON object found in text")


def _coerce_payload(payload: object) -> object:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise InvalidExtractionError(f"payload is not valid JSON: {exc}") from exc
    return payload


def parse_extraction(
    payload: object,
    citable_refs: Mapping[str, int],
    related_claim_ids: set[int],
) -> tuple[ExtractedClaim, ...]:
    data = _coerce_payload(payload)
    try:
        parsed = ExtractionOut.model_validate(data)
    except ValidationError as exc:
        raise InvalidExtractionError(str(exc)) from exc

    problems: list[str] = []
    for position, claim in enumerate(parsed.claims):
        uncitable = [ref for ref in claim.sources if ref not in citable_refs]
        if uncitable:
            problems.append(f"claim {position}: uncitable sources {uncitable}")
        if claim.supersedes is not None and claim.supersedes not in related_claim_ids:
            problems.append(
                f"claim {position}: supersedes {claim.supersedes} is not a related claim id"
            )
    if problems:
        raise InvalidExtractionError("; ".join(problems))

    return tuple(
        ExtractedClaim(
            statement=claim.statement,
            subject=claim.subject,
            kind=claim.kind,
            confidence=claim.confidence,
            probe_question=claim.probe_question,
            source_message_ids=tuple(citable_refs[ref] for ref in claim.sources),
            supersedes_claim_id=claim.supersedes,
        )
        for claim in parsed.claims
    )


def parse_judge(payload: object) -> Novelty:
    data = _coerce_payload(payload)
    try:
        parsed = JudgeOut.model_validate(data)
    except ValidationError as exc:
        raise InvalidExtractionError(str(exc)) from exc
    return Novelty(parsed.verdict)


def parse_recall(payload: object) -> str:
    data = _coerce_payload(payload)
    try:
        parsed = RecallOut.model_validate(data)
    except ValidationError as exc:
        raise InvalidExtractionError(str(exc)) from exc
    return parsed.answer
