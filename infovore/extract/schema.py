import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from infovore.extract.prompt import LIVE_PROMPT_VERSION
from infovore.extract.protocol import ExtractedClaim
from infovore.rows import ClaimKind, Novelty


class InvalidExtractionError(ValueError):
    pass


class _ClaimBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: str
    subject: str
    kind: ClaimKind
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = Field(min_length=1)
    supersedes: int | None

    @field_validator("statement", "subject", "probe_question", check_fields=False)
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


class ClaimOut(_ClaimBase):
    probe_question: str

    @model_validator(mode="after")
    def _probe_question_must_not_leak(self) -> "ClaimOut":
        if self.statement.lower() in self.probe_question.lower():
            raise ValueError("probe_question must not contain the statement verbatim")
        return self


class ExtractionOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: list[ClaimOut]


class ClaimOutV6(_ClaimBase):
    """v6 drops `probe_question`. It existed only to feed the closed-book
    novelty probe, and once novelty means CORPUS novelty the field is 34.8%
    of the claim payload bought for nothing (issue #165)."""


class ExtractionOutV6(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: list[ClaimOutV6]


ExtractionModel = type[ExtractionOut] | type[ExtractionOutV6]

OUTPUT_MODELS: dict[str, ExtractionModel] = {
    "v5": ExtractionOut,
    "v6": ExtractionOutV6,
    "v7": ExtractionOutV6,
    "v8": ExtractionOutV6,
}


def output_model_for(version: str) -> ExtractionModel:
    try:
        return OUTPUT_MODELS[version]
    except KeyError as error:
        raise InvalidExtractionError(f"no output schema for prompt version {version}") from error


class JudgeOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: Literal["unknown", "partial", "contradicts", "known"]
    reason: str


class RecallOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str


class BatchRecallItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    answer: str


class BatchRecallOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answers: list[BatchRecallItem]


class BatchJudgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    verdict: Literal["unknown", "partial", "contradicts", "known"]
    reason: str


class BatchJudgeOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdicts: list[BatchJudgeItem]


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
    version: str = LIVE_PROMPT_VERSION,
) -> tuple[ExtractedClaim, ...]:
    data = _coerce_payload(payload)
    model = output_model_for(version)
    try:
        parsed = model.model_validate(data)
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
            probe_question=getattr(claim, "probe_question", ""),
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


def _indexed(items: list[tuple[int, object]], expected: int) -> dict[int, object]:
    """Map a batch response's items onto 0..expected-1 by their own `index`.

    Never falls back to positional order: a batch whose indices do not cover
    exactly the expected set is rejected, so a misaligned response can never
    write one claim's verdict onto another claim.
    """
    seen: dict[int, object] = {}
    for index, value in items:
        if index >= expected:
            raise InvalidExtractionError(f"index {index} out of range for batch of {expected}")
        if index in seen:
            raise InvalidExtractionError(f"duplicate index {index} in batch response")
        seen[index] = value
    missing = sorted(set(range(expected)) - seen.keys())
    if missing:
        raise InvalidExtractionError(f"batch response missing indices {missing}")
    return seen


def parse_batch_recall(payload: object, expected: int) -> list[str]:
    data = _coerce_payload(payload)
    try:
        parsed = BatchRecallOut.model_validate(data)
    except ValidationError as exc:
        raise InvalidExtractionError(str(exc)) from exc
    answers = _indexed([(item.index, item.answer) for item in parsed.answers], expected)
    return [str(answers[index]) for index in range(expected)]


def parse_batch_judge(payload: object, expected: int) -> list[Novelty]:
    data = _coerce_payload(payload)
    try:
        parsed = BatchJudgeOut.model_validate(data)
    except ValidationError as exc:
        raise InvalidExtractionError(str(exc)) from exc
    verdicts = _indexed([(item.index, item.verdict) for item in parsed.verdicts], expected)
    return [Novelty(str(verdicts[index])) for index in range(expected)]
