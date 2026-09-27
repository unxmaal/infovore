import json

from infovore.extract.prompt import render_prompt
from infovore.extract.protocol import (
    ExtractionOutcome,
    ExtractionRequest,
    Failure,
    FailureKind,
    ProbeOutcome,
)
from infovore.extract.schema import (
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
from infovore.llm.protocol import ErrorKind, LLMBackend, LLMError, LLMRequest, LLMResult
from infovore.rows import ClaimRow

RECALL_SYSTEM_PROMPT = (
    "Answer the following question using only your own knowledge, with no "
    "other context. Be brief. If you are not sure of the answer, respond "
    'with exactly "I don\'t know".'
)

JUDGE_SYSTEM_PROMPT = (
    "You are comparing a closed-book recall answer against a claim, to "
    "judge how much the answering model already knew.\n"
    "\n"
    "Return unknown if the answer says it does not know, or is unrelated "
    "to the claim.\n"
    "Return partial if the answer gives some but not all of the claim's "
    "specifics.\n"
    "Return contradicts if the answer confidently asserts something "
    "incompatible with the claim.\n"
    "Return known if the answer is substantively the same fact as the "
    "claim.\n"
    "\n"
    "Output ONLY a JSON object matching the given schema. No other text."
)

RECALL_MAX_OUTPUT_TOKENS = 2000
JUDGE_MAX_OUTPUT_TOKENS = 2000

REPAIR_PREVIOUS_OUTPUT_HEADER = "--- PREVIOUS OUTPUT (invalid) ---"
REPAIR_VALIDATION_ERROR_HEADER = "--- VALIDATION ERROR ---"
REPAIR_INSTRUCTION = "Return a corrected JSON object only, matching the schema. No other text."

_ERROR_KIND_TO_FAILURE_KIND: dict[ErrorKind, FailureKind] = {
    ErrorKind.TRANSIENT: FailureKind.TRANSIENT,
    ErrorKind.FATAL: FailureKind.FATAL,
    ErrorKind.USAGE_LIMIT: FailureKind.USAGE_LIMIT,
}


def _failure_from_error(error: LLMError) -> Failure:
    return Failure(_ERROR_KIND_TO_FAILURE_KIND[error.kind], error.message, error.retry_after)


def _payload_from_result(result: LLMResult, native_json_schema: bool) -> object:
    if native_json_schema and result.structured is not None:
        return result.structured
    return first_json_object(result.text or "")


def _result_text(result: LLMResult) -> str:
    if result.text is not None:
        return result.text
    if result.structured is not None:
        return json.dumps(result.structured, sort_keys=True)
    return ""


def _sum_optional(first: int | None, second: int | None) -> int | None:
    if first is None and second is None:
        return None
    return (first or 0) + (second or 0)


def _repair_prompt(original_prompt: str, previous_output: str, error: str) -> str:
    return (
        f"{original_prompt}\n"
        "\n"
        f"{REPAIR_PREVIOUS_OUTPUT_HEADER}\n"
        f"{previous_output}\n"
        "\n"
        f"{REPAIR_VALIDATION_ERROR_HEADER}\n"
        f"{error}\n"
        "\n"
        f"{REPAIR_INSTRUCTION}"
    )


def _judge_prompt(claim: ClaimRow, answer: str) -> str:
    return (
        f"CLAIM SUBJECT: {claim.subject}\n"
        f"CLAIM STATEMENT: {claim.statement}\n"
        f"RECALL ANSWER: {answer}"
    )


class LLMClaimExtractor:
    def __init__(self, backend: LLMBackend, max_output_tokens: int = 8000) -> None:
        self._backend = backend
        self._max_output_tokens = max_output_tokens

    async def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        rendered = render_prompt(request)
        related_claim_ids = {claim.id for claim in request.related_claims if claim.id is not None}
        schema = json_schema_for(ExtractionOut)
        native = self._backend.capabilities().native_json_schema

        initial_request = LLMRequest(
            system=rendered.system,
            prompt=rendered.prompt,
            json_schema=schema,
            max_output_tokens=self._max_output_tokens,
        )
        initial_result = await self._backend.complete(initial_request)
        if initial_result.error is not None:
            return ExtractionOutcome(
                claims=(),
                model=None,
                input_tokens=initial_result.usage.input_tokens,
                output_tokens=initial_result.usage.output_tokens,
                failure=_failure_from_error(initial_result.error),
            )

        input_tokens = initial_result.usage.input_tokens
        output_tokens = initial_result.usage.output_tokens

        try:
            payload = _payload_from_result(initial_result, native)
            claims = parse_extraction(payload, rendered.refs, related_claim_ids)
        except InvalidExtractionError as exc:
            repair_request = LLMRequest(
                system=rendered.system,
                prompt=_repair_prompt(rendered.prompt, _result_text(initial_result), str(exc)),
                json_schema=schema,
                max_output_tokens=self._max_output_tokens,
            )
            repair_result = await self._backend.complete(repair_request)
            input_tokens = _sum_optional(input_tokens, repair_result.usage.input_tokens)
            output_tokens = _sum_optional(output_tokens, repair_result.usage.output_tokens)
            if repair_result.error is not None:
                return ExtractionOutcome(
                    claims=(),
                    model=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    failure=_failure_from_error(repair_result.error),
                )
            try:
                repair_payload = _payload_from_result(repair_result, native)
                claims = parse_extraction(repair_payload, rendered.refs, related_claim_ids)
            except InvalidExtractionError as repair_exc:
                return ExtractionOutcome(
                    claims=(),
                    model=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    failure=Failure(FailureKind.INVALID_OUTPUT, str(repair_exc), None),
                )
            return ExtractionOutcome(
                claims=claims,
                model=repair_result.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                failure=None,
            )

        return ExtractionOutcome(
            claims=claims,
            model=initial_result.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            failure=None,
        )


class LLMNoveltyProbe:
    def __init__(self, probe_backend: LLMBackend, judge_backend: LLMBackend) -> None:
        self._probe_backend = probe_backend
        self._judge_backend = judge_backend

    async def probe(self, claim: ClaimRow) -> ProbeOutcome:
        recall_request = LLMRequest(
            system=RECALL_SYSTEM_PROMPT,
            prompt=claim.probe_question,
            json_schema=json_schema_for(RecallOut),
            max_output_tokens=RECALL_MAX_OUTPUT_TOKENS,
        )
        recall_result = await self._probe_backend.complete(recall_request)
        if recall_result.error is not None:
            return ProbeOutcome(None, None, None, _failure_from_error(recall_result.error))

        try:
            recall_payload = _payload_from_result(
                recall_result, self._probe_backend.capabilities().native_json_schema
            )
            answer = parse_recall(recall_payload)
        except InvalidExtractionError as exc:
            return ProbeOutcome(
                None,
                recall_result.model,
                None,
                Failure(FailureKind.INVALID_OUTPUT, str(exc), None),
            )

        judge_request = LLMRequest(
            system=JUDGE_SYSTEM_PROMPT,
            prompt=_judge_prompt(claim, answer),
            json_schema=json_schema_for(JudgeOut),
            max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
        )
        judge_result = await self._judge_backend.complete(judge_request)
        if judge_result.error is not None:
            return ProbeOutcome(
                None, recall_result.model, answer, _failure_from_error(judge_result.error)
            )

        try:
            judge_payload = _payload_from_result(
                judge_result, self._judge_backend.capabilities().native_json_schema
            )
            verdict = parse_judge(judge_payload)
        except InvalidExtractionError as exc:
            return ProbeOutcome(
                None,
                recall_result.model,
                answer,
                Failure(FailureKind.INVALID_OUTPUT, str(exc), None),
            )

        return ProbeOutcome(verdict=verdict, model=recall_result.model, answer=answer, failure=None)
