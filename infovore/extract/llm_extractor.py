import json
from collections.abc import Sequence

from infovore.extract.prompt import render_batch_prompt, render_prompt
from infovore.extract.protocol import (
    BatchExtractionOutcome,
    BatchProbeOutcome,
    ExtractionOutcome,
    ExtractionRequest,
    Failure,
    FailureKind,
    ProbeOutcome,
    ProbeUsage,
)
from infovore.extract.schema import (
    BatchExtractionOut,
    BatchJudgeOut,
    BatchRecallOut,
    ExtractionOut,
    InvalidExtractionError,
    JudgeOut,
    RecallOut,
    first_json_object,
    json_schema_for,
    parse_batch_extraction,
    parse_batch_judge,
    parse_batch_recall,
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

BATCH_RECALL_SYSTEM_PROMPT = (
    "Answer each numbered question below using only your own knowledge, with "
    "no other context.\n"
    "\n"
    "The questions are unrelated to each other. Never use one question, or "
    "your answer to it, as a hint for another: answer each as though it were "
    "the only question you had been asked.\n"
    "\n"
    "Be brief. If you are not sure of an answer, respond with exactly "
    '"I don\'t know" for that question.\n'
    "\n"
    "Return one entry per question, each carrying that question's own index. "
    "Output ONLY a JSON object matching the given schema. No other text."
)

BATCH_JUDGE_SYSTEM_PROMPT = (
    "You are comparing closed-book recall answers against claims, to judge "
    "how much the answering model already knew. Each numbered item is "
    "independent; judge it only against its own claim and answer.\n"
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
    "Return one entry per item, each carrying that item's own index. "
    "Output ONLY a JSON object matching the given schema. No other text."
)

# Output caps scale with the batch: the per-claim caps were sized for one
# answer, and reusing them for N would truncate the response into a parse
# failure and a pointless fallback.
BATCH_RECALL_MAX_OUTPUT_TOKENS_PER_CLAIM = 400
BATCH_JUDGE_MAX_OUTPUT_TOKENS_PER_CLAIM = 400

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


def _probe_usage(
    recall: LLMResult | None,
    judge: LLMResult | None,
) -> ProbeUsage:
    """Usage for one call pair. A pair that failed part-way still cost what it
    already spent, so whichever call completed is recorded."""
    return ProbeUsage(
        probe_model=recall.model if recall is not None else None,
        judge_model=judge.model if judge is not None else None,
        recall_input_tokens=recall.usage.input_tokens if recall is not None else None,
        recall_output_tokens=recall.usage.output_tokens if recall is not None else None,
        judge_input_tokens=judge.usage.input_tokens if judge is not None else None,
        judge_output_tokens=judge.usage.output_tokens if judge is not None else None,
        cost_usd=_sum_optional_float(
            recall.usage.cost_usd if recall is not None else None,
            judge.usage.cost_usd if judge is not None else None,
        ),
    )


def _sum_optional_float(first: float | None, second: float | None) -> float | None:
    if first is None and second is None:
        return None
    return (first or 0.0) + (second or 0.0)


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
                cost_usd=initial_result.usage.cost_usd,
            )

        input_tokens = initial_result.usage.input_tokens
        output_tokens = initial_result.usage.output_tokens
        cost_usd = initial_result.usage.cost_usd

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
            cost_usd = _sum_optional_float(cost_usd, repair_result.usage.cost_usd)
            if repair_result.error is not None:
                return ExtractionOutcome(
                    claims=(),
                    model=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    failure=_failure_from_error(repair_result.error),
                    cost_usd=cost_usd,
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
                    cost_usd=cost_usd,
                )
            return ExtractionOutcome(
                claims=claims,
                model=repair_result.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                failure=None,
                cost_usd=cost_usd,
            )

        return ExtractionOutcome(
            claims=claims,
            model=initial_result.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            failure=None,
            cost_usd=cost_usd,
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
            return ProbeOutcome(
                None,
                None,
                None,
                _failure_from_error(recall_result.error),
                _probe_usage(None, None),
            )

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
                _probe_usage(recall_result, None),
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
                None,
                recall_result.model,
                answer,
                _failure_from_error(judge_result.error),
                _probe_usage(recall_result, None),
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
                _probe_usage(recall_result, judge_result),
            )

        return ProbeOutcome(
            verdict=verdict,
            model=recall_result.model,
            answer=answer,
            failure=None,
            usage=_probe_usage(recall_result, judge_result),
        )


def _batch_recall_prompt(claims: Sequence[ClaimRow]) -> str:
    return "\n".join(f"[{index}] {claim.probe_question}" for index, claim in enumerate(claims))


def _batch_judge_prompt(claims: Sequence[ClaimRow], answers: Sequence[str]) -> str:
    return "\n\n".join(
        f"[{index}]\n{_judge_prompt(claim, answer)}"
        for index, (claim, answer) in enumerate(zip(claims, answers, strict=True))
    )


class BatchedLLMNoveltyProbe:
    """Probes several claims per pair of LLM calls, falling back to one-at-a-time.

    The per-claim path is kept as `probe`, both to satisfy `NoveltyProbe` and
    because it is what a batch falls back to when the response cannot be
    mapped onto the claims.
    """

    def __init__(self, probe_backend: LLMBackend, judge_backend: LLMBackend) -> None:
        self._probe_backend = probe_backend
        self._judge_backend = judge_backend
        self._single = LLMNoveltyProbe(probe_backend, judge_backend)

    async def probe(self, claim: ClaimRow) -> ProbeOutcome:
        return await self._single.probe(claim)

    async def probe_batch(self, claims: Sequence[ClaimRow]) -> BatchProbeOutcome:
        if not claims:
            return BatchProbeOutcome(outcomes=(), failure=None)
        count = len(claims)

        recall_request = LLMRequest(
            system=BATCH_RECALL_SYSTEM_PROMPT,
            prompt=_batch_recall_prompt(claims),
            json_schema=json_schema_for(BatchRecallOut),
            max_output_tokens=BATCH_RECALL_MAX_OUTPUT_TOKENS_PER_CLAIM * count,
        )
        recall_result = await self._probe_backend.complete(recall_request)
        if recall_result.error is not None:
            return BatchProbeOutcome(
                None, _failure_from_error(recall_result.error), _probe_usage(None, None)
            )

        try:
            answers = parse_batch_recall(
                _payload_from_result(
                    recall_result, self._probe_backend.capabilities().native_json_schema
                ),
                count,
            )
        except InvalidExtractionError as exc:
            return BatchProbeOutcome(
                None,
                Failure(FailureKind.INVALID_OUTPUT, str(exc), None),
                _probe_usage(recall_result, None),
            )

        judge_request = LLMRequest(
            system=BATCH_JUDGE_SYSTEM_PROMPT,
            prompt=_batch_judge_prompt(claims, answers),
            json_schema=json_schema_for(BatchJudgeOut),
            max_output_tokens=BATCH_JUDGE_MAX_OUTPUT_TOKENS_PER_CLAIM * count,
        )
        judge_result = await self._judge_backend.complete(judge_request)
        if judge_result.error is not None:
            return BatchProbeOutcome(
                None,
                _failure_from_error(judge_result.error),
                _probe_usage(recall_result, None),
            )

        try:
            verdicts = parse_batch_judge(
                _payload_from_result(
                    judge_result, self._judge_backend.capabilities().native_json_schema
                ),
                count,
            )
        except InvalidExtractionError as exc:
            return BatchProbeOutcome(
                None,
                Failure(FailureKind.INVALID_OUTPUT, str(exc), None),
                _probe_usage(recall_result, judge_result),
            )

        return BatchProbeOutcome(
            outcomes=tuple(
                ProbeOutcome(
                    verdict=verdict,
                    model=recall_result.model,
                    answer=answer,
                    failure=None,
                )
                for verdict, answer in zip(verdicts, answers, strict=True)
            ),
            failure=None,
            usage=_probe_usage(recall_result, judge_result),
        )


DEFAULT_MAX_OUTPUT_TOKENS_PER_EXCHANGE = 4000


def _split_tokens(total: int | None, weights: Sequence[int]) -> list[int | None]:
    """Divide one call's tokens across the items that shared it.

    The parts sum to the total exactly, so summing the column still gives
    what the call actually billed. Zero total weight splits evenly.
    """
    if total is None:
        return [None] * len(weights)
    count = len(weights)
    if count == 0:
        return []
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1] * count
        total_weight = count
    parts = [total * weight // total_weight for weight in weights]
    parts[0] += total - sum(parts)
    return list(parts)


def _split_cost(total: float | None, weights: Sequence[int]) -> list[float | None]:
    if total is None:
        return [None] * len(weights)
    count = len(weights)
    total_weight = sum(weights) or count
    normalized = weights if sum(weights) else [1] * count
    parts = [total * weight / total_weight for weight in normalized]
    parts[0] += total - sum(parts)
    return list(parts)


class BatchedLLMClaimExtractor:
    """Extracts several exchanges per LLM call, falling back to one at a time.

    The per-exchange path is kept as `extract`, both to satisfy
    `ClaimExtractor` and because it is what a batch falls back to.
    """

    def __init__(
        self,
        backend: LLMBackend,
        max_output_tokens: int = 8000,
        max_output_tokens_per_exchange: int = DEFAULT_MAX_OUTPUT_TOKENS_PER_EXCHANGE,
    ) -> None:
        self._backend = backend
        self._per_exchange = max_output_tokens_per_exchange
        self._single = LLMClaimExtractor(backend, max_output_tokens)

    async def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        return await self._single.extract(request)

    async def extract_batch(self, requests: Sequence[ExtractionRequest]) -> BatchExtractionOutcome:
        if not requests:
            return BatchExtractionOutcome(outcomes=(), failure=None)
        rendered = render_batch_prompt(requests)
        result = await self._backend.complete(
            LLMRequest(
                system=rendered.system,
                prompt=rendered.prompt,
                json_schema=json_schema_for(BatchExtractionOut),
                max_output_tokens=self._per_exchange * len(requests),
            )
        )
        if result.error is not None:
            return BatchExtractionOutcome(None, _failure_from_error(result.error))
        try:
            per_exchange = parse_batch_extraction(
                _payload_from_result(result, self._backend.capabilities().native_json_schema),
                rendered.refs_by_index,
                rendered.related_by_index,
            )
        except InvalidExtractionError as exc:
            return BatchExtractionOutcome(None, Failure(FailureKind.INVALID_OUTPUT, str(exc), None))

        claim_counts = [len(claims) for claims in per_exchange]
        even = [1] * len(requests)
        inputs = _split_tokens(result.usage.input_tokens, even)
        outputs = _split_tokens(result.usage.output_tokens, claim_counts)
        costs = _split_cost(result.usage.cost_usd, even)
        return BatchExtractionOutcome(
            outcomes=tuple(
                ExtractionOutcome(
                    claims=claims,
                    model=result.model,
                    input_tokens=inputs[index],
                    output_tokens=outputs[index],
                    failure=None,
                    cost_usd=costs[index],
                )
                for index, claims in enumerate(per_exchange)
            ),
            failure=None,
        )
