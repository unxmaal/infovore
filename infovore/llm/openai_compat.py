import json

import openai

from infovore.config import StageSettings
from infovore.llm.protocol import Capabilities, ErrorKind, LLMBackend, LLMRequest, LLMResult, Usage


def _retry_after(exc: openai.RateLimitError) -> float | None:
    header = exc.response.headers.get("retry-after")
    return None if header is None else float(header)


def _is_usage_limit(exc: openai.RateLimitError) -> bool:
    return "insufficient_quota" in (exc.code, exc.type)


def _usage(raw: object) -> Usage:
    if raw is None:
        return Usage(None, None, None)
    return Usage(raw.prompt_tokens, raw.completion_tokens, None)  # type: ignore[attr-defined]


class OpenAICompatBackend:
    def __init__(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        json_schema_supported: bool,
        concurrency: int,
        timeout: float,
    ) -> None:
        self._client = client
        self._model = model
        self._json_schema_supported = json_schema_supported
        self._concurrency = concurrency
        self._timeout = timeout

    def capabilities(self) -> Capabilities:
        return Capabilities(
            native_json_schema=self._json_schema_supported, max_concurrency=self._concurrency
        )

    async def complete(self, request: LLMRequest) -> LLMResult:
        send_schema = self._json_schema_supported and request.json_schema is not None
        kwargs: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "max_tokens": request.max_output_tokens,
            "timeout": self._timeout,
        }
        if send_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "schema": request.json_schema,
                    "strict": True,
                },
            }
        try:
            response = await self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        except openai.RateLimitError as exc:
            kind = ErrorKind.USAGE_LIMIT if _is_usage_limit(exc) else ErrorKind.TRANSIENT
            return LLMResult.failed(kind, str(exc), _retry_after(exc))
        except (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as exc:
            return LLMResult.failed(ErrorKind.TRANSIENT, str(exc), None)
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.BadRequestError,
            openai.NotFoundError,
        ) as exc:
            return LLMResult.failed(ErrorKind.FATAL, str(exc), None)

        choice = response.choices[0]
        if choice.finish_reason in ("length", "content_filter"):
            return LLMResult.failed(
                ErrorKind.FATAL, f"finish_reason: {choice.finish_reason}", None
            )
        content = choice.message.content
        if not content:
            return LLMResult.failed(ErrorKind.FATAL, "empty completion content", None)
        model = response.model or self._model
        usage = _usage(response.usage)
        if request.json_schema is not None and self._json_schema_supported:
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                return LLMResult.failed(
                    ErrorKind.FATAL, f"invalid JSON in response: {exc}", None
                )
            return LLMResult.ok_structured(data, model, usage)
        return LLMResult.ok_text(content, model, usage)


class OpenAICompatFactory:
    name = "openai_compat"

    def validate(self, settings: StageSettings) -> list[str]:
        errors: list[str] = []
        if not settings.options.get("base_url"):
            errors.append("base_url is required")
        if not settings.options.get("api_key"):
            errors.append("api_key is required")
        flag = settings.options.get("json_schema_supported")
        if flag is not None and flag.strip().lower() not in {"true", "false"}:
            errors.append("json_schema_supported must be 'true' or 'false'")
        return errors

    def build(self, settings: StageSettings) -> LLMBackend:
        client = openai.AsyncOpenAI(
            base_url=settings.options["base_url"],
            api_key=settings.options["api_key"],
            timeout=settings.timeout_seconds,
            max_retries=0,
        )
        json_schema_supported = (
            settings.options.get("json_schema_supported", "false").strip().lower() == "true"
        )
        return OpenAICompatBackend(
            client=client,
            model=settings.model,
            json_schema_supported=json_schema_supported,
            concurrency=settings.concurrency,
            timeout=settings.timeout_seconds,
        )
