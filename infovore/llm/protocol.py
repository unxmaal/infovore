from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class ErrorKind(StrEnum):
    TRANSIENT = "transient"
    USAGE_LIMIT = "usage_limit"
    FATAL = "fatal"


@dataclass(frozen=True)
class LLMRequest:
    system: str
    prompt: str
    json_schema: Mapping[str, object] | None
    max_output_tokens: int


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None


@dataclass(frozen=True)
class LLMError:
    kind: ErrorKind
    message: str
    retry_after: float | None


@dataclass(frozen=True)
class LLMResult:
    text: str | None
    structured: Mapping[str, object] | None
    model: str | None
    usage: Usage = field(default_factory=lambda: Usage(None, None, None))
    error: LLMError | None = None

    @classmethod
    def ok_text(cls, text: str, model: str, usage: Usage | None = None) -> "LLMResult":
        return cls(text, None, model, usage or Usage(None, None, None))

    @classmethod
    def ok_structured(
        cls, data: Mapping[str, object], model: str, usage: Usage | None = None
    ) -> "LLMResult":
        return cls(None, data, model, usage or Usage(None, None, None))

    @classmethod
    def failed(cls, kind: ErrorKind, message: str, retry_after: float | None) -> "LLMResult":
        return cls(None, None, None, error=LLMError(kind, message, retry_after))


@dataclass(frozen=True)
class Capabilities:
    native_json_schema: bool
    max_concurrency: int


class LLMBackend(Protocol):
    def capabilities(self) -> Capabilities: ...

    async def complete(self, request: LLMRequest) -> LLMResult: ...
