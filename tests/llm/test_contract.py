from collections.abc import Mapping
from typing import Protocol

import pytest

from infovore.llm.fake import FakeBackend
from infovore.llm.protocol import ErrorKind, LLMBackend, LLMRequest, LLMResult


class BackendHarness(Protocol):
    def structured(self, data: Mapping[str, object], model: str) -> LLMBackend: ...
    def text(self, text: str, model: str) -> LLMBackend: ...
    def failing(self, kind: ErrorKind, retry_after: float | None) -> LLMBackend: ...


class FakeHarness:
    def structured(self, data: Mapping[str, object], model: str) -> LLMBackend:
        return FakeBackend.scripted([LLMResult.ok_structured(data, model)])

    def text(self, text: str, model: str) -> LLMBackend:
        return FakeBackend.scripted([LLMResult.ok_text(text, model)])

    def failing(self, kind: ErrorKind, retry_after: float | None) -> LLMBackend:
        return FakeBackend.scripted([LLMResult.failed(kind, "boom", retry_after)])


HARNESSES: list[BackendHarness] = [FakeHarness()]

SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def schema_request() -> LLMRequest:
    return LLMRequest(system="s", prompt="p", json_schema=SCHEMA, max_output_tokens=256)


def text_request() -> LLMRequest:
    return LLMRequest(system="s", prompt="p", json_schema=None, max_output_tokens=256)


@pytest.mark.parametrize("harness", HARNESSES)
async def test_structured_success(harness: BackendHarness) -> None:
    result = await harness.structured({"answer": "42"}, "model-x").complete(schema_request())
    assert result.error is None
    assert result.structured == {"answer": "42"}
    assert result.model == "model-x"


@pytest.mark.parametrize("harness", HARNESSES)
async def test_text_success(harness: BackendHarness) -> None:
    result = await harness.text("plain answer", "model-x").complete(text_request())
    assert result.error is None
    assert result.text == "plain answer"
    assert result.model == "model-x"


@pytest.mark.parametrize("harness", HARNESSES)
@pytest.mark.parametrize("kind", list(ErrorKind))
async def test_each_error_kind_is_reported(harness: BackendHarness, kind: ErrorKind) -> None:
    result = await harness.failing(kind, None).complete(text_request())
    assert result.error is not None
    assert result.error.kind is kind
    assert result.structured is None


@pytest.mark.parametrize("harness", HARNESSES)
async def test_usage_limit_carries_retry_after(harness: BackendHarness) -> None:
    result = await harness.failing(ErrorKind.USAGE_LIMIT, 120.0).complete(text_request())
    assert result.error is not None
    assert result.error.retry_after == 120.0
