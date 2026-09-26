import json
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pytest

from infovore.llm.claude_cli import ClaudeCliBackend
from infovore.llm.fake import FakeBackend
from infovore.llm.process import ProcessResult
from infovore.llm.protocol import ErrorKind, LLMBackend, LLMRequest, LLMResult
from tests.llm.test_openai_compat import OpenAICompatHarness


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


class _ScriptedRunner:
    def __init__(self, result: ProcessResult) -> None:
        self._result = result

    async def run(self, argv: Sequence[str], stdin: str, cwd: str, timeout: float) -> ProcessResult:
        return self._result


def _ok(**fields: object) -> ProcessResult:
    return ProcessResult(exit_code=0, stdout=json.dumps(fields), stderr="", timed_out=False)


def _failure_payload(kind: ErrorKind, retry_after: float | None) -> ProcessResult:
    if kind is ErrorKind.TRANSIENT:
        return _ok(is_error=True, api_error_status=529, result="overloaded_error")
    if kind is ErrorKind.USAGE_LIMIT:
        if retry_after is None:
            return _ok(is_error=True, api_error_status=429, result="usage limit reached")
        target = datetime.now(UTC) + timedelta(seconds=retry_after)
        return _ok(
            is_error=True,
            api_error_status=429,
            result=f"usage limit reached, resets at {target.isoformat()}",
        )
    return _ok(is_error=True, result="something odd happened")


class ClaudeCliHarness:
    def structured(self, data: Mapping[str, object], model: str) -> LLMBackend:
        payload = _ok(structured_output=dict(data), modelUsage={model: {"outputTokens": 1}})
        return self._backend(payload)

    def text(self, text: str, model: str) -> LLMBackend:
        payload = _ok(result=text, modelUsage={model: {"outputTokens": 1}})
        return self._backend(payload)

    def failing(self, kind: ErrorKind, retry_after: float | None) -> LLMBackend:
        return self._backend(_failure_payload(kind, retry_after))

    def _backend(self, result: ProcessResult) -> LLMBackend:
        return ClaudeCliBackend(
            _ScriptedRunner(result),
            "model-x",
            timeout=30.0,
            scratch_dir=Path(tempfile.mkdtemp()),
            concurrency=2,
        )


HARNESSES: list[BackendHarness] = [FakeHarness(), ClaudeCliHarness(), OpenAICompatHarness()]

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
