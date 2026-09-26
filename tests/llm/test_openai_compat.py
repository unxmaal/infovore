import json
from collections.abc import Callable, Mapping

import httpx
import openai
import pytest

from infovore.llm.openai_compat import OpenAICompatBackend
from infovore.llm.protocol import Capabilities, ErrorKind, LLMBackend, LLMRequest, Usage

SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> openai.AsyncOpenAI:
    transport = httpx.MockTransport(handler)
    return openai.AsyncOpenAI(
        base_url="http://fake.local/v1",
        api_key="k",
        http_client=httpx.AsyncClient(transport=transport),
        timeout=5.0,
        max_retries=0,
    )


def completion_response(
    content: str | None,
    model: str = "model-x",
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> httpx.Response:
    body: dict[str, object] = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def backend(
    handler: Callable[[httpx.Request], httpx.Response],
    json_schema_supported: bool = True,
    model: str = "model-x",
) -> OpenAICompatBackend:
    return OpenAICompatBackend(
        client=make_client(handler),
        model=model,
        json_schema_supported=json_schema_supported,
        concurrency=3,
        timeout=7.0,
    )


def request(json_schema: Mapping[str, object] | None = None) -> LLMRequest:
    return LLMRequest(system="s", prompt="p", json_schema=json_schema, max_output_tokens=64)


def test_capabilities_reflect_config() -> None:
    back = backend(lambda req: completion_response("x"), json_schema_supported=True)
    assert back.capabilities() == Capabilities(native_json_schema=True, max_concurrency=3)


async def test_text_success() -> None:
    back = backend(lambda req: completion_response("hello there"))
    result = await back.complete(request())
    assert result.error is None
    assert result.text == "hello there"
    assert result.structured is None
    assert result.model == "model-x"


async def test_structured_success_parses_json() -> None:
    back = backend(lambda req: completion_response('{"answer": "42"}'))
    result = await back.complete(request(SCHEMA))
    assert result.error is None
    assert result.structured == {"answer": "42"}
    assert result.text is None


async def test_schema_not_sent_when_unsupported() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return completion_response('{"answer": "raw"}')

    back = backend(handler, json_schema_supported=False)
    result = await back.complete(request(SCHEMA))
    assert "response_format" not in captured
    assert result.error is None
    assert result.text == '{"answer": "raw"}'
    assert result.structured is None


async def test_schema_sent_when_supported() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return completion_response('{"answer": "42"}')

    back = backend(handler, json_schema_supported=True)
    await back.complete(request(SCHEMA))
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "output", "schema": dict(SCHEMA), "strict": True},
    }


async def test_usage_recorded_when_present() -> None:
    back = backend(
        lambda req: completion_response(
            "hi", usage={"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}
        )
    )
    result = await back.complete(request())
    assert result.usage == Usage(11, 22, None)


async def test_usage_missing_is_none() -> None:
    back = backend(lambda req: completion_response("hi"))
    result = await back.complete(request())
    assert result.usage == Usage(None, None, None)


async def test_model_falls_back_to_configured_when_blank() -> None:
    back = backend(lambda req: completion_response("hi", model=""), model="configured-model")
    result = await back.complete(request())
    assert result.model == "configured-model"


async def test_rate_limit_maps_to_transient_with_retry_after() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "30"}, json={"error": {"message": "s"}})

    back = backend(handler)
    result = await back.complete(request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.TRANSIENT
    assert result.error.retry_after == 30.0


async def test_rate_limit_without_retry_after_header() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "s"}})

    back = backend(handler)
    result = await back.complete(request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.TRANSIENT
    assert result.error.retry_after is None


async def test_rate_limit_insufficient_quota_maps_to_usage_limit() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "120"},
            json={"error": {"message": "q", "code": "insufficient_quota"}},
        )

    back = backend(handler)
    result = await back.complete(request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.USAGE_LIMIT
    assert result.error.retry_after == 120.0


@pytest.mark.parametrize("status", [500, 502, 503])
async def test_server_errors_map_to_transient(status: int) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "boom"}})

    back = backend(handler)
    result = await back.complete(request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.TRANSIENT
    assert result.error.retry_after is None


async def test_timeout_maps_to_transient() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    back = backend(handler)
    result = await back.complete(request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.TRANSIENT


async def test_connection_error_maps_to_transient() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    back = backend(handler)
    result = await back.complete(request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.TRANSIENT


@pytest.mark.parametrize("status", [401, 403, 400, 404])
async def test_client_errors_map_to_fatal(status: int) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "bad"}})

    back = backend(handler)
    result = await back.complete(request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL
    assert result.error.retry_after is None


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
async def test_finish_reason_maps_to_fatal(finish_reason: str) -> None:
    back = backend(lambda req: completion_response("partial", finish_reason=finish_reason))
    result = await back.complete(request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL


async def test_empty_content_maps_to_fatal() -> None:
    back = backend(lambda req: completion_response(""))
    result = await back.complete(request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL


async def test_none_content_maps_to_fatal() -> None:
    back = backend(lambda req: completion_response(None, finish_reason="stop"))
    result = await back.complete(request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL


async def test_bad_json_with_schema_maps_to_fatal() -> None:
    back = backend(lambda req: completion_response("not json"))
    result = await back.complete(request(SCHEMA))
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL
    assert result.error.message


class OpenAICompatHarness:
    def _completion(self, content: str, model: str) -> httpx.Response:
        return completion_response(content, model=model)

    def structured(self, data: Mapping[str, object], model: str) -> LLMBackend:
        def handler(req: httpx.Request) -> httpx.Response:
            return self._completion(json.dumps(dict(data)), model)

        return backend(handler, json_schema_supported=True, model=model)

    def text(self, text: str, model: str) -> LLMBackend:
        def handler(req: httpx.Request) -> httpx.Response:
            return self._completion(text, model)

        return backend(handler, json_schema_supported=True, model=model)

    def failing(self, kind: ErrorKind, retry_after: float | None) -> LLMBackend:
        headers = {} if retry_after is None else {"retry-after": str(retry_after)}
        if kind is ErrorKind.FATAL:

            def handler(req: httpx.Request) -> httpx.Response:
                return httpx.Response(401, json={"error": {"message": "boom"}})
        elif kind is ErrorKind.USAGE_LIMIT:

            def handler(req: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    429, headers=headers, json={"error": {"message": "boom", "code": "insufficient_quota"}}
                )
        else:

            def handler(req: httpx.Request) -> httpx.Response:
                return httpx.Response(500, headers=headers, json={"error": {"message": "boom"}})

        return backend(handler, json_schema_supported=True)
