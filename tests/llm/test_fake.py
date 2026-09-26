from infovore.llm.fake import FakeBackend
from infovore.llm.protocol import Capabilities, ErrorKind, LLMRequest, LLMResult, Usage

REQUEST = LLMRequest(system="sys", prompt="hi", json_schema=None, max_output_tokens=100)


async def test_responder_backend_records_requests_and_answers() -> None:
    backend = FakeBackend(lambda request: LLMResult.ok_text(request.prompt.upper(), "fake-1"))
    result = await backend.complete(REQUEST)
    assert result.text == "HI"
    assert result.model == "fake-1"
    assert backend.requests == [REQUEST]


async def test_scripted_backend_returns_results_in_order_then_fails_fatally() -> None:
    first = LLMResult.ok_structured({"a": 1}, "fake-1", Usage(10, 2, None))
    backend = FakeBackend.scripted([first])
    assert await backend.complete(REQUEST) == first
    exhausted = await backend.complete(REQUEST)
    assert exhausted.error is not None
    assert exhausted.error.kind is ErrorKind.FATAL
    assert "exhausted" in exhausted.error.message


def test_capabilities_default_and_override() -> None:
    assert FakeBackend.scripted([]).capabilities() == Capabilities(
        native_json_schema=True, max_concurrency=4
    )
    custom = Capabilities(native_json_schema=False, max_concurrency=1)
    assert FakeBackend.scripted([], capabilities=custom).capabilities() == custom


def test_result_constructors() -> None:
    failed = LLMResult.failed(ErrorKind.USAGE_LIMIT, "limit", retry_after=30.0)
    assert failed.error is not None
    assert failed.error.retry_after == 30.0
    assert failed.text is None and failed.structured is None and failed.model is None
    assert LLMResult.ok_text("t", "m").usage == Usage(None, None, None)
