from collections.abc import Callable, Sequence

from infovore.llm.protocol import Capabilities, ErrorKind, LLMRequest, LLMResult

DEFAULT_CAPABILITIES = Capabilities(native_json_schema=True, max_concurrency=4)


class FakeBackend:
    def __init__(
        self,
        responder: Callable[[LLMRequest], LLMResult],
        capabilities: Capabilities = DEFAULT_CAPABILITIES,
    ) -> None:
        self._responder = responder
        self._capabilities = capabilities
        self.requests: list[LLMRequest] = []

    @classmethod
    def scripted(
        cls, results: Sequence[LLMResult], capabilities: Capabilities = DEFAULT_CAPABILITIES
    ) -> "FakeBackend":
        remaining = list(results)

        def next_result(_: LLMRequest) -> LLMResult:
            if remaining:
                return remaining.pop(0)
            return LLMResult.failed(ErrorKind.FATAL, "fake script exhausted", None)

        return cls(next_result, capabilities)

    def capabilities(self) -> Capabilities:
        return self._capabilities

    async def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        return self._responder(request)
