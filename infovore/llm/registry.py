from collections.abc import Mapping
from typing import Protocol

from infovore.config import ConfigError, Settings, Stage, StageSettings
from infovore.llm.claude_cli import ClaudeCliFactory
from infovore.llm.fake import FakeBackend
from infovore.llm.openai_compat import OpenAICompatFactory
from infovore.llm.protocol import LLMBackend, LLMRequest, LLMResult


class BackendFactory(Protocol):
    name: str

    def validate(self, settings: StageSettings) -> list[str]: ...

    def build(self, settings: StageSettings) -> LLMBackend: ...


class FakeBackendFactory:
    name = "fake"

    def validate(self, settings: StageSettings) -> list[str]:
        return []

    def build(self, settings: StageSettings) -> LLMBackend:
        model = settings.model

        def responder(request: LLMRequest) -> LLMResult:
            return LLMResult.ok_text("ok", model)

        return FakeBackend(responder)


class Registry:
    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, factory: BackendFactory) -> None:
        self._factories[factory.name] = factory

    def validate(self, settings: Settings) -> list[str]:
        errors: list[str] = []
        for stage, stage_settings in settings.stages.items():
            factory = self._factories.get(stage_settings.backend)
            if factory is None:
                errors.append(f"{stage.value}: unknown backend '{stage_settings.backend}'")
                continue
            errors.extend(
                f"{stage.value}: {message}" for message in factory.validate(stage_settings)
            )
        return errors

    def build_backends(self, settings: Settings) -> dict[Stage, LLMBackend]:
        errors = self.validate(settings)
        if errors:
            raise ConfigError("; ".join(errors))
        return {
            stage: self._factories[stage_settings.backend].build(stage_settings)
            for stage, stage_settings in settings.stages.items()
        }

    async def health_check(self, backends: Mapping[Stage, LLMBackend]) -> dict[Stage, str | None]:
        results: dict[Stage, str | None] = {}
        for stage, backend in backends.items():
            request = LLMRequest(
                system="health check", prompt="ping", json_schema=None, max_output_tokens=16
            )
            result = await backend.complete(request)
            results[stage] = result.error.message if result.error is not None else None
        return results


def default_registry() -> Registry:
    registry = Registry()
    registry.register(FakeBackendFactory())
    registry.register(ClaudeCliFactory())
    registry.register(OpenAICompatFactory())
    return registry
