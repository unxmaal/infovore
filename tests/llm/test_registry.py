from pathlib import Path

import pytest

from infovore.config import ConfigError, Settings, Stage, StageSettings
from infovore.llm.claude_cli import ClaudeCliBackend
from infovore.llm.protocol import Capabilities, ErrorKind, LLMBackend, LLMRequest, LLMResult
from infovore.llm.registry import BackendFactory, FakeBackendFactory, Registry, default_registry


def stage_settings(**overrides: object) -> StageSettings:
    defaults: dict[str, object] = {
        "backend": "fake",
        "model": "sonnet",
        "concurrency": 2,
        "timeout_seconds": 30.0,
    }
    defaults.update(overrides)
    return StageSettings(**defaults)  # type: ignore[arg-type]


def settings_with(stages: dict[Stage, StageSettings]) -> Settings:
    return Settings(
        discord_token="tok",
        guild_id=1,
        channel_ids=(1,),
        db_path=Path("db.sqlite"),
        scratch_dir=Path("scratch"),
        stages=stages,
    )


class BrokenBackendFactory:
    name = "broken"

    def validate(self, settings: StageSettings) -> list[str]:
        return [] if settings.model else ["model is required"]

    def build(self, settings: StageSettings) -> LLMBackend:
        raise AssertionError("build should not be called when validation fails")


def test_fake_backend_factory_name() -> None:
    assert FakeBackendFactory().name == "fake"


def test_fake_backend_factory_validate_is_always_clean() -> None:
    assert FakeBackendFactory().validate(stage_settings()) == []


async def test_fake_backend_factory_build_answers_with_model_name() -> None:
    backend = FakeBackendFactory().build(stage_settings(model="haiku"))
    request = LLMRequest(system="s", prompt="p", json_schema=None, max_output_tokens=8)
    result = await backend.complete(request)
    assert result.error is None
    assert result.text == "ok"
    assert result.model == "haiku"


async def test_fake_backend_factory_build_ignores_request_contents() -> None:
    backend = FakeBackendFactory().build(stage_settings(model="sonnet"))
    first = await backend.complete(
        LLMRequest(system="a", prompt="one", json_schema=None, max_output_tokens=1)
    )
    second = await backend.complete(
        LLMRequest(system="b", prompt="two", json_schema={"type": "object"}, max_output_tokens=2)
    )
    assert first.text == second.text == "ok"


def test_registry_validate_reports_unknown_backend() -> None:
    registry = Registry()
    registry.register(FakeBackendFactory())
    settings = settings_with(
        {
            Stage.EXTRACT: stage_settings(backend="nope"),
            Stage.PROBE: stage_settings(backend="fake"),
            Stage.JUDGE: stage_settings(backend="fake"),
        }
    )
    errors = registry.validate(settings)
    assert len(errors) == 1
    assert "extract" in errors[0]
    assert "nope" in errors[0]


def test_registry_validate_delegates_to_factory() -> None:
    registry = Registry()
    registry.register(BrokenBackendFactory())
    settings = settings_with(
        {
            Stage.EXTRACT: stage_settings(backend="broken", model=""),
            Stage.PROBE: stage_settings(backend="broken", model="m"),
            Stage.JUDGE: stage_settings(backend="broken", model="m"),
        }
    )
    errors = registry.validate(settings)
    assert len(errors) == 1
    assert "extract" in errors[0]


def test_registry_validate_clean_returns_no_errors() -> None:
    registry = Registry()
    registry.register(FakeBackendFactory())
    settings = settings_with({stage: stage_settings() for stage in Stage})
    assert registry.validate(settings) == []


def test_registry_build_backends_returns_one_per_stage() -> None:
    registry = Registry()
    registry.register(FakeBackendFactory())
    settings = settings_with({stage: stage_settings(model=stage.value) for stage in Stage})
    backends = registry.build_backends(settings)
    assert set(backends) == set(Stage)


def test_registry_build_backends_raises_config_error_on_unknown_backend() -> None:
    registry = Registry()
    registry.register(FakeBackendFactory())
    settings = settings_with(
        {
            Stage.EXTRACT: stage_settings(backend="nope"),
            Stage.PROBE: stage_settings(),
            Stage.JUDGE: stage_settings(),
        }
    )
    with pytest.raises(ConfigError, match="nope"):
        registry.build_backends(settings)


def test_default_registry_builds_fake_backends() -> None:
    registry = default_registry()
    settings = settings_with({stage: stage_settings(model=stage.value) for stage in Stage})
    backends = registry.build_backends(settings)
    assert set(backends) == set(Stage)


def test_default_registry_builds_claude_cli_backend() -> None:
    registry = default_registry()
    settings = settings_with(
        {stage: stage_settings(backend="claude_cli", model=stage.value) for stage in Stage}
    )
    backends = registry.build_backends(settings)
    assert isinstance(backends[Stage.EXTRACT], ClaudeCliBackend)


def test_default_registry_builds_openai_compat_backends() -> None:
    registry = default_registry()
    settings = settings_with(
        {
            stage: stage_settings(
                backend="openai_compat",
                model=stage.value,
                options={"base_url": "http://localhost:11434/v1", "api_key": "k"},
            )
            for stage in Stage
        }
    )
    backends = registry.build_backends(settings)
    assert set(backends) == set(Stage)


def test_default_registry_reports_missing_openai_compat_options() -> None:
    registry = default_registry()
    settings = settings_with({stage: stage_settings(backend="openai_compat") for stage in Stage})
    errors = registry.validate(settings)
    assert len(errors) == 6
    assert all(
        "base_url is required" in error or "api_key is required" in error for error in errors
    )


async def test_health_check_reports_none_when_backend_is_healthy() -> None:
    registry = default_registry()
    settings = settings_with({stage: stage_settings() for stage in Stage})
    backends = registry.build_backends(settings)
    results = await registry.health_check(backends)
    assert results == {stage: None for stage in Stage}


async def test_health_check_reports_error_message() -> None:
    class FailingBackend:
        def capabilities(self) -> Capabilities:
            return Capabilities(native_json_schema=False, max_concurrency=1)

        async def complete(self, request: LLMRequest) -> LLMResult:
            return LLMResult.failed(ErrorKind.FATAL, "backend is down", None)

    registry = Registry()
    results = await registry.health_check({Stage.EXTRACT: FailingBackend()})
    assert results == {Stage.EXTRACT: "backend is down"}


def test_backend_factory_is_a_protocol_with_expected_shape() -> None:
    factory: BackendFactory = FakeBackendFactory()
    assert factory.name == "fake"
