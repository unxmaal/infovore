import argparse
import io
from pathlib import Path

import pytest

from infovore.cli import AppContext, ExitCode, main, stage_backend
from infovore.config import ConfigError, Stage, StageSettings, load_settings
from infovore.llm.fake import FakeBackend
from infovore.llm.protocol import ErrorKind, LLMBackend, LLMResult
from infovore.llm.registry import Registry


def environment(tmp_path: Path, backend: str = "scripted") -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_EXTRACT_BACKEND": backend,
    }


class ScriptedFactory:
    name = "scripted"

    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy
        self.built: list[StageSettings] = []

    def validate(self, settings: StageSettings) -> list[str]:
        return []

    def build(self, settings: StageSettings) -> LLMBackend:
        self.built.append(settings)
        if self.healthy:
            return FakeBackend.scripted([LLMResult.ok_text("pong", "scripted-1")])
        return FakeBackend.scripted([LLMResult.failed(ErrorKind.FATAL, "not logged in", None)])


class UsesExtractBackend:
    name = "needs-extract"
    help = "test"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        return None

    async def run(self, context: AppContext, args: argparse.Namespace) -> int:
        backend = await stage_backend(context, Stage.EXTRACT)
        context.stdout.write(f"got {type(backend).__name__}\n")
        return ExitCode.OK


def run(env: dict[str, str], registry: Registry) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["needs-extract"],
        environ=env,
        dotenv_path=None,
        stdout=out,
        stderr=err,
        commands=[UsesExtractBackend()],
        registry=registry,
    )
    return code, out.getvalue(), err.getvalue()


def registry_with(factory: ScriptedFactory) -> Registry:
    registry = Registry()
    registry.register(factory)
    return registry


def test_healthy_stage_backend_is_built_from_that_stages_settings(tmp_path: Path) -> None:
    factory = ScriptedFactory(healthy=True)
    code, out, _ = run(environment(tmp_path), registry_with(factory))
    assert code == ExitCode.OK
    assert "got FakeBackend" in out
    assert [settings.backend for settings in factory.built] == ["scripted"]


def test_unhealthy_stage_backend_exits_backend_code(tmp_path: Path) -> None:
    code, _, err = run(environment(tmp_path), registry_with(ScriptedFactory(healthy=False)))
    assert code == ExitCode.BACKEND
    assert "extract: not logged in" in err


def test_unknown_stage_backend_exits_config_code(tmp_path: Path) -> None:
    code, _, err = run(environment(tmp_path, backend="nope"), Registry())
    assert code == ExitCode.CONFIG
    assert "extract: unknown backend 'nope'" in err


def test_build_stage_validates_only_the_requested_stage(tmp_path: Path) -> None:
    settings = load_settings(environment(tmp_path))
    registry = registry_with(ScriptedFactory(healthy=True))
    assert isinstance(registry.build_stage(settings, Stage.EXTRACT), FakeBackend)
    with pytest.raises(ConfigError, match="probe: unknown backend 'claude_cli'"):
        registry.build_stage(settings, Stage.PROBE)


class RejectingFactory(ScriptedFactory):
    def validate(self, settings: StageSettings) -> list[str]:
        return ["model is required"]


def test_build_stage_reports_factory_validation_errors(tmp_path: Path) -> None:
    settings = load_settings(environment(tmp_path))
    with pytest.raises(ConfigError, match="extract: model is required"):
        registry_with(RejectingFactory(healthy=True)).build_stage(settings, Stage.EXTRACT)
