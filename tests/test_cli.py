import argparse
import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from infovore import cli
from infovore.cli import (
    AppContext,
    BackendUnavailableError,
    Command,
    ExitCode,
    SourceFactory,
    main,
)
from infovore.config import Settings
from infovore.source.protocol import DiscordSource, SourceUnavailableError


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "secret-token",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1,2",
        "INFOVORE_DB_PATH": str(tmp_path / "nested" / "dir" / "infovore.db"),
        "INFOVORE_JUDGE_BACKEND": "fake",
    }


def run(
    argv: list[str],
    env: dict[str, str],
    commands: list[Command] | None = None,
    source_factory: SourceFactory | None = None,
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        argv,
        environ=env,
        dotenv_path=None,
        stdout=out,
        stderr=err,
        commands=commands,
        source_factory=source_factory,
    )
    return code, out.getvalue(), err.getvalue()


def test_status_on_a_fresh_database_reports_zeros(tmp_path: Path) -> None:
    code, out, _ = run(["status"], environment(tmp_path))
    assert code == ExitCode.OK
    assert "messages: 0" in out
    assert "exchanges: none" in out
    assert "claims: none" in out
    assert "last extraction: never" in out
    assert "live prompt version: none" in out
    assert "extract: claude_cli / sonnet" in out
    assert "judge: fake / haiku" in out
    assert "secret-token" not in out
    assert "triaged: 0 (above threshold 0)" in out
    assert (tmp_path / "nested" / "dir" / "infovore.db").exists()


def test_status_lists_non_empty_counts(tmp_path: Path) -> None:
    env = environment(tmp_path)
    assert run(["status"], env)[0] == ExitCode.OK
    from infovore.db.connection import open_database

    conn = open_database(env["INFOVORE_DB_PATH"])
    now = "2026-01-01T00:00:00+00:00"
    conn.executescript(
        f"""
        INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,
          created_at, content, ingested_at, raw_json)
          VALUES (1, 1, 9, 1, 'a', '{now}', 'x', '{now}', '{{}}');
        INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,
          ended_at, message_count, grouping_rule, content_hash)
          VALUES (1, 1, 1, '{now}', '{now}', 1, 'quiet_gap', 'a');
        INSERT INTO prompt_versions VALUES ('v1', 'sha', '{now}', '{now}');
        INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at, mode, outcome)
          VALUES (1, 'm', 'v1', '{now}', 'live', 'ok');
        INSERT INTO claims (exchange_id, extraction_run_id, statement, subject, kind, confidence,
          probe_question, permalink, novelty, probed_at)
          VALUES (1, 1, 's', 'subj', 'fact', 0.5, 'q?', 'p', 'unknown', '{now}');
        """
    )
    code, out, _ = run(["status"], env)
    assert code == ExitCode.OK
    assert "exchanges: pending=1" in out
    assert "claims: unknown=1" in out
    assert "runs: ok=1" in out
    assert "last extraction: 2026-01-01T00:00:00+00:00" in out
    assert "last probe: 2026-01-01T00:00:00+00:00" in out
    assert "live prompt version: v1" in out


def test_status_shows_triaged_and_above_threshold_counts(tmp_path: Path) -> None:
    env = environment(tmp_path)
    env["INFOVORE_TRIAGE_MIN_SCORE"] = "0.3"
    assert run(["status"], env)[0] == ExitCode.OK
    from infovore.db.connection import open_database

    conn = open_database(env["INFOVORE_DB_PATH"])
    now = "2026-01-01T00:00:00+00:00"
    conn.executescript(
        f"""
        INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,
          created_at, content, ingested_at, raw_json)
          VALUES (1, 1, 9, 1, 'a', '{now}', 'x', '{now}', '{{}}'),
                 (2, 1, 9, 1, 'a', '{now}', 'y', '{now}', '{{}}');
        INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,
          ended_at, message_count, grouping_rule, content_hash, triage_score, triage_version)
          VALUES (1, 1, 1, '{now}', '{now}', 1, 'quiet_gap', 'a', 0.9, 't1'),
                 (1, 2, 2, '{now}', '{now}', 1, 'quiet_gap', 'b', 0.1, 't1');
        """
    )
    code, out, _ = run(["status"], env)
    assert code == ExitCode.OK
    assert "triaged: 2 (above threshold 1)" in out


def test_missing_configuration_exits_with_config_code(tmp_path: Path) -> None:
    code, _, err = run(["status"], {})
    assert code == ExitCode.CONFIG
    assert "INFOVORE_DISCORD_TOKEN" in err


def test_no_subcommand_prints_usage_and_exits_with_config_code(tmp_path: Path) -> None:
    code, _, err = run([], environment(tmp_path))
    assert code == ExitCode.CONFIG
    assert "usage" in err


def test_unknown_subcommand_exits_with_config_code(tmp_path: Path) -> None:
    code, _, err = run(["nope"], environment(tmp_path))
    assert code == ExitCode.CONFIG
    assert "invalid choice" in err


class ExplodingCommand:
    name = "explode"
    help = "raises"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--flag", action="store_true")

    async def run(self, context: AppContext, args: argparse.Namespace) -> int:
        assert args.flag
        assert context.settings.guild_id == 9
        raise self.error


def test_unexpected_errors_exit_with_failure_code(tmp_path: Path) -> None:
    code, _, err = run(
        ["explode", "--flag"],
        environment(tmp_path),
        commands=[ExplodingCommand(RuntimeError("kaboom"))],
    )
    assert code == ExitCode.FAILURE
    assert "kaboom" in err


def test_backend_unavailable_exits_with_backend_code(tmp_path: Path) -> None:
    code, _, err = run(
        ["explode", "--flag"],
        environment(tmp_path),
        commands=[ExplodingCommand(BackendUnavailableError("extract: claude not logged in"))],
    )
    assert code == ExitCode.BACKEND
    assert "claude not logged in" in err


def test_dotenv_file_is_read_when_given(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("\n".join(f"{k}={v}" for k, v in environment(tmp_path).items()))
    out, err = io.StringIO(), io.StringIO()
    assert main(["status"], environ={}, dotenv_path=dotenv, stdout=out, stderr=err) == ExitCode.OK


def test_console_entrypoint_exits_with_mains_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["infovore", "status"])
    for key, value in environment(tmp_path).items():
        monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == ExitCode.OK


def test_builtin_commands_include_status() -> None:
    assert "status" in [command.name for command in cli.builtin_commands()]


def test_builtin_commands_include_sync_optouts() -> None:
    assert "sync-optouts" in [command.name for command in cli.builtin_commands()]


def test_sync_optouts_with_unavailable_source_exits_backend_code(tmp_path: Path) -> None:
    @asynccontextmanager
    async def unavailable(settings: Settings) -> AsyncIterator[DiscordSource]:
        raise SourceUnavailableError("discord login failed: LoginFailure")
        yield

    code, _, err = run(["sync-optouts"], environment(tmp_path), source_factory=unavailable)
    assert code == ExitCode.BACKEND
    assert "discord login failed" in err
    assert "secret-token" not in err


def test_default_source_factory_builds_a_context_manager_without_connecting(
    tmp_path: Path,
) -> None:
    from infovore.config import load_settings

    manager = cli.default_source_factory(load_settings(environment(tmp_path)))
    assert hasattr(manager, "__aenter__")
    assert hasattr(manager, "__aexit__")


def test_sync_optouts_command_reports_results(tmp_path: Path) -> None:
    from infovore.source.fake import FakeDiscordSource

    fake_source = FakeDiscordSource(role_members={9: {"no-archive": [11]}})
    exited: list[bool] = []

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[DiscordSource]:
        assert settings.guild_id == 9
        yield fake_source
        exited.append(True)

    code, out, _ = run(["sync-optouts"], environment(tmp_path), source_factory=factory)
    assert code == ExitCode.OK
    assert "added=1" in out
    assert "removed=0" in out
    assert "redacted_messages=0" in out
    assert "retracted_claims=0" in out
    assert exited == [True]


class FlushCountingIO(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes_at: list[int] = []

    def flush(self) -> None:
        self.flushes_at.append(self.getvalue().count("\n"))
        super().flush()


def test_sync_optouts_streams_flushed_opening_line(tmp_path: Path) -> None:
    from infovore.source.fake import FakeDiscordSource

    fake_source = FakeDiscordSource(role_members={9: {"no-archive": [11]}})

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[DiscordSource]:
        yield fake_source

    out = FlushCountingIO()
    err = io.StringIO()
    code = main(
        ["sync-optouts"],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=err,
        source_factory=factory,
    )
    assert code == ExitCode.OK
    lines = out.getvalue().splitlines()
    assert lines[0] == "opening discord source..."
    assert out.flushes_at[0] == 1


def test_sync_optouts_writes_opening_line_before_connecting_to_source(tmp_path: Path) -> None:
    from infovore.source.fake import FakeDiscordSource

    fake_source = FakeDiscordSource(role_members={9: {"no-archive": [11]}})
    out = io.StringIO()
    seen_before_connect: list[bool] = []

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[DiscordSource]:
        seen_before_connect.append("opening discord source..." in out.getvalue())
        yield fake_source

    code = main(
        ["sync-optouts"],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=io.StringIO(),
        source_factory=factory,
    )
    assert code == ExitCode.OK
    assert seen_before_connect == [True]
