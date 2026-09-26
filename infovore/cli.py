import argparse
import asyncio
import os
import sqlite3
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, NoReturn, Protocol, TextIO

from infovore.config import ConfigError, Settings, Stage, settings_from_environment
from infovore.db.connection import migrate, open_database
from infovore.db.status import collect_status
from infovore.ingest.backfill import BackfillReport, backfill
from infovore.llm.protocol import LLMBackend
from infovore.llm.registry import Registry, default_registry
from infovore.privacy.optout import sync_opt_outs
from infovore.source.protocol import DiscordSource, SourceUnavailableError
from infovore.timing import AsyncioSleeper, Clock, Sleeper, SystemClock


class ExitCode(IntEnum):
    OK = 0
    FAILURE = 1
    CONFIG = 2
    BACKEND = 3


class BackendUnavailableError(Exception):
    pass


SourceFactory = Callable[[Settings], AbstractAsyncContextManager[DiscordSource]]


@dataclass(frozen=True)
class AppContext:
    settings: Settings
    conn: sqlite3.Connection
    registry: Registry
    clock: Clock
    sleeper: Sleeper
    stdout: TextIO
    source_factory: SourceFactory


class Command(Protocol):
    name: str
    help: str

    def configure(self, parser: argparse.ArgumentParser) -> None: ...

    async def run(self, context: AppContext, args: argparse.Namespace) -> int: ...


def _format_counts(counts: Mapping[str, int]) -> str:
    return " ".join(f"{key}={value}" for key, value in counts.items()) or "none"


def _format_time(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "never"


class StatusCommand:
    name = "status"
    help = "row counts, queues, last runs, and configured backends"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        return None

    async def run(self, context: AppContext, args: argparse.Namespace) -> int:
        report = collect_status(context.conn)
        lines = [
            f"channels: {report.channels}",
            f"messages: {report.messages} (deleted {report.deleted_messages})",
            f"exchanges: {_format_counts(report.exchanges_by_status)}",
            f"claims: {_format_counts(report.claims_by_novelty)}"
            f" (retracted {report.retracted_claims})",
            f"runs: {_format_counts(report.runs_by_outcome)}",
            f"last extraction: {_format_time(report.last_extraction_at)}",
            f"last probe: {_format_time(report.last_probe_at)}",
            f"live prompt version: {report.live_prompt_version or 'none'}",
            *(
                f"{stage.value}: {stage_settings.backend} / {stage_settings.model}"
                for stage, stage_settings in context.settings.stages.items()
            ),
        ]
        context.stdout.write("\n".join(lines) + "\n")
        return ExitCode.OK


class SyncOptOutsCommand:
    name = "sync-optouts"
    help = "sync the opt-out role and redact newly opted-out users' history"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        return None

    async def run(self, context: AppContext, args: argparse.Namespace) -> int:
        async with context.source_factory(context.settings) as source:
            report = await sync_opt_outs(
                context.conn,
                source,
                context.settings.guild_id,
                context.settings.opt_out_role_name,
                context.clock,
            )
        context.stdout.write(
            f"added={len(report.added)} removed={len(report.removed)}"
            f" redacted_messages={report.redacted_messages}"
            f" retracted_claims={len(report.retracted_claims)}\n"
        )
        return ExitCode.OK


def default_source_factory(settings: Settings) -> AbstractAsyncContextManager[DiscordSource]:
    from infovore.source.live import open_discord_source

    return open_discord_source(settings.discord_token)


def _write_backfill_report(stdout: TextIO, report: BackfillReport) -> None:
    for channel_id in sorted(report.channels):
        channel_report = report.channels[channel_id]
        stdout.write(
            f"channel {channel_id}: pages={channel_report.pages}"
            f" inserted={channel_report.inserted} updated={channel_report.updated}"
            f" unchanged={channel_report.unchanged} skipped={channel_report.skipped}\n"
        )
    for failure in report.failed:
        stdout.write(f"channel {failure.channel_id} failed: {failure.reason}\n")


class BackfillCommand:
    name = "backfill"
    help = "walk allowlisted channels' history into raw tables, resuming from checkpoint"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--page-size", type=int, default=100)

    async def run(self, context: AppContext, args: argparse.Namespace) -> int:
        async with context.source_factory(context.settings) as source:
            report = await backfill(
                context.conn,
                source,
                context.settings.guild_id,
                context.settings.channel_ids,
                context.clock,
                context.sleeper,
                context.settings.include_bot_messages,
                page_size=args.page_size,
            )
        _write_backfill_report(context.stdout, report)
        return ExitCode.FAILURE if report.failed else ExitCode.OK


async def stage_backend(context: AppContext, stage: Stage) -> LLMBackend:
    backend = context.registry.build_stage(context.settings, stage)
    health = await context.registry.health_check({stage: backend})
    problem = health[stage]
    if problem is not None:
        raise BackendUnavailableError(f"{stage.value}: {problem}")
    return backend


def builtin_commands() -> list[Command]:
    from infovore.chunk.command import ChunkCommand

    return [StatusCommand(), SyncOptOutsCommand(), ChunkCommand(), BackfillCommand()]


class _Parser(argparse.ArgumentParser):
    def __init__(self, stderr: TextIO, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stderr = stderr

    def error(self, message: str) -> NoReturn:
        self.print_usage(self._stderr)
        self._stderr.write(f"{self.prog}: error: {message}\n")
        raise _UsageError


class _UsageError(Exception):
    pass


def _build_parser(commands: Sequence[Command], stderr: TextIO) -> _Parser:
    parser = _Parser(stderr, prog="infovore")
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    for command in commands:
        subparser = subparsers.add_parser(command.name, help=command.help, stderr=stderr)
        command.configure(subparser)
    return parser


def main(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | str | None = ".env",
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    commands: Sequence[Command] | None = None,
    source_factory: SourceFactory | None = None,
    registry: Registry | None = None,
) -> int:
    available = list(commands) if commands is not None else builtin_commands()
    try:
        args = _build_parser(available, stderr).parse_args(argv)
    except _UsageError:
        return ExitCode.CONFIG
    try:
        settings = settings_from_environment(
            environ if environ is not None else os.environ,
            dotenv_path if dotenv_path is not None else Path(os.devnull),
        )
    except ConfigError as error:
        stderr.write(f"configuration error: {error}\n")
        return ExitCode.CONFIG
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_database(settings.db_path)
    try:
        migrate(conn)
        context = AppContext(
            settings,
            conn,
            registry if registry is not None else default_registry(),
            SystemClock(),
            AsyncioSleeper(),
            stdout,
            source_factory if source_factory is not None else default_source_factory,
        )
        command = next(command for command in available if command.name == args.command)
        return asyncio.run(command.run(context, args))
    except ConfigError as error:
        stderr.write(f"configuration error: {error}\n")
        return ExitCode.CONFIG
    except (BackendUnavailableError, SourceUnavailableError) as error:
        stderr.write(f"backend unavailable: {error}\n")
        return ExitCode.BACKEND
    except Exception as error:
        stderr.write(f"error: {error}\n")
        return ExitCode.FAILURE
    finally:
        conn.close()


def run() -> None:
    sys.exit(main())
