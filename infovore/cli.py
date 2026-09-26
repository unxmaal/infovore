import argparse
import asyncio
import os
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, NoReturn, Protocol, TextIO

from infovore.config import ConfigError, Settings, settings_from_environment
from infovore.db.connection import migrate, open_database
from infovore.db.status import collect_status
from infovore.llm.registry import Registry, default_registry
from infovore.timing import AsyncioSleeper, Clock, Sleeper, SystemClock


class ExitCode(IntEnum):
    OK = 0
    FAILURE = 1
    CONFIG = 2
    BACKEND = 3


class BackendUnavailableError(Exception):
    pass


@dataclass(frozen=True)
class AppContext:
    settings: Settings
    conn: sqlite3.Connection
    registry: Registry
    clock: Clock
    sleeper: Sleeper
    stdout: TextIO


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


def builtin_commands() -> list[Command]:
    from infovore.chunk.command import ChunkCommand

    return [StatusCommand(), ChunkCommand()]


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
            settings, conn, default_registry(), SystemClock(), AsyncioSleeper(), stdout
        )
        command = next(command for command in available if command.name == args.command)
        return asyncio.run(command.run(context, args))
    except BackendUnavailableError as error:
        stderr.write(f"backend unavailable: {error}\n")
        return ExitCode.BACKEND
    except Exception as error:
        stderr.write(f"error: {error}\n")
        return ExitCode.FAILURE
    finally:
        conn.close()


def run() -> None:
    sys.exit(main())
