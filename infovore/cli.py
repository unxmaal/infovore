import argparse
import asyncio
import contextlib
import os
import sqlite3
import sys
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, NoReturn, Protocol, TextIO

from infovore.config import (
    ConfigError,
    Settings,
    SourceKind,
    Stage,
    resolve_guild_id,
    settings_from_environment,
)
from infovore.db.connection import migrate, open_database
from infovore.db.snapshot import snapshot as snapshot_db
from infovore.db.status import collect_status
from infovore.ingest.backfill import (
    BackfillEvent,
    BackfillReport,
    ChannelFinished,
    ChannelsFound,
    ChannelStarted,
    PageSaved,
    backfill,
)
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


def _format_time(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "never"


class StatusCommand:
    name = "status"
    help = "row counts, queues, last runs, and configured backends"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        return None

    async def run(self, context: AppContext, args: argparse.Namespace) -> int:
        report = collect_status(
            context.conn,
            context.settings.triage_min_score,
            rules=context.settings.triage_rules,
            exclude_channels=context.settings.exclude_channels,
            now=context.clock.now(),
            max_retries=context.settings.max_retries,
        )
        irrelevant = report.irrelevant_denylist + report.irrelevant_short + report.irrelevant_embed
        lines = [
            f"channels: {report.channels}",
            f"messages: {report.messages} (deleted {report.deleted_messages})",
            f"current exchanges: {report.current_exchanges}",
            f"archived: {report.archived_exchanges}"
            f" (cascade relevant {report.cascade_relevant}, residue {report.cascade_residue})",
            f"irrelevant: {irrelevant}"
            f" (denylist {report.irrelevant_denylist}, short no tech {report.irrelevant_short},"
            f" embed {report.irrelevant_embed})",
            f"set aside (no_text): {report.set_aside_no_text}",
            f"residue: {report.cascade_residue}",
            f"not yet cascaded: {report.unscored}",
            f"last cascade: {_format_time(report.last_cascade_at)}",
            f"human labels: relevant {report.human_relevant}, irrelevant {report.human_irrelevant}",
            f"excluded channels: {', '.join(report.excluded_channels) or 'none'}",
            f"extraction (history): done {report.exchanges_by_status.get('done', 0)},"
            f" pending {report.exchanges_by_status.get('pending', 0)}",
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
        _say(context.stdout, f"opening {context.settings.source.value} source...")
        async with context.source_factory(context.settings) as source:
            guild_id = resolve_guild_id(context.settings, source)
            report = await sync_opt_outs(
                context.conn,
                source,
                guild_id,
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
    if settings.source is SourceKind.EXPORT:
        return _open_export_source(settings)
    from infovore.source.live import open_discord_source

    return open_discord_source(settings.discord_token)


@contextlib.asynccontextmanager
async def _open_export_source(settings: Settings) -> AsyncIterator[DiscordSource]:
    if settings.export_dir is None:
        raise ConfigError("INFOVORE_EXPORT_DIR is required")
    from infovore.source.export import ExportDiscordSource

    yield ExportDiscordSource(settings.export_dir)


def _say(stdout: TextIO, line: str) -> None:
    stdout.write(line + "\n")
    stdout.flush()


def _describe_backfill_event(event: BackfillEvent) -> str:
    match event:
        case ChannelsFound(total=total, selected=selected):
            return f"found {total} channels ({selected} selected)"
        case ChannelStarted(channel_id=channel_id, name=name, resume_after=None):
            return f"channel {channel_id} {name}: start"
        case ChannelStarted(channel_id=channel_id, name=name, resume_after=resume_after):
            return f"channel {channel_id} {name}: start (resuming after message {resume_after})"
        case PageSaved():
            return (
                f"channel {event.channel_id}: page +{event.inserted} new,"
                f" {event.updated} updated, {event.unchanged} unchanged"
                f" ({event.messages_total} messages so far)"
            )
        case ChannelFinished(channel_id=channel_id, report=report):
            return f"channel {channel_id}: done ({report.pages} pages, {report.inserted} new)"
        case _:
            return f"channel {event.channel_id}: failed: {event.reason}"


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
        _say(context.stdout, f"opening {context.settings.source.value} source...")
        async with context.source_factory(context.settings) as source:
            guild_id = resolve_guild_id(context.settings, source)
            report = await backfill(
                context.conn,
                source,
                guild_id,
                context.settings.channel_ids,
                context.clock,
                context.sleeper,
                context.settings.include_bot_messages,
                page_size=args.page_size,
                progress=lambda event: _say(context.stdout, _describe_backfill_event(event)),
            )
        _write_backfill_report(context.stdout, report)
        return ExitCode.FAILURE if report.failed else ExitCode.OK


async def stage_backend(context: AppContext, stage: Stage) -> LLMBackend:
    stage_settings = context.settings.stages[stage]
    _say(
        context.stdout,
        f"checking {stage.value} backend ({stage_settings.backend} / {stage_settings.model})...",
    )
    backend = context.registry.build_stage(context.settings, stage)
    health = await context.registry.health_check({stage: backend})
    problem = health[stage]
    if problem is not None:
        raise BackendUnavailableError(f"{stage.value}: {problem}")
    return backend


class SnapshotCommand:
    name = "snapshot"
    migrates = False
    help = "write a consistent copy of the database via the SQLite backup API"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("dest", type=Path)
        parser.add_argument("--force", action="store_true")

    async def run(self, context: AppContext, args: argparse.Namespace) -> int:
        try:
            report = snapshot_db(context.conn, args.dest, force=args.force)
        except FileExistsError as error:
            context.stdout.write(f"snapshot: {error}\n")
            return ExitCode.FAILURE
        context.stdout.write(
            f"snapshot: {report.dest} ({report.size_bytes} bytes,"
            f" user_version={report.user_version})\n"
        )
        return ExitCode.OK


def builtin_commands() -> list[Command]:
    from infovore.chunk.command import ChunkCommand
    from infovore.claims.command import ClaimsCommand
    from infovore.doctor import DoctorCommand
    from infovore.eval.command import JudgeCommand, SliceCommand
    from infovore.extract.command import ExtractCommand
    from infovore.extract.novelty import ProbeCommand
    from infovore.extract.review import PromoteCommand, ReviewCommand
    from infovore.run import RunCommand
    from infovore.search.command import ExportArchiveCommand, SearchCommand
    from infovore.sift.command import SiftCommand
    from infovore.triage.command import TriageCommand
    from infovore.triage.label import LabelCommand
    from infovore.triage.labels_export import LabelsCommand
    from infovore.triage.relevance_command import RelevanceCommand
    from infovore.wiki.command import WikiCommand
    from infovore.words.command import WordsCommand

    return [
        StatusCommand(),
        SearchCommand(),
        SliceCommand(),
        JudgeCommand(),
        ExportArchiveCommand(),
        SyncOptOutsCommand(),
        ChunkCommand(),
        BackfillCommand(),
        SnapshotCommand(),
        ProbeCommand(),
        TriageCommand(),
        ExtractCommand(),
        RunCommand(),
        ReviewCommand(),
        PromoteCommand(),
        LabelCommand(),
        LabelsCommand(),
        RelevanceCommand(),
        SiftCommand(),
        WordsCommand(),
        ClaimsCommand(),
        DoctorCommand(),
        WikiCommand(),
    ]


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
    chosen: Any = next(command for command in available if command.name == args.command)
    if getattr(chosen, "standalone", False):
        return int(
            chosen.run_standalone(args, environ if environ is not None else os.environ, stdout)
        )
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
        command = next(command for command in available if command.name == args.command)
        if getattr(command, "migrates", True):
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
