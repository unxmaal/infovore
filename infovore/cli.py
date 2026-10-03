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


def _format_counts(counts: Mapping[str, int]) -> str:
    return " ".join(f"{key}={value}" for key, value in counts.items()) or "none"


def _format_nested_counts(nested: Mapping[str, Mapping[str, int]]) -> str:
    return (
        " ".join(f"{source}({_format_counts(counts)})" for source, counts in nested.items())
        or "none"
    )


def _format_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.1f}"


def _format_eta(hours: float | None) -> str:
    if hours is None:
        return "unknown (no recent runs)"
    if hours == 0.0:
        return "queue empty"
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def _format_cost(cost: float | None) -> str:
    return "cost unreported" if cost is None else f"${cost:.2f}"


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
            context.settings.triage_min_p_lore,
            rules=context.settings.triage_rules,
            exclude_channels=context.settings.exclude_channels,
            now=context.clock.now(),
            max_retries=context.settings.max_retries,
        )
        lines = [
            f"channels: {report.channels}",
            f"messages: {report.messages} (deleted {report.deleted_messages})",
            f"exchanges: {_format_counts(report.exchanges_by_status)}",
            f"queue: {report.pending_gated} gated"
            f" (of {report.pending_exchanges} claimable before the gate)",
            f"claims: {_format_counts(report.claims_by_novelty)}"
            f" (retracted {report.retracted_claims})",
            f"runs: {_format_counts(report.runs_by_outcome)}",
            f"last extraction: {_format_time(report.last_extraction_at)}",
            f"last probe: {_format_time(report.last_probe_at)}",
            f"live prompt version: {report.live_prompt_version or 'none'}",
            f"triaged: {report.triaged_exchanges}"
            f" (above threshold {report.above_threshold_exchanges})",
            f"labels by source: {_format_nested_counts(report.labels_by_source)}",
            f"labels effective: {_format_counts(report.labels_effective)}",
            "triage model: "
            + (
                f"v{report.latest_model_version} (labels_used={report.latest_model_labels_used})"
                if report.latest_model_version is not None
                else "none"
            ),
            f"p_lore scored: {report.p_lore_scored}",
            f"passing gate: {report.passing_gate}",
            f"excluded by denylist: {report.excluded_by_denylist}",
            f"throughput (last {report.throughput_window_hours}h):"
            f" extract {_format_rate(report.extraction_per_hour)}/h,"
            f" probe {_format_rate(report.probe_per_hour)}/h",
            f"extract eta: {_format_eta(report.extraction_eta_hours)}",
            f"extract spend: {report.extraction_input_tokens} in /"
            f" {report.extraction_output_tokens} out"
            f" / {_format_cost(report.extraction_cost_usd)}",
            f"probe spend: {report.probe_input_tokens} in /"
            f" {report.probe_output_tokens} out"
            f" / {_format_cost(report.probe_cost_usd)}"
            f" over {report.probe_claims} claims",
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
    from infovore.eval.command import JudgeCommand, SliceCommand
    from infovore.extract.command import ExtractCommand
    from infovore.extract.novelty import ProbeCommand
    from infovore.extract.review import PromoteCommand, ReviewCommand
    from infovore.run import RunCommand
    from infovore.search.command import SearchCommand
    from infovore.sift.command import SiftCommand
    from infovore.triage.command import TriageCommand
    from infovore.triage.label import LabelCommand

    return [
        StatusCommand(),
        SearchCommand(),
        SliceCommand(),
        JudgeCommand(),
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
        SiftCommand(),
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
