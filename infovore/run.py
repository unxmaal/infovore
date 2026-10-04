import argparse
import asyncio
import contextlib
import logging
import signal
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from infovore.chunk.grouper import group_pending
from infovore.chunk.recipe import settings_recipe
from infovore.config import Settings, resolve_guild_id
from infovore.ingest.live import EventOutcome, handle_event
from infovore.privacy.optout import sync_opt_outs
from infovore.source.protocol import DiscordSource
from infovore.timing import Clock, Sleeper
from infovore.triage.embed_stage import default_cache_path
from infovore.triage.incremental import cascade_new_exchanges
from infovore.triage.runner import triage_pending

if TYPE_CHECKING:
    from infovore.cli import AppContext

logger = logging.getLogger(__name__)


@dataclass
class RunReport:
    events_handled: int = 0
    events_failed: int = 0
    events_ignored: int = 0
    cycles_completed: int = 0
    cycles_failed: int = 0


@dataclass(frozen=True)
class CycleStepStarted:
    step: str


RunProgress = Callable[[CycleStepStarted], None]


def _ignore_run_progress(event: CycleStepStarted) -> None:
    return None


async def _consume_events(
    conn: sqlite3.Connection,
    source: DiscordSource,
    clock: Clock,
    include_bots: bool,
    channel_ids: Sequence[int],
    report: RunReport,
) -> None:
    async for event in source.events():
        try:
            outcome = await handle_event(conn, event, clock, include_bots, channel_ids, source)
        except Exception:
            logger.exception("live event handling failed: %r", event)
            report.events_failed += 1
        else:
            if outcome is EventOutcome.CHANNEL_IGNORED:
                report.events_ignored += 1
            else:
                report.events_handled += 1


async def _run_cycle(
    conn: sqlite3.Connection,
    source: DiscordSource,
    clock: Clock,
    sleeper: Sleeper,
    settings: Settings,
    progress: RunProgress = _ignore_run_progress,
) -> None:
    progress(CycleStepStarted(step="sync-optouts"))
    await sync_opt_outs(
        conn,
        source,
        resolve_guild_id(settings, source),
        settings.opt_out_role_name,
        clock,
    )
    progress(CycleStepStarted(step="chunk"))
    group_pending(
        conn,
        clock,
        quiet_gap=timedelta(minutes=settings.quiet_gap_minutes),
        max_messages=settings.exchange_max_messages,
        include_bots=settings.include_bot_messages,
        recipe=settings_recipe(
            timedelta(minutes=settings.quiet_gap_minutes), settings.exchange_max_messages
        ),
    )
    progress(CycleStepStarted(step="triage"))
    triage_pending(conn, rules=settings.triage_rules, workers=settings.workers)
    progress(CycleStepStarted(step="cascade"))
    cascade_new_exchanges(
        conn, settings.exclude_channels, default_cache_path(settings.db_path), clock.now()
    )


async def _interruptible_wait(sleeper: Sleeper, seconds: float, stop: asyncio.Event) -> None:
    sleep_task = asyncio.ensure_future(sleeper.sleep(seconds))
    stop_task = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait({sleep_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (sleep_task, stop_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(sleep_task, stop_task, return_exceptions=True)


async def _cycle_until_stop(
    conn: sqlite3.Connection,
    source: DiscordSource,
    clock: Clock,
    sleeper: Sleeper,
    settings: Settings,
    interval_seconds: float,
    stop: asyncio.Event,
    report: RunReport,
    progress: RunProgress = _ignore_run_progress,
) -> None:
    while not stop.is_set():
        try:
            await _run_cycle(conn, source, clock, sleeper, settings, progress)
        except Exception:
            logger.exception("periodic cycle failed")
            report.cycles_failed += 1
        else:
            report.cycles_completed += 1
        if stop.is_set():
            return
        await _interruptible_wait(sleeper, interval_seconds, stop)


async def run_once(
    conn: sqlite3.Connection,
    source: DiscordSource,
    clock: Clock,
    sleeper: Sleeper,
    settings: Settings,
    progress: RunProgress = _ignore_run_progress,
) -> RunReport:
    report = RunReport()
    try:
        await _run_cycle(conn, source, clock, sleeper, settings, progress)
    except Exception:
        logger.exception("periodic cycle failed")
        report.cycles_failed += 1
    else:
        report.cycles_completed += 1
    return report


async def run_forever(
    conn: sqlite3.Connection,
    source: DiscordSource,
    clock: Clock,
    sleeper: Sleeper,
    settings: Settings,
    *,
    interval_seconds: float,
    stop: asyncio.Event,
    progress: RunProgress = _ignore_run_progress,
) -> RunReport:
    report = RunReport()
    consume_task = asyncio.ensure_future(
        _consume_events(
            conn, source, clock, settings.include_bot_messages, settings.channel_ids, report
        )
    )
    cycle_task = asyncio.ensure_future(
        _cycle_until_stop(
            conn,
            source,
            clock,
            sleeper,
            settings,
            interval_seconds,
            stop,
            report,
            progress,
        )
    )
    await stop.wait()
    await cycle_task
    consume_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consume_task
    return report


def install_stop_handlers(stop: asyncio.Event) -> Sequence[signal.Signals]:
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
        installed.append(sig)
    return installed


def remove_stop_handlers(signals: Sequence[signal.Signals]) -> None:
    loop = asyncio.get_running_loop()
    for sig in signals:
        loop.remove_signal_handler(sig)


class RunCommand:
    name = "run"
    help = "live ingest plus a periodic chunk/triage/cascade loop (no LLM), until SIGTERM/SIGINT"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--interval", type=float, default=600.0)
        parser.add_argument("--once", action="store_true")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode, _say

        def report_progress(event: CycleStepStarted) -> None:
            _say(context.stdout, f"cycle: {event.step}")

        stop = asyncio.Event()
        signals = install_stop_handlers(stop)
        try:
            _say(context.stdout, f"opening {context.settings.source.value} source...")
            async with context.source_factory(context.settings) as source:
                if args.once:
                    report = await run_once(
                        context.conn,
                        source,
                        context.clock,
                        context.sleeper,
                        context.settings,
                        progress=report_progress,
                    )
                else:
                    report = await run_forever(
                        context.conn,
                        source,
                        context.clock,
                        context.sleeper,
                        context.settings,
                        interval_seconds=args.interval,
                        stop=stop,
                        progress=report_progress,
                    )
        finally:
            remove_stop_handlers(signals)

        context.stdout.write(
            f"events_handled={report.events_handled} events_failed={report.events_failed}"
            f" events_ignored={report.events_ignored}"
            f" cycles_completed={report.cycles_completed} cycles_failed={report.cycles_failed}\n"
        )
        return ExitCode.OK
