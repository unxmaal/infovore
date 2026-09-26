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
from infovore.config import Settings, Stage, resolve_guild_id
from infovore.extract.llm_extractor import LLMClaimExtractor, LLMNoveltyProbe
from infovore.extract.novelty import run_probe
from infovore.extract.protocol import ClaimExtractor, NoveltyProbe
from infovore.extract.runner import PromptNotPromotedError, run_extraction
from infovore.ingest.live import EventOutcome, handle_event
from infovore.privacy.optout import sync_opt_outs
from infovore.rows import RunMode
from infovore.source.protocol import DiscordSource
from infovore.timing import Clock, Sleeper

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
    extractor: ClaimExtractor,
    probe: NoveltyProbe,
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
    )
    extract_stage = settings.stages[Stage.EXTRACT]
    progress(CycleStepStarted(step="extract"))
    try:
        await run_extraction(
            conn,
            extractor,
            clock,
            sleeper,
            mode=RunMode.LIVE,
            model_label=extract_stage.model,
            batch_size=settings.batch_size,
            max_retries=settings.max_retries,
            concurrency=extract_stage.concurrency,
        )
    except PromptNotPromotedError as error:
        logger.warning(
            "live prompt version %s is not promoted; skipping extraction this cycle",
            error.version,
        )
    probe_stage = settings.stages[Stage.PROBE]
    progress(CycleStepStarted(step="probe"))
    await run_probe(
        conn,
        probe,
        clock,
        sleeper,
        probe_model=None,
        limit=settings.batch_size,
        concurrency=probe_stage.concurrency,
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
    extractor: ClaimExtractor,
    probe: NoveltyProbe,
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
            await _run_cycle(conn, source, extractor, probe, clock, sleeper, settings, progress)
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
    extractor: ClaimExtractor,
    probe: NoveltyProbe,
    clock: Clock,
    sleeper: Sleeper,
    settings: Settings,
    progress: RunProgress = _ignore_run_progress,
) -> RunReport:
    report = RunReport()
    try:
        await _run_cycle(conn, source, extractor, probe, clock, sleeper, settings, progress)
    except Exception:
        logger.exception("periodic cycle failed")
        report.cycles_failed += 1
    else:
        report.cycles_completed += 1
    return report


async def run_forever(
    conn: sqlite3.Connection,
    source: DiscordSource,
    extractor: ClaimExtractor,
    probe: NoveltyProbe,
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
            extractor,
            probe,
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
    help = "live ingest plus a periodic chunk/extract/probe loop, until SIGTERM/SIGINT"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--interval", type=float, default=600.0)
        parser.add_argument("--once", action="store_true")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode, _say, stage_backend

        extract_backend = await stage_backend(context, Stage.EXTRACT)
        probe_backend = await stage_backend(context, Stage.PROBE)
        judge_backend = await stage_backend(context, Stage.JUDGE)
        extractor = LLMClaimExtractor(extract_backend)
        probe = LLMNoveltyProbe(probe_backend, judge_backend)

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
                        extractor,
                        probe,
                        context.clock,
                        context.sleeper,
                        context.settings,
                        progress=report_progress,
                    )
                else:
                    report = await run_forever(
                        context.conn,
                        source,
                        extractor,
                        probe,
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
