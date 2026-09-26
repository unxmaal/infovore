import argparse
import asyncio
import io
import logging
import os
import signal
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.cli import AppContext, ExitCode, builtin_commands, main
from infovore.config import Settings, Stage, StageSettings
from infovore.db.claims import promote_prompt_version, register_prompt_version
from infovore.db.connection import migrate, open_database
from infovore.extract.fake import MarkerExtractor, MarkerProbe
from infovore.extract.prompt import PROMPT_SHA256, PROMPT_VERSION
from infovore.llm.fake import FakeBackend
from infovore.llm.protocol import ErrorKind, LLMBackend, LLMRequest, LLMResult
from infovore.llm.registry import Registry, default_registry
from infovore.run import (
    CycleStepStarted,
    RunCommand,
    install_stop_handlers,
    remove_stop_handlers,
    run_forever,
    run_once,
)
from infovore.source.fake import FakeDiscordSource
from infovore.source.protocol import (
    DiscordSource,
    MessageCreated,
    SourceChannel,
    SourceEvent,
    SourceMessage,
)
from infovore.timing import FixedClock, RecordingSleeper

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "infovore.db")
    migrate(conn)
    return conn


def make_settings(tmp_path: Path) -> Settings:
    stage = StageSettings(backend="fake", model="m", concurrency=2, timeout_seconds=60.0)
    return Settings(
        discord_token="t",
        guild_id=9,
        channel_ids=(1,),
        db_path=tmp_path / "infovore.db",
        scratch_dir=tmp_path / "scratch",
        stages={Stage.EXTRACT: stage, Stage.PROBE: stage, Stage.JUDGE: stage},
        triage_min_score=0.0,
    )


def make_message(
    msg_id: int, created_at: datetime, content: str, channel_id: int = 1
) -> SourceMessage:
    return SourceMessage(
        id=msg_id,
        channel_id=channel_id,
        guild_id=9,
        author_id=5,
        author_name="alice",
        author_is_bot=False,
        is_system=False,
        created_at=created_at,
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        attachments=(),
        reactions=(),
        raw={},
    )


class _UnroutableEvent:
    pass


class StoppingSource:
    def __init__(self, inner: DiscordSource, stop: asyncio.Event, after_calls: int = 1) -> None:
        self._inner = inner
        self._stop = stop
        self._after_calls = after_calls
        self._calls = 0

    async def list_channels(self, guild_id: int) -> Sequence[SourceChannel]:
        return await self._inner.list_channels(guild_id)

    def history(
        self, channel_id: int, after_id: int | None, page_size: int
    ) -> AsyncIterator[Sequence[SourceMessage]]:
        return self._inner.history(channel_id, after_id, page_size)

    def events(self) -> AsyncIterator[SourceEvent]:
        return self._inner.events()

    async def role_member_ids(self, guild_id: int, role_name: str) -> frozenset[int]:
        result = await self._inner.role_member_ids(guild_id, role_name)
        self._calls += 1
        if self._calls >= self._after_calls:
            self._stop.set()
        return result


class ExplodingSource:
    def __init__(self, inner: DiscordSource, stop: asyncio.Event) -> None:
        self._inner = inner
        self._stop = stop

    async def list_channels(self, guild_id: int) -> Sequence[SourceChannel]:
        return await self._inner.list_channels(guild_id)

    def history(
        self, channel_id: int, after_id: int | None, page_size: int
    ) -> AsyncIterator[Sequence[SourceMessage]]:
        return self._inner.history(channel_id, after_id, page_size)

    def events(self) -> AsyncIterator[SourceEvent]:
        return self._inner.events()

    async def role_member_ids(self, guild_id: int, role_name: str) -> frozenset[int]:
        self._stop.set()
        raise RuntimeError("role lookup exploded")


class BlockingSleeper:
    def __init__(self) -> None:
        self.slept: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        await asyncio.sleep(100000)


async def test_run_forever_ingests_groups_extracts_and_probes_then_stops(tmp_path: Path) -> None:
    conn = db(tmp_path)
    register_prompt_version(conn, PROMPT_VERSION, PROMPT_SHA256, NOW)
    promote_prompt_version(conn, PROMPT_VERSION, NOW)
    settings = make_settings(tmp_path)
    fake_source = FakeDiscordSource()
    old = NOW - timedelta(minutes=40)
    fake_source.push(MessageCreated(make_message(1, old, "FACT: Octane2 :: needs a jumper")))
    fake_source.close()
    stop = asyncio.Event()
    source = StoppingSource(fake_source, stop)

    report = await asyncio.wait_for(
        run_forever(
            conn,
            source,
            MarkerExtractor(),
            MarkerProbe(),
            FixedClock(NOW),
            RecordingSleeper(),
            settings,
            interval_seconds=1000.0,
            stop=stop,
        ),
        timeout=2.0,
    )

    assert report.events_handled == 1
    assert report.events_failed == 0
    assert report.cycles_completed == 1
    assert report.cycles_failed == 0

    assert conn.execute("SELECT id FROM messages WHERE id = 1").fetchone() is not None
    exchange = conn.execute("SELECT extraction_status FROM exchanges").fetchone()
    assert exchange is not None
    assert exchange["extraction_status"] == "done"
    claim = conn.execute("SELECT novelty, probe_model FROM claims").fetchone()
    assert claim is not None
    assert claim["novelty"] == "unknown"
    assert claim["probe_model"] == "fake-probe"


async def test_run_forever_skips_extraction_when_prompt_not_promoted_but_still_groups(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    conn = db(tmp_path)
    settings = make_settings(tmp_path)
    fake_source = FakeDiscordSource()
    old = NOW - timedelta(minutes=40)
    fake_source.push(MessageCreated(make_message(1, old, "FACT: Octane2 :: needs a jumper")))
    fake_source.close()
    stop = asyncio.Event()
    source = StoppingSource(fake_source, stop)

    with caplog.at_level(logging.WARNING):
        report = await asyncio.wait_for(
            run_forever(
                conn,
                source,
                MarkerExtractor(),
                MarkerProbe(),
                FixedClock(NOW),
                RecordingSleeper(),
                settings,
                interval_seconds=1000.0,
                stop=stop,
            ),
            timeout=2.0,
        )

    assert report.cycles_completed == 1
    assert report.cycles_failed == 0
    exchange = conn.execute("SELECT extraction_status FROM exchanges").fetchone()
    assert exchange is not None
    assert exchange["extraction_status"] == "pending"
    assert conn.execute("SELECT COUNT(*) AS c FROM claims").fetchone()["c"] == 0
    assert any("not promoted" in message for message in caplog.messages)


async def test_run_forever_counts_a_failing_event_and_continues(tmp_path: Path) -> None:
    conn = db(tmp_path)
    settings = make_settings(tmp_path)
    fake_source = FakeDiscordSource()
    fake_source.push(_UnroutableEvent())  # type: ignore[arg-type]
    fake_source.push(MessageCreated(make_message(1, NOW, "hello")))
    fake_source.close()
    stop = asyncio.Event()
    source = StoppingSource(fake_source, stop)

    report = await asyncio.wait_for(
        run_forever(
            conn,
            source,
            MarkerExtractor(),
            MarkerProbe(),
            FixedClock(NOW),
            RecordingSleeper(),
            settings,
            interval_seconds=1000.0,
            stop=stop,
        ),
        timeout=2.0,
    )

    assert report.events_failed == 1
    assert report.events_handled == 1
    assert conn.execute("SELECT id FROM messages WHERE id = 1").fetchone() is not None


async def test_run_forever_counts_an_ignored_event_and_continues(tmp_path: Path) -> None:
    conn = db(tmp_path)
    settings = make_settings(tmp_path)
    fake_source = FakeDiscordSource()
    fake_source.push(MessageCreated(make_message(1, NOW, "hello", channel_id=2)))
    fake_source.push(MessageCreated(make_message(2, NOW, "hello", channel_id=1)))
    fake_source.close()
    stop = asyncio.Event()
    source = StoppingSource(fake_source, stop)

    report = await asyncio.wait_for(
        run_forever(
            conn,
            source,
            MarkerExtractor(),
            MarkerProbe(),
            FixedClock(NOW),
            RecordingSleeper(),
            settings,
            interval_seconds=1000.0,
            stop=stop,
        ),
        timeout=2.0,
    )

    assert report.events_ignored == 1
    assert report.events_handled == 1
    assert report.events_failed == 0
    assert conn.execute("SELECT id FROM messages WHERE id = 1").fetchone() is None
    assert conn.execute("SELECT id FROM messages WHERE id = 2").fetchone() is not None


async def test_run_forever_counts_a_failing_cycle_step_and_continues(tmp_path: Path) -> None:
    conn = db(tmp_path)
    settings = make_settings(tmp_path)
    stop = asyncio.Event()
    source = ExplodingSource(FakeDiscordSource(), stop)

    report = await asyncio.wait_for(
        run_forever(
            conn,
            source,
            MarkerExtractor(),
            MarkerProbe(),
            FixedClock(NOW),
            RecordingSleeper(),
            settings,
            interval_seconds=1000.0,
            stop=stop,
        ),
        timeout=2.0,
    )

    assert report.cycles_failed == 1
    assert report.cycles_completed == 0


async def test_run_forever_multiple_cycles_then_stop_between_intervals(tmp_path: Path) -> None:
    conn = db(tmp_path)
    settings = make_settings(tmp_path)
    stop = asyncio.Event()
    source = StoppingSource(FakeDiscordSource(), stop, after_calls=2)

    report = await asyncio.wait_for(
        run_forever(
            conn,
            source,
            MarkerExtractor(),
            MarkerProbe(),
            FixedClock(NOW),
            RecordingSleeper(),
            settings,
            interval_seconds=1000.0,
            stop=stop,
        ),
        timeout=2.0,
    )

    assert report.cycles_completed == 2
    assert report.cycles_failed == 0


async def test_stop_during_interval_wait_returns_promptly(tmp_path: Path) -> None:
    conn = db(tmp_path)
    settings = make_settings(tmp_path)
    stop = asyncio.Event()
    source = FakeDiscordSource()
    sleeper = BlockingSleeper()

    task = asyncio.ensure_future(
        run_forever(
            conn,
            source,
            MarkerExtractor(),
            MarkerProbe(),
            FixedClock(NOW),
            sleeper,
            settings,
            interval_seconds=100000.0,
            stop=stop,
        )
    )
    for _ in range(50):
        await asyncio.sleep(0)
    assert sleeper.slept == [100000.0]
    stop.set()
    report = await asyncio.wait_for(task, timeout=2.0)

    assert report.cycles_completed == 1
    assert report.cycles_failed == 0


async def test_run_once_emits_cycle_step_started_events_in_order(tmp_path: Path) -> None:
    conn = db(tmp_path)
    register_prompt_version(conn, PROMPT_VERSION, PROMPT_SHA256, NOW)
    promote_prompt_version(conn, PROMPT_VERSION, NOW)
    settings = make_settings(tmp_path)
    source = FakeDiscordSource()
    events: list[CycleStepStarted] = []

    report = await run_once(
        conn,
        source,
        MarkerExtractor(),
        MarkerProbe(),
        FixedClock(NOW),
        RecordingSleeper(),
        settings,
        progress=events.append,
    )

    assert report.cycles_completed == 1
    assert events == [
        CycleStepStarted(step="sync-optouts"),
        CycleStepStarted(step="chunk"),
        CycleStepStarted(step="triage"),
        CycleStepStarted(step="extract"),
        CycleStepStarted(step="probe"),
    ]


async def test_run_once_progress_defaults_to_noop(tmp_path: Path) -> None:
    conn = db(tmp_path)
    settings = make_settings(tmp_path)
    source = FakeDiscordSource()

    report = await run_once(
        conn,
        source,
        MarkerExtractor(),
        MarkerProbe(),
        FixedClock(NOW),
        RecordingSleeper(),
        settings,
    )
    assert report.cycles_completed == 1


async def test_install_stop_handlers_sets_stop_on_sigterm_and_sigint() -> None:
    stop = asyncio.Event()
    signals = install_stop_handlers(stop)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(10):
            await asyncio.sleep(0)
        assert stop.is_set()
    finally:
        remove_stop_handlers(signals)


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_EXTRACT_BACKEND": "fake",
        "INFOVORE_PROBE_BACKEND": "fake",
        "INFOVORE_JUDGE_BACKEND": "fake",
    }


def run_cli(
    argv: list[str],
    env: dict[str, str],
    source_factory: object = None,
    registry: Registry | None = None,
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        argv,
        environ=env,
        dotenv_path=None,
        stdout=out,
        stderr=err,
        source_factory=source_factory,  # type: ignore[arg-type]
        registry=registry,
    )
    return code, out.getvalue(), err.getvalue()


class UnhealthyFactory:
    name = "unhealthy"

    def validate(self, settings: StageSettings) -> list[str]:
        return []

    def build(self, settings: StageSettings) -> LLMBackend:
        from infovore.llm.fake import FakeBackend

        return FakeBackend.scripted([LLMResult.failed(ErrorKind.FATAL, "not logged in", None)])


def test_run_is_a_builtin_command() -> None:
    assert "run" in [command.name for command in builtin_commands()]


def test_run_command_configure_defaults() -> None:
    parser = argparse.ArgumentParser()
    RunCommand().configure(parser)
    args = parser.parse_args([])
    assert args.interval == 600.0
    assert args.once is False


def test_run_command_once_flag_runs_one_cycle_and_exits_ok_and_closes_source(
    tmp_path: Path,
) -> None:
    fake_source = FakeDiscordSource()
    exited: list[bool] = []

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[DiscordSource]:
        yield fake_source
        exited.append(True)

    code, out, _ = run_cli(["run", "--once"], environment(tmp_path), source_factory=factory)

    assert code == ExitCode.OK
    assert "cycles_completed=1" in out
    assert "cycles_failed=0" in out
    assert "events_ignored=0" in out
    assert exited == [True]


def test_run_command_once_flag_counts_a_failing_cycle(tmp_path: Path) -> None:
    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[DiscordSource]:
        yield ExplodingSource(FakeDiscordSource(), asyncio.Event())

    code, out, _ = run_cli(["run", "--once"], environment(tmp_path), source_factory=factory)

    assert code == ExitCode.OK
    assert "cycles_failed=1" in out
    assert "cycles_completed=0" in out


def test_run_command_health_check_failure_exits_backend(tmp_path: Path) -> None:
    env = environment(tmp_path)
    env["INFOVORE_EXTRACT_BACKEND"] = "unhealthy"
    registry = Registry()
    registry.register(UnhealthyFactory())

    code, _, _ = run_cli(["run", "--once"], env, registry=registry)

    assert code == ExitCode.BACKEND


async def test_run_command_continuous_mode_stops_on_signal_and_closes_source(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    settings = make_settings(tmp_path)
    exited: list[bool] = []
    fake_source = FakeDiscordSource()

    @asynccontextmanager
    async def factory(app_settings: Settings) -> AsyncIterator[DiscordSource]:
        yield fake_source
        exited.append(True)

    stdout = io.StringIO()
    context = AppContext(
        settings,
        conn,
        default_registry(),
        FixedClock(NOW),
        RecordingSleeper(),
        stdout,
        factory,
    )
    args = argparse.Namespace(interval=100000.0, once=False)

    command_task = asyncio.ensure_future(RunCommand().run(context, args))
    for _ in range(20):
        await asyncio.sleep(0)
    os.kill(os.getpid(), signal.SIGTERM)
    code = await asyncio.wait_for(command_task, timeout=2.0)

    assert code == ExitCode.OK
    assert exited == [True]
    assert "cycles_failed=0" in stdout.getvalue()


class FlushCountingIO(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes_at: list[int] = []

    def flush(self) -> None:
        self.flushes_at.append(self.getvalue().count("\n"))
        super().flush()


def test_run_command_once_flag_streams_flushed_progress_lines(tmp_path: Path) -> None:
    fake_source = FakeDiscordSource()

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[DiscordSource]:
        yield fake_source

    out = FlushCountingIO()
    err = io.StringIO()
    code = main(
        ["run", "--once"],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=err,
        source_factory=factory,
    )
    assert code == ExitCode.OK
    lines = out.getvalue().splitlines()
    assert lines[0] == "checking extract backend (fake / sonnet)..."
    assert lines[1] == "checking probe backend (fake / sonnet)..."
    assert lines[2] == "checking judge backend (fake / haiku)..."
    assert lines[3] == "opening discord source..."
    assert lines[4] == "cycle: sync-optouts"
    assert lines[5] == "cycle: chunk"
    assert lines[6] == "cycle: triage"
    assert lines[7] == "cycle: extract"
    assert lines[8] == "cycle: probe"
    assert out.flushes_at[:9] == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_run_command_writes_opening_line_before_connecting_to_source(tmp_path: Path) -> None:
    fake_source = FakeDiscordSource()
    out = io.StringIO()
    seen_before_connect: list[bool] = []

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[DiscordSource]:
        seen_before_connect.append("opening discord source..." in out.getvalue())
        yield fake_source

    code = main(
        ["run", "--once"],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=io.StringIO(),
        source_factory=factory,
    )
    assert code == ExitCode.OK
    assert seen_before_connect == [True]


def test_run_command_prints_checking_backend_lines_before_each_stage_health_check(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    env["INFOVORE_EXTRACT_BACKEND"] = "spy"
    env["INFOVORE_PROBE_BACKEND"] = "spy"
    env["INFOVORE_JUDGE_BACKEND"] = "spy"
    fake_source = FakeDiscordSource()

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[DiscordSource]:
        yield fake_source

    out = io.StringIO()
    snapshots: list[str] = []

    def responder(request: LLMRequest) -> LLMResult:
        snapshots.append(out.getvalue())
        return LLMResult.ok_text("pong", "m")

    class SpyFactory:
        name = "spy"

        def validate(self, settings: object) -> list[str]:
            return []

        def build(self, settings: object) -> LLMBackend:
            return FakeBackend(responder)

    registry = Registry()
    registry.register(SpyFactory())

    code = main(
        ["run", "--once"],
        environ=env,
        dotenv_path=None,
        stdout=out,
        stderr=io.StringIO(),
        source_factory=factory,
        registry=registry,
    )

    assert code == ExitCode.OK
    assert "checking extract backend (spy / sonnet)..." in snapshots[0]
    assert "checking probe backend (spy / sonnet)..." in snapshots[1]
    assert "checking judge backend (spy / haiku)..." in snapshots[2]
