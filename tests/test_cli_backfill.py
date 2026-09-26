import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from infovore.cli import ExitCode, SourceFactory, main
from infovore.config import Settings
from infovore.rows import ChannelKind
from infovore.source.fake import FakeDiscordSource
from infovore.source.protocol import (
    DiscordSource,
    SourceChannel,
    SourceMessage,
    SourceUnavailableError,
)

GUILD_ID = 9
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "secret-token",
        "INFOVORE_GUILD_ID": str(GUILD_ID),
        "INFOVORE_CHANNEL_IDS": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_JUDGE_BACKEND": "fake",
    }


def make_channel(channel_id: int) -> SourceChannel:
    return SourceChannel(
        channel_id, GUILD_ID, None, f"channel-{channel_id}", ChannelKind.TEXT, False
    )


def make_message(msg_id: int, channel_id: int = 1) -> SourceMessage:
    return SourceMessage(
        id=msg_id,
        channel_id=channel_id,
        guild_id=GUILD_ID,
        author_id=5,
        author_name="alice",
        author_is_bot=False,
        is_system=False,
        created_at=BASE + timedelta(minutes=msg_id),
        edited_at=None,
        content="hi",
        reply_to_id=None,
        thread_id=None,
        attachments=(),
        reactions=(),
        raw={},
    )


def serving(source: DiscordSource) -> SourceFactory:
    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[DiscordSource]:
        assert settings.guild_id == GUILD_ID
        yield source

    return factory


def run(
    argv: list[str], env: dict[str, str], source_factory: SourceFactory | None = None
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        argv,
        environ=env,
        dotenv_path=None,
        stdout=out,
        stderr=err,
        source_factory=source_factory,
    )
    return code, out.getvalue(), err.getvalue()


def test_backfill_with_unavailable_source_is_backend_unavailable(tmp_path: Path) -> None:
    @asynccontextmanager
    async def unavailable(settings: Settings) -> AsyncIterator[DiscordSource]:
        raise SourceUnavailableError("discord client not ready within 60.0s")
        yield

    code, _, err = run(["backfill"], environment(tmp_path), source_factory=unavailable)
    assert code == ExitCode.BACKEND
    assert "not ready" in err


def test_backfill_reports_success_with_injected_source(tmp_path: Path) -> None:
    source = FakeDiscordSource(
        channels=[make_channel(1)], messages=[make_message(i) for i in range(1, 4)]
    )

    code, out, _ = run(["backfill"], environment(tmp_path), source_factory=serving(source))
    assert code == ExitCode.OK
    assert "channel 1" in out
    assert "inserted=3" in out


def test_backfill_reports_failure_exit_code_when_a_channel_fails(tmp_path: Path) -> None:
    source = FakeDiscordSource(channels=[make_channel(1)], messages=[make_message(1)])
    for _ in range(5):
        source.fail_next_history_call(SourceUnavailableError("down"), channel_id=1)

    code, out, _ = run(["backfill"], environment(tmp_path), source_factory=serving(source))
    assert code == ExitCode.FAILURE
    assert "channel 1 failed" in out


def test_backfill_page_size_option_is_respected(tmp_path: Path) -> None:
    source = FakeDiscordSource(
        channels=[make_channel(1)], messages=[make_message(i) for i in range(1, 4)]
    )

    code, out, _ = run(
        ["backfill", "--page-size", "1"], environment(tmp_path), source_factory=serving(source)
    )
    assert code == ExitCode.OK
    assert "pages=3" in out


def test_builtin_commands_include_backfill() -> None:
    from infovore.cli import builtin_commands

    assert "backfill" in [command.name for command in builtin_commands()]


class FlushCountingIO(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes_at: list[int] = []

    def flush(self) -> None:
        self.flushes_at.append(self.getvalue().count("\n"))
        super().flush()


def test_backfill_streams_flushed_progress_lines(tmp_path: Path) -> None:
    source = FakeDiscordSource(
        channels=[make_channel(1)], messages=[make_message(i) for i in range(1, 3)]
    )
    out, err = FlushCountingIO(), io.StringIO()
    code = main(
        ["backfill", "--page-size", "1"],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=err,
        source_factory=serving(source),
    )
    assert code == ExitCode.OK
    lines = out.getvalue().splitlines()
    assert lines[0] == "opening discord source..."
    assert lines[1] == "found 1 channels (1 selected)"
    assert lines[2] == "channel 1 channel-1: start"
    assert lines[3] == "channel 1: page +1 new, 0 updated, 0 unchanged (1 messages so far)"
    assert lines[4] == "channel 1: page +1 new, 0 updated, 0 unchanged (2 messages so far)"
    assert lines[5] == "channel 1: done (2 pages, 2 new)"
    assert out.flushes_at[:6] == [1, 2, 3, 4, 5, 6]
