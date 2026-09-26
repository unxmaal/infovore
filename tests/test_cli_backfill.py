import io
from datetime import UTC, datetime, timedelta
from pathlib import Path

from infovore.cli import ExitCode, main
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


def run(
    argv: list[str], env: dict[str, str], source_factory: object | None = None
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        argv,
        environ=env,
        dotenv_path=None,
        stdout=out,
        stderr=err,
        source_factory=source_factory,  # type: ignore[arg-type]
    )
    return code, out.getvalue(), err.getvalue()


def test_backfill_without_source_factory_is_backend_unavailable(tmp_path: Path) -> None:
    code, _, err = run(["backfill"], environment(tmp_path))
    assert code == ExitCode.BACKEND
    assert "discord source not configured" in err


def test_backfill_reports_success_with_injected_source(tmp_path: Path) -> None:
    source = FakeDiscordSource(
        channels=[make_channel(1)], messages=[make_message(i) for i in range(1, 4)]
    )

    def factory(settings: Settings) -> DiscordSource:
        assert settings.guild_id == GUILD_ID
        return source

    code, out, _ = run(["backfill"], environment(tmp_path), source_factory=factory)
    assert code == ExitCode.OK
    assert "channel 1" in out
    assert "inserted=3" in out


def test_backfill_reports_failure_exit_code_when_a_channel_fails(tmp_path: Path) -> None:
    source = FakeDiscordSource(channels=[make_channel(1)], messages=[make_message(1)])
    for _ in range(5):
        source.fail_next_history_call(SourceUnavailableError("down"), channel_id=1)

    def factory(settings: Settings) -> DiscordSource:
        return source

    code, out, _ = run(["backfill"], environment(tmp_path), source_factory=factory)
    assert code == ExitCode.FAILURE
    assert "channel 1 failed" in out


def test_backfill_page_size_option_is_respected(tmp_path: Path) -> None:
    source = FakeDiscordSource(
        channels=[make_channel(1)], messages=[make_message(i) for i in range(1, 4)]
    )

    def factory(settings: Settings) -> DiscordSource:
        return source

    code, out, _ = run(
        ["backfill", "--page-size", "1"], environment(tmp_path), source_factory=factory
    )
    assert code == ExitCode.OK
    assert "pages=3" in out


def test_builtin_commands_include_backfill() -> None:
    from infovore.cli import builtin_commands

    assert "backfill" in [command.name for command in builtin_commands()]
