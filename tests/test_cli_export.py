import io
import json
from pathlib import Path

from infovore.cli import ExitCode, default_source_factory, main
from infovore.config import Settings, SourceKind, Stage, StageSettings
from infovore.source.export import ExportDiscordSource

GUILD_ID = 900
CHANNEL_ID = 100
BASE_TIMESTAMP = "2023-08-01T12:00:00.0000000+00:00"


def make_stage() -> StageSettings:
    return StageSettings(backend="fake", model="m", concurrency=2, timeout_seconds=30.0)


def make_message(message_id: int) -> dict[str, object]:
    return {
        "id": str(message_id),
        "type": "Default",
        "timestamp": f"2023-08-01T12:00:{message_id:02d}.0000000+00:00",
        "timestampEdited": None,
        "callEndedTimestamp": None,
        "isPinned": False,
        "content": "hello",
        "author": {
            "id": "1",
            "name": "alice",
            "discriminator": "0000",
            "nickname": "alice",
            "color": None,
            "isBot": False,
            "roles": [],
            "avatarUrl": "",
        },
        "attachments": [],
        "embeds": [],
        "stickers": [],
        "reactions": [],
        "mentions": [],
        "inlineEmojis": [],
    }


def write_export(
    path: Path, guild_id: int = GUILD_ID, channel_id: int = CHANNEL_ID, count: int = 3
) -> None:
    export = {
        "guild": {"id": str(guild_id), "name": "Guild", "iconUrl": ""},
        "channel": {
            "id": str(channel_id),
            "type": "GuildTextChat",
            "categoryId": None,
            "category": None,
            "name": "general",
            "topic": None,
        },
        "dateRange": {"after": None, "before": None},
        "exportedAt": BASE_TIMESTAMP,
        "messages": [make_message(i) for i in range(1, count + 1)],
        "messageCount": count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export))


def environment(tmp_path: Path, export_dir: Path, **overrides: str) -> dict[str, str]:
    env = {
        "INFOVORE_SOURCE": "export",
        "INFOVORE_EXPORT_DIR": str(export_dir),
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_JUDGE_BACKEND": "fake",
    }
    env.update(overrides)
    return env


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


async def test_default_source_factory_dispatches_to_export_source(tmp_path: Path) -> None:
    write_export(tmp_path / "general.json")
    settings = Settings(
        db_path=tmp_path / "x.db",
        scratch_dir=tmp_path / "scratch",
        stages={stage: make_stage() for stage in Stage},
        source=SourceKind.EXPORT,
        export_dir=tmp_path,
    )
    async with default_source_factory(settings) as source:
        assert isinstance(source, ExportDiscordSource)


async def test_default_source_factory_export_without_dir_raises_config_error(
    tmp_path: Path,
) -> None:
    from infovore.config import ConfigError

    settings = Settings(
        db_path=tmp_path / "x.db",
        scratch_dir=tmp_path / "scratch",
        stages={stage: make_stage() for stage in Stage},
        source=SourceKind.EXPORT,
        export_dir=None,
    )
    raised = False
    try:
        async with default_source_factory(settings):
            pass
    except ConfigError:
        raised = True
    assert raised


def test_backfill_end_to_end_against_real_export_source_infers_guild_id(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "export"
    write_export(export_dir / "general.json", count=5)
    env = environment(tmp_path, export_dir)

    code, out, _ = run(["backfill"], env)
    assert code == ExitCode.OK
    assert f"channel {CHANNEL_ID}" in out
    assert "inserted=5" in out

    code, out, _ = run(["backfill"], env)
    assert code == ExitCode.OK
    assert "inserted=0" in out
    assert "unchanged=5" in out


def test_backfill_multiple_guilds_with_no_guild_id_is_config_error(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    write_export(export_dir / "a.json", guild_id=1, channel_id=10)
    write_export(export_dir / "b.json", guild_id=2, channel_id=20)
    env = environment(tmp_path, export_dir)

    code, _, err = run(["backfill"], env)
    assert code == ExitCode.CONFIG
    assert "INFOVORE_GUILD_ID" in err


def test_backfill_empty_channel_ids_walks_every_channel_in_export(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    write_export(export_dir / "a.json", channel_id=10, count=2)
    write_export(export_dir / "b.json", channel_id=20, count=3)
    env = environment(tmp_path, export_dir)

    code, out, _ = run(["backfill"], env)
    assert code == ExitCode.OK
    assert "channel 10" in out
    assert "channel 20" in out


def test_sync_optouts_against_real_export_source_infers_guild_id(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    write_export(export_dir / "general.json")
    env = environment(tmp_path, export_dir)

    code, out, _ = run(["sync-optouts"], env)
    assert code == ExitCode.OK
    assert "added=0" in out
