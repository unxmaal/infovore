import io
from pathlib import Path

from infovore.cli import ExitCode, builtin_commands, main
from infovore.db.connection import migrate, open_database


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_QUIET_GAP_MINUTES": "30",
    }


def seed(db_path: str) -> None:
    conn = open_database(db_path)
    migrate(conn)
    rows = [
        (1, "2026-01-01T00:00:00+00:00"),
        (2, "2026-01-01T00:05:00+00:00"),
        (3, "2026-01-01T03:00:00+00:00"),
    ]
    for message_id, created in rows:
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 9, 1, 'a', ?, 'x', ?, '{}')",
            (message_id, created, created),
        )
    conn.close()


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_chunk_is_a_builtin_command() -> None:
    assert "chunk" in [command.name for command in builtin_commands()]


def test_chunk_groups_closed_exchanges_and_defers_open_ones(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    code, out, _ = run(["chunk", "--now", "2026-01-01T03:10:00+00:00"], env)
    assert code == ExitCode.OK
    assert "exchanges created: 1" in out
    assert "messages grouped: 2" in out
    assert "groups deferred: 1" in out


def test_chunk_uses_the_system_clock_by_default(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    code, out, _ = run(["chunk"], env)
    assert code == ExitCode.OK
    assert "exchanges created: 2" in out
    assert "groups deferred: 0" in out


def test_invalid_now_is_a_usage_error(tmp_path: Path) -> None:
    code, _, err = run(["chunk", "--now", "yesterday"], environment(tmp_path))
    assert code == ExitCode.CONFIG
    assert "--now" in err


def test_naive_now_is_a_usage_error(tmp_path: Path) -> None:
    code, _, err = run(["chunk", "--now", "2026-01-01T00:00:00"], environment(tmp_path))
    assert code == ExitCode.CONFIG
    assert "timezone" in err
