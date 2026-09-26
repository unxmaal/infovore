import io
import sqlite3
from pathlib import Path

from infovore.cli import ExitCode, main


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "secret-token",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1,2",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_JUDGE_BACKEND": "fake",
    }


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_snapshot_writes_a_consistent_copy(tmp_path: Path) -> None:
    dest = tmp_path / "out" / "snap.db"
    code, out, _ = run(["snapshot", str(dest)], environment(tmp_path))
    assert code == ExitCode.OK
    assert dest.exists()
    assert f"snapshot: {dest}" in out
    assert "user_version=" in out
    check = sqlite3.connect(dest)
    assert check.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] > 0
    check.close()


def test_snapshot_refuses_to_overwrite_existing_dest_without_force(tmp_path: Path) -> None:
    dest = tmp_path / "snap.db"
    dest.write_bytes(b"not a real snapshot")
    code, out, _ = run(["snapshot", str(dest)], environment(tmp_path))
    assert code == ExitCode.FAILURE
    assert "already exists" in out
    assert dest.read_bytes() == b"not a real snapshot"


def test_snapshot_force_overwrites_existing_dest(tmp_path: Path) -> None:
    dest = tmp_path / "snap.db"
    dest.write_bytes(b"not a real snapshot")
    code, _, _ = run(["snapshot", str(dest), "--force"], environment(tmp_path))
    assert code == ExitCode.OK
    check = sqlite3.connect(dest)
    assert check.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] > 0
    check.close()


def test_builtin_commands_include_snapshot() -> None:
    from infovore.cli import builtin_commands

    assert "snapshot" in [command.name for command in builtin_commands()]
