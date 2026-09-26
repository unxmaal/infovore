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
    }


def seed(db_path: str, content: str = "just chatting, nothing to see") -> int:
    conn = open_database(db_path)
    migrate(conn)
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (1, 1, 9, 1, 'a', '2026-01-01T00:00:00+00:00', ?,"
        " '2026-01-01T00:00:00+00:00', '{}')",
        (content,),
    )
    conn.execute(
        "INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash, extraction_status)"
        " VALUES (1, 1, 1, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 1,"
        " 'quiet_gap', 'h1', 'pending')"
    )
    exchange_id = conn.execute("SELECT id FROM exchanges").fetchone()["id"]
    conn.close()
    return int(exchange_id)


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_triage_is_a_builtin_command() -> None:
    assert "triage" in [command.name for command in builtin_commands()]


def test_triage_scores_pending_exchanges(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["triage"], env)

    assert code == ExitCode.OK
    assert "scored=1" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    row = conn.execute("SELECT triage_version FROM exchanges").fetchone()
    assert row["triage_version"] == "t1"


def test_triage_streams_progress_lines(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["triage"], env)

    assert code == ExitCode.OK
    lines = out.splitlines()
    assert lines[0] == "triage: 1 exchanges to score"
    assert any(line.startswith("exchange 1: score=") for line in lines)


class FlushCountingIO(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes_at: list[int] = []

    def flush(self) -> None:
        self.flushes_at.append(self.getvalue().count("\n"))
        super().flush()


def test_triage_flushes_progress_lines(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out = FlushCountingIO()
    code = main(["triage"], environ=env, dotenv_path=None, stdout=out, stderr=io.StringIO())

    assert code == ExitCode.OK
    assert len(out.flushes_at) >= 3
    assert out.flushes_at == sorted(out.flushes_at)


def test_triage_report_prints_histogram_channels_thresholds_and_reasons(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"], content="Octane needs 6.5.22, check /usr/var/log")

    code, out, _ = run(["triage", "--report"], env)

    assert code == ExitCode.OK
    assert "score histogram:" in out
    assert "channels:" in out
    assert "above threshold:" in out
    assert "below threshold:" in out
    assert "top reasons:" in out


def test_triage_without_report_flag_omits_report(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["triage"], env)

    assert code == ExitCode.OK
    assert "score histogram:" not in out


def test_triage_is_idempotent_over_two_cli_runs(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    run(["triage"], env)

    code, out, _ = run(["triage"], env)

    assert code == ExitCode.OK
    assert "scored=0" in out
