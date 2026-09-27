import io
import sqlite3
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


LORE_CONTENT = "PROM 6.5.22 part 030-1234-001 /usr/sbin/inst"
NOISE_CONTENT = "lol gg no cap"


def _insert_labeled_exchange(
    conn: sqlite3.Connection, message_id: int, channel_id: int, content: str, label: str
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 9, 1, 'a', '2026-01-01T00:00:00+00:00', ?,"
        " '2026-01-01T00:00:00+00:00', '{}')",
        (message_id, channel_id, content),
    )
    content_hash = f"h{message_id}"
    conn.execute(
        "INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash, extraction_status)"
        " VALUES (?, ?, ?, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 1,"
        " 'quiet_gap', ?, 'pending')",
        (channel_id, message_id, message_id, content_hash),
    )
    exchange_id = conn.execute(
        "SELECT id FROM exchanges WHERE content_hash = ?", (content_hash,)
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO exchange_labels (exchange_id, label, source, source_ref, labeled_at)"
        " VALUES (?, ?, 'human', NULL, '2026-01-01T00:00:00+00:00')",
        (exchange_id, label),
    )


def seed_labeled(db_path: str, lore_count: int, noise_count: int) -> None:
    conn = open_database(db_path)
    migrate(conn)
    message_id = 1
    for _ in range(lore_count):
        _insert_labeled_exchange(conn, message_id, channel_id=1, content=LORE_CONTENT, label="lore")
        message_id += 1
    for _ in range(noise_count):
        _insert_labeled_exchange(
            conn, message_id, channel_id=2, content=NOISE_CONTENT, label="noise"
        )
        message_id += 1
    conn.close()


def test_triage_train_refuses_with_insufficient_labels(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 3, 3)

    code, _, err = run(["triage", "--train"], env)

    assert code == ExitCode.CONFIG
    assert "10" in err


def test_triage_train_prints_report_and_scores_every_exchange(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)

    code, out, _ = run(["triage", "--train"], env)

    assert code == ExitCode.OK
    assert "trained model v" in out
    assert "threshold" in out
    assert "confusion" in out
    assert "top tokens:" in out
    assert "scored 40 exchanges" in out

    conn = open_database(env["INFOVORE_DB_PATH"])
    rows = conn.execute("SELECT p_lore, p_lore_model FROM exchanges").fetchall()
    assert len(rows) == 40
    assert all(row["p_lore"] is not None for row in rows)


def test_triage_recommend_threshold_without_a_model_exits_config_error(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)

    code, _, err = run(["triage", "--recommend-threshold"], env)

    assert code == ExitCode.CONFIG
    assert "triage --train" in err


def test_triage_recommend_threshold_prints_recommendation(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    code, out, _ = run(["triage", "--recommend-threshold", "--min-recall", "0.5"], env)

    assert code == ExitCode.OK
    assert "INFOVORE_TRIAGE_MIN_P_LORE=" in out
    assert "expected share" in out


def test_triage_recommend_threshold_with_unreachable_recall_reports_none(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    code, out, _ = run(["triage", "--recommend-threshold", "--min-recall", "1.01"], env)

    assert code == ExitCode.OK
    assert "no threshold meets recall" in out


def test_triage_recommend_threshold_uses_precision_beyond_the_fixed_grid(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    code, out, _ = run(["triage", "--recommend-threshold", "--min-recall", "1.0"], env)

    assert code == ExitCode.OK
    line = next(line for line in out.splitlines() if line.startswith("recommended"))
    value_str = line.split("=", 1)[1].split(" ", 1)[0]
    assert float(value_str) > 0.9
    assert value_str not in {f"{i / 10:.1f}" for i in range(1, 10)}


def test_triage_recommend_threshold_prints_a_recall_target_table(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    code, out, _ = run(["triage", "--recommend-threshold", "--min-recall", "0.6"], env)

    assert code == ExitCode.OK
    for target in ("0.95", "0.90", "0.80", "0.70", "0.50"):
        assert target in out
