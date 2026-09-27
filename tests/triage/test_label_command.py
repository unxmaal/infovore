import io
from datetime import UTC, datetime
from pathlib import Path

from infovore.cli import ExitCode, builtin_commands, main
from infovore.db.claims import (
    NewClaim,
    record_run,
    register_prompt_version,
    set_batch_id,
    set_novelty,
)
from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database
from infovore.rows import ClaimKind, ExtractionRunRow, Novelty, RunMode, RunOutcome

NOW_TEXT = to_db_time(datetime(2026, 1, 1, tzinfo=UTC))


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
    }


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def seed_exchange(db_path: str, exchange_id: int = 1, message_id: int = 1) -> None:
    conn = open_database(db_path)
    migrate(conn)
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, 1, 9, 1, 'a', ?, 'x', ?, '{}')",
        (message_id, NOW_TEXT, NOW_TEXT),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, 1, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
        (exchange_id, message_id, message_id, NOW_TEXT, NOW_TEXT, f"h{exchange_id}"),
    )
    conn.close()


def seed_run(
    db_path: str, exchange_id: int = 1, message_id: int = 1, verdict: Novelty | None = None
) -> int:
    seed_exchange(db_path, exchange_id=exchange_id, message_id=message_id)
    conn = open_database(db_path)
    migrate(conn)
    register_prompt_version(conn, "v1", "sha", datetime(2026, 1, 1, tzinfo=UTC))
    claims = [
        NewClaim(
            exchange_id=exchange_id,
            statement="s",
            subject="subj",
            kind=ClaimKind.FACT,
            confidence=0.9,
            probe_question="q?",
            permalink="https://discord.com/channels/1/1/1",
            supersedes_claim_id=None,
            source_message_ids=(message_id,),
        )
    ]
    result = record_run(
        conn,
        ExtractionRunRow(
            id=None,
            exchange_id=exchange_id,
            model="m",
            prompt_version="v1",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            finished_at=datetime(2026, 1, 1, tzinfo=UTC),
            input_tokens=1,
            output_tokens=1,
            mode=RunMode.TRIAL,
            outcome=RunOutcome.OK,
            error=None,
        ),
        claims,
    )
    if verdict is not None:
        for claim_id in result.claim_ids:
            set_novelty(
                conn, claim_id, verdict, "probe-model", "answer", datetime(2026, 1, 1, tzinfo=UTC)
            )
    conn.close()
    return result.run_id


def test_label_is_a_builtin_command() -> None:
    assert "label" in [command.name for command in builtin_commands()]


def test_label_requires_from_runs_or_exchange_id(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_exchange(env["INFOVORE_DB_PATH"])
    code, _, err = run(["label"], env)
    assert code == ExitCode.CONFIG
    assert "--from-runs" in err


def test_label_from_runs_and_exchange_id_are_mutually_exclusive(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_exchange(env["INFOVORE_DB_PATH"])
    code, _, err = run(["label", "--from-runs", "1", "--exchange-id", "1", "--lore"], env)
    assert code == ExitCode.CONFIG
    assert "mutually exclusive" in err


def test_label_lore_and_noise_are_mutually_exclusive(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_exchange(env["INFOVORE_DB_PATH"])
    code, _, err = run(["label", "--exchange-id", "1", "--lore", "--noise"], env)
    assert code == ExitCode.CONFIG
    assert "not allowed" in err


def test_label_exchange_id_requires_lore_or_noise(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_exchange(env["INFOVORE_DB_PATH"])
    code, _, err = run(["label", "--exchange-id", "1"], env)
    assert code == ExitCode.CONFIG
    assert "--lore" in err


def test_label_from_runs_rejects_lore_and_noise_flags(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_exchange(env["INFOVORE_DB_PATH"])
    code, _, err = run(["label", "--from-runs", "1", "--lore"], env)
    assert code == ExitCode.CONFIG
    assert "--exchange-id" in err


def test_label_human_lore(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_exchange(env["INFOVORE_DB_PATH"])
    code, out, _ = run(["label", "--exchange-id", "1", "--lore"], env)
    assert code == ExitCode.OK
    assert "exchange 1 as lore (human)" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    row = conn.execute("SELECT label, source FROM exchange_labels WHERE exchange_id = 1").fetchone()
    assert (row["label"], row["source"]) == ("lore", "human")


def test_label_human_noise(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_exchange(env["INFOVORE_DB_PATH"])
    code, out, _ = run(["label", "--exchange-id", "1", "--noise"], env)
    assert code == ExitCode.OK
    assert "exchange 1 as noise (human)" in out


def test_label_human_unknown_exchange_fails(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_exchange(env["INFOVORE_DB_PATH"])
    code, _, err = run(["label", "--exchange-id", "999", "--lore"], env)
    assert code == ExitCode.FAILURE
    assert "FOREIGN KEY" in err


def test_label_from_runs_derives_and_prints_progress_and_summary(tmp_path: Path) -> None:
    env = environment(tmp_path)
    lore_run = seed_run(
        env["INFOVORE_DB_PATH"], exchange_id=1, message_id=1, verdict=Novelty.UNKNOWN
    )
    noise_run = seed_run(
        env["INFOVORE_DB_PATH"], exchange_id=2, message_id=2, verdict=Novelty.KNOWN
    )
    failed_run = seed_run(env["INFOVORE_DB_PATH"], exchange_id=3, message_id=3, verdict=None)
    conn = open_database(env["INFOVORE_DB_PATH"])
    conn.execute("UPDATE extraction_runs SET outcome = 'failed' WHERE id = ?", (failed_run,))
    conn.close()

    code, out, _ = run(
        ["label", "--from-runs", str(lore_run), str(noise_run), str(failed_run)], env
    )
    assert code == ExitCode.OK
    assert f"run {lore_run}: lore" in out
    assert f"run {noise_run}: noise" in out
    assert f"run {failed_run}: skipped (failed run)" in out
    assert "labeled: lore=1 noise=1 skipped=1" in out
    assert f"{failed_run}:failed run" in out


def test_label_from_runs_summary_omits_skip_detail_when_nothing_was_skipped(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    lore_run = seed_run(
        env["INFOVORE_DB_PATH"], exchange_id=1, message_id=1, verdict=Novelty.UNKNOWN
    )
    code, out, _ = run(["label", "--from-runs", str(lore_run)], env)
    assert code == ExitCode.OK
    assert "labeled: lore=1 noise=0 skipped=0\n" in out


def test_label_from_runs_unknown_run_id_exits_config(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_exchange(env["INFOVORE_DB_PATH"])
    code, _, err = run(["label", "--from-runs", "999"], env)
    assert code == ExitCode.CONFIG
    assert "999" in err


def test_label_from_runs_accepts_range_syntax(tmp_path: Path) -> None:
    env = environment(tmp_path)
    lore_run = seed_run(
        env["INFOVORE_DB_PATH"], exchange_id=1, message_id=1, verdict=Novelty.UNKNOWN
    )
    noise_run = seed_run(
        env["INFOVORE_DB_PATH"], exchange_id=2, message_id=2, verdict=Novelty.KNOWN
    )
    code, out, _ = run(["label", "--from-runs", f"{lore_run}-{noise_run}"], env)
    assert code == ExitCode.OK
    assert "labeled: lore=1 noise=1 skipped=0" in out


def test_label_from_runs_bare_flag_uses_latest_trial_batch(tmp_path: Path) -> None:
    env = environment(tmp_path)
    old_run = seed_run(
        env["INFOVORE_DB_PATH"], exchange_id=1, message_id=1, verdict=Novelty.UNKNOWN
    )
    new_run = seed_run(
        env["INFOVORE_DB_PATH"], exchange_id=2, message_id=2, verdict=Novelty.UNKNOWN
    )
    conn = open_database(env["INFOVORE_DB_PATH"])
    set_batch_id(conn, [old_run], "2026-01-01T00:00:00+00:00")
    set_batch_id(conn, [new_run], "2026-01-02T00:00:00+00:00")
    conn.close()

    code, out, _ = run(["label", "--from-runs"], env)

    assert code == ExitCode.OK
    assert f"run {new_run}: lore" in out
    assert f"run {old_run}" not in out


def test_label_from_runs_bare_flag_without_a_trial_batch_exits_config(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_exchange(env["INFOVORE_DB_PATH"])
    code, _, err = run(["label", "--from-runs"], env)
    assert code == ExitCode.CONFIG
    assert "extract --mode trial" in err


def test_label_from_runs_rejects_an_invalid_selector_token(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_exchange(env["INFOVORE_DB_PATH"])
    code, _, err = run(["label", "--from-runs", "abc"], env)
    assert code == ExitCode.CONFIG
    assert "abc" in err
