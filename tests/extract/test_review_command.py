import io
from datetime import UTC, datetime
from pathlib import Path

from infovore.cli import ExitCode, builtin_commands, main
from infovore.db.claims import NewClaim, record_run, register_prompt_version
from infovore.db.connection import migrate, open_database
from infovore.extract.prompt import PROMPT_SHA256, PROMPT_VERSION
from infovore.rows import (
    ClaimKind,
    ExchangeRow,
    ExtractionRunRow,
    ExtractionStatus,
    GroupingRule,
    RunMode,
    RunOutcome,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def environment(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_SCRATCH_DIR": str(tmp_path / "scratch"),
    }
    env.update(overrides)
    return env


def seed_run(db_path: str, prompt_version: str = "v1", message_id: int = 1) -> int:
    conn = open_database(db_path)
    migrate(conn)
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, 1, 9, 1, 'alice', '2026-01-01T00:00:00+00:00', 'x',"
        " '2026-01-01T00:00:00+00:00', '{}')",
        (message_id,),
    )
    exchange = ExchangeRow(
        id=None,
        channel_id=1,
        thread_id=None,
        first_message_id=message_id,
        last_message_id=message_id,
        started_at=NOW,
        ended_at=NOW,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=f"h{message_id}",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.DONE,
        retry_count=0,
        last_error=None,
    )
    from infovore.db.exchanges import insert_exchange

    exchange_id = insert_exchange(conn, exchange, [message_id])
    register_prompt_version(conn, prompt_version, f"sha-{prompt_version}", NOW)
    recorded = record_run(
        conn,
        ExtractionRunRow(
            id=None,
            exchange_id=exchange_id,
            model="m",
            prompt_version=prompt_version,
            started_at=NOW,
            finished_at=NOW,
            input_tokens=1,
            output_tokens=1,
            mode=RunMode.TRIAL,
            outcome=RunOutcome.OK,
            error=None,
        ),
        [
            NewClaim(
                exchange_id=exchange_id,
                statement="needs a jumper on pin 3",
                subject="Octane2",
                kind=ClaimKind.FACT,
                confidence=0.9,
                probe_question="what about it?",
                permalink=f"https://discord.com/channels/9/1/{message_id}",
                supersedes_claim_id=None,
                source_message_ids=(message_id,),
            )
        ],
    )
    conn.close()
    return recorded.run_id


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_review_is_a_builtin_command() -> None:
    assert "review" in [command.name for command in builtin_commands()]


def test_promote_is_a_builtin_command() -> None:
    assert "promote" in [command.name for command in builtin_commands()]


def test_review_command_writes_default_path_and_prints_it(tmp_path: Path) -> None:
    env = environment(tmp_path)
    run_id = seed_run(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["review", "--run-ids", str(run_id)], env)

    assert code == ExitCode.OK
    written_path = Path(out.strip())
    assert written_path == tmp_path / "scratch" / "review.html"
    assert written_path.exists()
    assert "<!DOCTYPE html>" in written_path.read_text()


def test_review_command_writes_to_explicit_out_path_creating_parents(tmp_path: Path) -> None:
    env = environment(tmp_path)
    run_id = seed_run(env["INFOVORE_DB_PATH"])
    out_path = tmp_path / "nested" / "dir" / "report.html"

    code, out, _ = run(["review", "--run-ids", str(run_id), "--out", str(out_path)], env)

    assert code == ExitCode.OK
    assert out.strip() == str(out_path)
    assert out_path.exists()


def test_review_command_unknown_run_id_exits_config(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_run(env["INFOVORE_DB_PATH"])

    code, _, err = run(["review", "--run-ids", "999"], env)

    assert code == ExitCode.CONFIG
    assert "999" in err


def test_review_command_accepts_multiple_run_ids(tmp_path: Path) -> None:
    env = environment(tmp_path)
    run1 = seed_run(env["INFOVORE_DB_PATH"], prompt_version="v1", message_id=1)
    run2 = seed_run(env["INFOVORE_DB_PATH"], prompt_version="v2", message_id=2)

    code, out, _ = run(["review", "--run-ids", str(run1), str(run2)], env)

    assert code == ExitCode.OK
    written_path = Path(out.strip())
    assert "v1" in written_path.read_text()
    assert "v2" in written_path.read_text()


def test_promote_registers_and_promotes_current_prompt_version(tmp_path: Path) -> None:
    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    conn.close()

    code, out, _ = run(["promote", "--prompt-version", PROMPT_VERSION], env)

    assert code == ExitCode.OK
    assert f"live prompt version: {PROMPT_VERSION}" in out


def test_promote_switches_the_live_version_between_two_registered_versions(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    register_prompt_version(conn, "v0", "sha-v0", NOW)
    conn.close()

    code, out, _ = run(["promote", "--prompt-version", "v0"], env)
    assert code == ExitCode.OK
    assert "live prompt version: v0" in out

    code, out, _ = run(["promote", "--prompt-version", PROMPT_VERSION], env)
    assert code == ExitCode.OK
    assert f"live prompt version: {PROMPT_VERSION}" in out


def test_promote_unknown_version_exits_config(tmp_path: Path) -> None:
    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    conn.close()

    code, _, err = run(["promote", "--prompt-version", "does-not-exist"], env)

    assert code == ExitCode.CONFIG
    assert "does-not-exist" in err


def test_promote_does_not_reregister_when_already_registered(tmp_path: Path) -> None:
    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    register_prompt_version(conn, PROMPT_VERSION, PROMPT_SHA256, NOW)
    conn.close()

    code, out, _ = run(["promote", "--prompt-version", PROMPT_VERSION], env)

    assert code == ExitCode.OK
    assert f"live prompt version: {PROMPT_VERSION}" in out
