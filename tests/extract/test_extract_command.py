import io
from datetime import UTC, datetime
from pathlib import Path

from infovore.cli import ExitCode, builtin_commands, main
from infovore.config import StageSettings
from infovore.db.claims import promote_prompt_version, register_prompt_version
from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import get_exchange, insert_exchange
from infovore.extract.prompt import PROMPT_SHA256, PROMPT_VERSION
from infovore.llm.fake import FakeBackend
from infovore.llm.protocol import ErrorKind, LLMBackend, LLMRequest, LLMResult
from infovore.llm.registry import Registry
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule

NOW = datetime(2026, 1, 1, tzinfo=UTC)

VALID_EXTRACTION_OUT = {
    "claims": [
        {
            "statement": "needs a jumper on pin 3",
            "subject": "Octane2",
            "kind": "fact",
            "confidence": 0.9,
            "probe_question": "what does the octane2 need on its board?",
            "source_message_ids": [1],
            "supersedes": None,
        }
    ]
}

HEALTH_OK = LLMResult.ok_text("pong", "scripted-model")


def success_results() -> list[LLMResult]:
    return [HEALTH_OK, LLMResult.ok_structured(VALID_EXTRACTION_OUT, "scripted-model")]


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_EXTRACT_BACKEND": "scripted",
    }


class ScriptedFactory:
    name = "scripted"

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results

    def validate(self, settings: StageSettings) -> list[str]:
        return []

    def build(self, settings: StageSettings) -> LLMBackend:
        return FakeBackend.scripted(self._results)


def registry_with(results: list[LLMResult]) -> Registry:
    registry = Registry()
    registry.register(ScriptedFactory(results))
    return registry


def seed_pending_exchange(db_path: str, promoted: bool = True) -> int:
    conn = open_database(db_path)
    migrate(conn)
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (1, 1, 9, 1, 'alice', '2026-01-01T00:00:00+00:00', 'Octane2 jumper talk',"
        " '2026-01-01T00:00:00+00:00', '{}')"
    )
    row = ExchangeRow(
        id=None,
        channel_id=1,
        thread_id=None,
        first_message_id=1,
        last_message_id=1,
        started_at=NOW,
        ended_at=NOW,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash="hash-1",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    exchange_id = insert_exchange(conn, row, [1])
    if promoted:
        register_prompt_version(conn, PROMPT_VERSION, PROMPT_SHA256, NOW)
        promote_prompt_version(conn, PROMPT_VERSION, NOW)
    conn.close()
    return exchange_id


def run(argv: list[str], env: dict[str, str], registry: Registry) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err, registry=registry)
    return code, out.getvalue(), err.getvalue()


def test_extract_is_a_builtin_command() -> None:
    assert "extract" in [command.name for command in builtin_commands()]


def test_extract_live_mode_success_end_to_end(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"])
    registry = registry_with(success_results())

    code, out, _ = run(["extract"], env, registry)

    assert code == ExitCode.OK
    assert "succeeded=1" in out
    assert "claims_recorded=1" in out
    assert "run ids:" in out


def test_extract_live_mode_failure_exits_1(tmp_path: Path) -> None:
    env = environment(tmp_path)
    env["INFOVORE_MAX_RETRIES"] = "1"
    seed_pending_exchange(env["INFOVORE_DB_PATH"])
    registry = registry_with([HEALTH_OK, LLMResult.failed(ErrorKind.FATAL, "boom", None)])

    code, out, _ = run(["extract"], env, registry)

    assert code == ExitCode.FAILURE
    assert "failed=1" in out


def test_extract_live_mode_without_promotion_exits_config(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"], promoted=False)
    registry = registry_with(success_results())

    code, _, err = run(["extract"], env, registry)

    assert code == ExitCode.CONFIG
    assert "promote --prompt-version" in err
    assert PROMPT_VERSION in err


def test_extract_trial_mode_requires_sample_or_exchange_id(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"])
    registry = registry_with(success_results())

    code, _, err = run(["extract", "--mode", "trial"], env, registry)

    assert code == ExitCode.CONFIG
    assert "--sample" in err
    assert "--exchange-id" in err


def test_extract_trial_mode_with_exchange_id_never_mutates_status(tmp_path: Path) -> None:
    env = environment(tmp_path)
    exchange_id = seed_pending_exchange(env["INFOVORE_DB_PATH"], promoted=False)
    registry = registry_with(success_results())

    code, out, _ = run(
        ["extract", "--mode", "trial", "--exchange-id", str(exchange_id)], env, registry
    )

    assert code == ExitCode.OK
    assert "claims_recorded=1" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    exchange = get_exchange(conn, exchange_id)
    assert exchange is not None
    assert exchange.extraction_status is ExtractionStatus.PENDING
    conn.close()


def test_extract_trial_mode_with_sample(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"], promoted=False)
    registry = registry_with(success_results())

    code, out, _ = run(
        ["extract", "--mode", "trial", "--sample", "1", "--seed", "5"], env, registry
    )

    assert code == ExitCode.OK
    assert "processed=1" in out


def test_extract_health_check_failure_exits_backend(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"])
    registry = registry_with([LLMResult.failed(ErrorKind.FATAL, "not logged in", None)])

    code, _, err = run(["extract"], env, registry)

    assert code == ExitCode.BACKEND
    assert "not logged in" in err


class FlushCountingIO(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes_at: list[int] = []

    def flush(self) -> None:
        self.flushes_at.append(self.getvalue().count("\n"))
        super().flush()


def test_extract_live_mode_streams_flushed_progress_lines(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"])
    registry = registry_with(success_results())
    out = FlushCountingIO()
    err = io.StringIO()
    code = main(
        ["extract"], environ=env, dotenv_path=None, stdout=out, stderr=err, registry=registry
    )
    assert code == ExitCode.OK
    lines = out.getvalue().splitlines()
    assert lines[0] == "extract: live mode, draining queued exchanges"
    assert lines[1] == "exchange 1: 1 claims (1 done)"
    assert out.flushes_at[:2] == [1, 2]


def test_extract_trial_mode_streams_known_total_counter(tmp_path: Path) -> None:
    env = environment(tmp_path)
    exchange_id = seed_pending_exchange(env["INFOVORE_DB_PATH"], promoted=False)
    registry = registry_with(success_results())

    code, out, _ = run(
        ["extract", "--mode", "trial", "--exchange-id", str(exchange_id)], env, registry
    )

    assert code == ExitCode.OK
    lines = out.splitlines()
    assert lines[0] == "extract: trial mode, 1 exchanges queued"
    assert lines[1] == f"exchange {exchange_id}: 1 claims (1/1)"


def test_extract_start_line_is_written_before_backend_processes_any_exchange(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"])
    out = io.StringIO()
    seen_first_line_early: list[bool] = []

    def responder(request: LLMRequest) -> LLMResult:
        if request.system == "health check":
            return HEALTH_OK
        seen_first_line_early.append(
            "extract: live mode, draining queued exchanges" in out.getvalue()
        )
        return LLMResult.ok_structured(VALID_EXTRACTION_OUT, "scripted-model")

    class SpyFactory:
        name = "scripted"

        def validate(self, settings: StageSettings) -> list[str]:
            return []

        def build(self, settings: StageSettings) -> LLMBackend:
            return FakeBackend(responder)

    registry = Registry()
    registry.register(SpyFactory())

    code = main(
        ["extract"],
        environ=env,
        dotenv_path=None,
        stdout=out,
        stderr=io.StringIO(),
        registry=registry,
    )

    assert code == ExitCode.OK
    assert seen_first_line_early == [True]
