import io
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
from infovore.triage.score import TRIAGE_VERSION
from tests.cascade_marks import mark

NOW = datetime(2026, 1, 1, tzinfo=UTC)

VALID_EXTRACTION_OUT = {
    "claims": [
        {
            "statement": "needs a jumper on pin 3",
            "subject": "Octane2",
            "kind": "fact",
            "confidence": 0.9,
            "sources": ["m1"],
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


def mark_triaged(
    conn: sqlite3.Connection,
    exchange_id: int,
    triage_score: float | None = 1.0,
    cascade: str | None = "residue",
) -> None:
    if cascade is not None:
        mark(conn, exchange_id, cascade)
    conn.execute(
        "UPDATE exchanges SET triage_score = ?, triage_reasons = ?, triage_version = ?"
        " WHERE id = ?",
        (
            triage_score,
            "[]" if triage_score is not None else None,
            TRIAGE_VERSION if triage_score is not None else None,
            exchange_id,
        ),
    )


def seed_pending_exchange(
    db_path: str,
    promoted: bool = True,
    triage_score: float | None = 1.0,
    cascade: str | None = "residue",
) -> int:
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
    mark_triaged(conn, exchange_id, triage_score, cascade)
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
    assert lines[0] == "checking extract backend (scripted / sonnet)..."
    assert lines[1] == "extract: live mode, draining queued exchanges"
    assert lines[2] == "exchange 1: 1 claims (1 done)"
    assert out.flushes_at[:3] == [1, 2, 3]


def test_extract_trial_mode_streams_known_total_counter(tmp_path: Path) -> None:
    env = environment(tmp_path)
    exchange_id = seed_pending_exchange(env["INFOVORE_DB_PATH"], promoted=False)
    registry = registry_with(success_results())

    code, out, _ = run(
        ["extract", "--mode", "trial", "--exchange-id", str(exchange_id)], env, registry
    )

    assert code == ExitCode.OK
    lines = out.splitlines()
    assert lines[0] == "checking extract backend (scripted / sonnet)..."
    assert lines[1] == "extract: trial mode, 1 exchanges queued"
    assert lines[2] == f"exchange {exchange_id}: 1 claims (1/1)"


def test_extract_streams_skipped_progress_line_for_opted_out_exchange(tmp_path: Path) -> None:
    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (99, '2026-01-01T00:00:00+00:00')")
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (1, 1, 9, 99, 'bob', '2026-01-01T00:00:00+00:00', 'FACT: X :: y',"
        " '2026-01-01T00:00:00+00:00', '{}')"
    )
    exchange_id = insert_exchange(
        conn,
        ExchangeRow(
            id=None,
            channel_id=1,
            thread_id=None,
            first_message_id=1,
            last_message_id=1,
            started_at=NOW,
            ended_at=NOW,
            message_count=1,
            grouping_rule=GroupingRule.QUIET_GAP,
            content_hash="hash-skip",
            parent_exchange_id=None,
            extraction_status=ExtractionStatus.PENDING,
            retry_count=0,
            last_error=None,
        ),
        [1],
    )
    mark_triaged(conn, exchange_id)
    register_prompt_version(conn, PROMPT_VERSION, PROMPT_SHA256, NOW)
    promote_prompt_version(conn, PROMPT_VERSION, NOW)
    conn.close()
    registry = registry_with(success_results())

    code, out, _ = run(["extract"], env, registry)

    assert code == ExitCode.OK
    lines = out.splitlines()
    assert lines[0] == "checking extract backend (scripted / sonnet)..."
    assert lines[1] == "extract: live mode, draining queued exchanges"
    assert lines[2] == f"exchange {exchange_id}: skipped (1 done)"


def test_extract_streams_failed_progress_line(tmp_path: Path) -> None:
    env = environment(tmp_path)
    env["INFOVORE_MAX_RETRIES"] = "1"
    exchange_id = seed_pending_exchange(env["INFOVORE_DB_PATH"])
    registry = registry_with([HEALTH_OK, LLMResult.failed(ErrorKind.FATAL, "boom", None)])

    code, out, _ = run(["extract"], env, registry)

    assert code == ExitCode.FAILURE
    lines = out.splitlines()
    assert lines[0] == "checking extract backend (scripted / sonnet)..."
    assert lines[1] == "extract: live mode, draining queued exchanges"
    assert lines[2] == f"exchange {exchange_id}: failed: fatal (1 done)"


def test_extract_streams_paused_then_claimed_progress_lines(tmp_path: Path) -> None:
    env = environment(tmp_path)
    exchange_id = seed_pending_exchange(env["INFOVORE_DB_PATH"])
    calls = {"extract": 0}

    def responder(request: LLMRequest) -> LLMResult:
        if request.system == "health check":
            return HEALTH_OK
        calls["extract"] += 1
        if calls["extract"] == 1:
            return LLMResult.failed(ErrorKind.USAGE_LIMIT, "slow down", 0.01)
        return LLMResult.ok_structured(VALID_EXTRACTION_OUT, "scripted-model")

    class SpyFactory:
        name = "scripted"

        def validate(self, settings: StageSettings) -> list[str]:
            return []

        def build(self, settings: StageSettings) -> LLMBackend:
            return FakeBackend(responder)

    registry = Registry()
    registry.register(SpyFactory())

    code, out, _ = run(["extract"], env, registry)

    assert code == ExitCode.OK
    lines = out.splitlines()
    assert lines[0] == "checking extract backend (scripted / sonnet)..."
    assert lines[1] == "extract: live mode, draining queued exchanges"
    assert lines[2] == f"exchange {exchange_id}: paused 0.01s (usage limit) (0 done)"
    assert lines[3] == f"exchange {exchange_id}: 1 claims (1 done)"


def test_extract_prints_checking_backend_line_before_health_check(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"])
    out = io.StringIO()
    seen_before_health_check: list[bool] = []
    checked = {"done": False}

    def responder(request: LLMRequest) -> LLMResult:
        if not checked["done"]:
            seen_before_health_check.append(
                "checking extract backend (scripted / sonnet)..." in out.getvalue()
            )
            checked["done"] = True
        return HEALTH_OK

    class SpyFactory:
        name = "scripted"

        def validate(self, settings: StageSettings) -> list[str]:
            return []

        def build(self, settings: StageSettings) -> LLMBackend:
            return FakeBackend(responder)

    registry = Registry()
    registry.register(SpyFactory())

    main(
        ["extract"],
        environ=env,
        dotenv_path=None,
        stdout=out,
        stderr=io.StringIO(),
        registry=registry,
    )

    assert seen_before_health_check == [True]


def test_extract_live_mode_refuses_untriaged_exchange(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"], triage_score=None)
    registry = registry_with(success_results())

    code, _, err = run(["extract"], env, registry)

    assert code == ExitCode.CONFIG
    assert "infovore triage" in err


@pytest.mark.parametrize("cascade", [None, "embed_irrelevant", "denylist", "no_text"])
def test_extract_live_mode_only_claims_archived_exchanges(
    tmp_path: Path, cascade: str | None
) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"], cascade=cascade)
    registry = registry_with(success_results())

    code, out, _ = run(["extract"], env, registry)

    assert code == ExitCode.OK
    assert "processed=0" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    exchange = get_exchange(conn, 1)
    assert exchange is not None
    assert exchange.extraction_status is ExtractionStatus.PENDING
    conn.close()


def _seed_second_exchange(conn: sqlite3.Connection, triage_score: float) -> int:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (2, 1, 9, 1, 'alice', '2026-01-01T00:00:00+00:00', 'Octane2 jumper talk',"
        " '2026-01-01T00:00:00+00:00', '{}')"
    )
    row = ExchangeRow(
        id=None,
        channel_id=1,
        thread_id=None,
        first_message_id=2,
        last_message_id=2,
        started_at=NOW,
        ended_at=NOW,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash="hash-2",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    exchange_id = insert_exchange(conn, row, [2])
    mark_triaged(conn, exchange_id, triage_score)
    return exchange_id


def test_extract_trial_mode_min_score_filters_the_sample(tmp_path: Path) -> None:
    env = environment(tmp_path)
    qualifying_id = seed_pending_exchange(env["INFOVORE_DB_PATH"], promoted=False, triage_score=0.9)
    conn = open_database(env["INFOVORE_DB_PATH"])
    excluded_id = _seed_second_exchange(conn, triage_score=0.1)
    conn.close()
    assert qualifying_id != excluded_id
    registry = registry_with(success_results())

    code, out, _ = run(
        ["extract", "--mode", "trial", "--sample", "10", "--min-score", "0.5"], env, registry
    )

    assert code == ExitCode.OK
    assert "processed=1" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    claims = conn.execute("SELECT exchange_id FROM claims").fetchall()
    assert [row["exchange_id"] for row in claims] == [qualifying_id]
    conn.close()


def test_extract_trial_mode_max_score_filters_the_sample(tmp_path: Path) -> None:
    env = environment(tmp_path)
    qualifying_id = seed_pending_exchange(env["INFOVORE_DB_PATH"], promoted=False, triage_score=0.1)
    conn = open_database(env["INFOVORE_DB_PATH"])
    excluded_id = _seed_second_exchange(conn, triage_score=0.9)
    conn.close()
    assert qualifying_id != excluded_id
    registry = registry_with(success_results())

    code, out, _ = run(
        ["extract", "--mode", "trial", "--sample", "10", "--max-score", "0.5"], env, registry
    )

    assert code == ExitCode.OK
    assert "processed=1" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    claims = conn.execute("SELECT exchange_id FROM claims").fetchall()
    assert [row["exchange_id"] for row in claims] == [qualifying_id]
    conn.close()


def test_extract_records_an_extraction_batch_for_live_mode(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"])
    registry = registry_with(success_results())

    code, _, _ = run(["extract"], env, registry)

    assert code == ExitCode.OK
    conn = open_database(env["INFOVORE_DB_PATH"])
    rows = conn.execute("SELECT mode, strategy, seed, sample FROM extraction_batches").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["mode"] == "live"
    assert rows[0]["strategy"] is None
    assert rows[0]["sample"] is None


def test_extract_records_an_extraction_batch_for_trial_mode_with_sample(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"], promoted=False)
    registry = registry_with(success_results())

    code, _, _ = run(
        ["extract", "--mode", "trial", "--sample", "1", "--seed", "5", "--strategy", "random"],
        env,
        registry,
    )

    assert code == ExitCode.OK
    conn = open_database(env["INFOVORE_DB_PATH"])
    rows = conn.execute("SELECT mode, strategy, seed, sample FROM extraction_batches").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["mode"] == "trial"
    assert rows[0]["strategy"] == "random"
    assert rows[0]["seed"] == 5
    assert rows[0]["sample"] == 1


def test_extract_trial_mode_with_only_exchange_id_records_a_batch_with_no_strategy(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    exchange_id = seed_pending_exchange(env["INFOVORE_DB_PATH"], promoted=False)
    registry = registry_with(success_results())

    code, _, _ = run(
        ["extract", "--mode", "trial", "--exchange-id", str(exchange_id)], env, registry
    )

    assert code == ExitCode.OK
    conn = open_database(env["INFOVORE_DB_PATH"])
    rows = conn.execute("SELECT mode, strategy, sample FROM extraction_batches").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["mode"] == "trial"
    assert rows[0]["strategy"] is None
    assert rows[0]["sample"] is None


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


# --- INFOVORE_EXCLUDE_CHANNELS (issue #138) ----------------------------------


def seed_two_channel_exchanges(db_path: str) -> tuple[int, int]:
    """One exchange in `#general` (channel 1) and one in `#food` (channel 2),
    both pending and triaged, both promoted."""
    conn = open_database(db_path)
    migrate(conn)
    conn.executescript(
        """
        INSERT INTO channels (id, guild_id, name, kind) VALUES
          (1, 9, 'general', 'text'), (2, 9, 'food', 'text');
        INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,
          created_at, content, ingested_at, raw_json) VALUES
          (1, 1, 9, 1, 'alice', '2026-01-01T00:00:00+00:00', 'Octane2 jumper talk',
           '2026-01-01T00:00:00+00:00', '{}'),
          (2, 2, 9, 1, 'alice', '2026-01-01T00:00:00+00:00', 'Octane2 jumper talk',
           '2026-01-01T00:00:00+00:00', '{}');
        """
    )
    kept = ExchangeRow(
        id=None,
        channel_id=1,
        thread_id=None,
        first_message_id=1,
        last_message_id=1,
        started_at=NOW,
        ended_at=NOW,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash="kept",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    excluded = ExchangeRow(
        id=None,
        channel_id=2,
        thread_id=None,
        first_message_id=2,
        last_message_id=2,
        started_at=NOW,
        ended_at=NOW,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash="excluded",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    kept_id = insert_exchange(conn, kept, [1])
    excluded_id = insert_exchange(conn, excluded, [2])
    mark_triaged(conn, kept_id, 1.0)
    mark_triaged(conn, excluded_id, 1.0)
    register_prompt_version(conn, PROMPT_VERSION, PROMPT_SHA256, NOW)
    promote_prompt_version(conn, PROMPT_VERSION, NOW)
    conn.close()
    return kept_id, excluded_id


def test_extract_live_mode_never_claims_a_denylisted_channel(tmp_path: Path) -> None:
    env = environment(tmp_path)
    env["INFOVORE_EXCLUDE_CHANNELS"] = "food"
    kept_id, excluded_id = seed_two_channel_exchanges(env["INFOVORE_DB_PATH"])
    registry = registry_with(success_results())

    code, out, _ = run(["extract"], env, registry)

    assert code == ExitCode.OK
    assert "succeeded=1" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    kept = get_exchange(conn, kept_id)
    excluded = get_exchange(conn, excluded_id)
    assert kept is not None
    assert excluded is not None
    assert kept.extraction_status is ExtractionStatus.DONE
    assert excluded.extraction_status is ExtractionStatus.PENDING
    conn.close()


def test_extract_trial_mode_sample_excludes_denylisted_channel(tmp_path: Path) -> None:
    env = environment(tmp_path)
    env["INFOVORE_EXCLUDE_CHANNELS"] = "food"
    kept_id, _ = seed_two_channel_exchanges(env["INFOVORE_DB_PATH"])
    registry = registry_with(success_results())

    code, out, _ = run(
        ["extract", "--mode", "trial", "--sample", "10", "--seed", "0"], env, registry
    )

    assert code == ExitCode.OK
    assert "processed=1" in out
    assert f"exchange {kept_id}:" in out


def test_compare_prompt_rejects_an_unknown_version(tmp_path: Path) -> None:
    env = environment(tmp_path)
    registry = Registry()
    registry.register(ScriptedFactory(success_results()))

    code, _, err = run(["extract", "--compare-prompt", "v99"], env, registry)

    assert code == int(ExitCode.CONFIG)
    assert "unknown prompt version v99" in err


def test_compare_prompt_refuses_with_no_baseline_to_compare_against(tmp_path: Path) -> None:
    """The comparison re-extracts ALREADY-extracted exchanges, so with none
    present there is no baseline and it must say so rather than report zeroes."""
    env = environment(tmp_path)
    registry = Registry()
    registry.register(ScriptedFactory(success_results()))

    code, _, err = run(["extract", "--compare-prompt", "v6"], env, registry)

    assert code == int(ExitCode.CONFIG)
    assert "no already-extracted exchanges" in err


def test_compare_prompt_reports_both_arms(tmp_path: Path) -> None:
    """Both arms see the same exchanges, and nothing is written: the baseline
    a comparison measures against must survive being measured."""
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"])
    registry = Registry()
    registry.register(ScriptedFactory([HEALTH_OK] + [success_results()[1]] * 8))

    code, out, _ = run(["extract", "--mode", "live"], env, registry)
    assert code == int(ExitCode.OK)
    claims_before = _claim_count(env["INFOVORE_DB_PATH"])

    dump = tmp_path / "arms"
    code, out, _ = run(
        [
            "extract",
            "--compare-prompt",
            "v6",
            "--compare-limit",
            "1",
            "--compare-dump",
            str(dump),
        ],
        env,
        registry,
    )

    assert code == int(ExitCode.OK)
    # The baseline arm is whatever is live, not a fixed "v5".
    assert (dump / f"{PROMPT_VERSION}.jsonl").exists()
    assert (dump / "v6.jsonl").exists()
    assert "prompt comparison over 1 already-extracted exchanges" in out
    assert PROMPT_VERSION in out
    assert "v6" in out
    assert _claim_count(env["INFOVORE_DB_PATH"]) == claims_before


def _claim_count(db_path: str) -> int:
    conn = open_database(db_path)
    migrate(conn)
    return int(conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"])


def test_compare_prompt_runs_without_a_dump_directory(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"])
    registry = Registry()
    registry.register(ScriptedFactory([HEALTH_OK] + [success_results()[1]] * 8))

    assert run(["extract", "--mode", "live"], env, registry)[0] == int(ExitCode.OK)
    code, out, _ = run(["extract", "--compare-prompt", "v6", "--compare-limit", "1"], env, registry)

    assert code == int(ExitCode.OK)
    assert "prompt comparison" in out


@pytest.mark.parametrize(
    "flags",
    [
        ["--order", "best"],
        ["--mix", "0.5"],
        ["--mode", "trial", "--sample", "1", "--strategy", "uncertain"],
        ["--mode", "trial", "--sample", "1", "--strategy", "mixed"],
    ],
)
def test_extract_rejects_the_retired_p_lore_options(tmp_path: Path, flags: list[str]) -> None:
    env = environment(tmp_path)
    seed_pending_exchange(env["INFOVORE_DB_PATH"], promoted=False)

    code, _, err = run(["extract", *flags], env, registry_with(success_results()))

    assert code == ExitCode.CONFIG
    assert flags[-1] in err or flags[0] in err
