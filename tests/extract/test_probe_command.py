import io
from datetime import UTC, datetime
from pathlib import Path

from infovore.cli import ExitCode, builtin_commands, main
from infovore.db.claims import NewClaim, get_claim, record_run, register_prompt_version
from infovore.db.connection import migrate, open_database
from infovore.extract.llm_extractor import JUDGE_SYSTEM_PROMPT, RECALL_SYSTEM_PROMPT
from infovore.llm.fake import FakeBackend
from infovore.llm.protocol import ErrorKind, LLMBackend, LLMRequest, LLMResult
from infovore.llm.registry import Registry
from infovore.rows import ClaimKind, ExtractionRunRow, Novelty, RunMode, RunOutcome

NOW = datetime(2026, 1, 1, tzinfo=UTC)
NOW_TEXT = NOW.isoformat()


def environment(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_PROBE_BACKEND": "scripted",
        "INFOVORE_JUDGE_BACKEND": "scripted",
    }
    env.update(overrides)
    return env


def seed(
    db_path: str, exchange_id: int = 1, message_id: int = 1, statement: str = "widget A"
) -> tuple[int, int]:
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
    register_prompt_version(conn, "v1", "sha", NOW)
    result = record_run(
        conn,
        ExtractionRunRow(
            id=None,
            exchange_id=exchange_id,
            model="m",
            prompt_version="v1",
            started_at=NOW,
            finished_at=NOW,
            input_tokens=1,
            output_tokens=1,
            mode=RunMode.LIVE,
            outcome=RunOutcome.OK,
            error=None,
        ),
        [
            NewClaim(
                exchange_id=exchange_id,
                statement=statement,
                subject="s",
                kind=ClaimKind.FACT,
                confidence=0.9,
                probe_question=f"what about {statement}?",
                permalink="https://discord.com/channels/1/1/1",
                supersedes_claim_id=None,
                source_message_ids=(message_id,),
            )
        ],
    )
    conn.close()
    return result.claim_ids[0], result.run_id


class ScriptedLLMFactory:
    name = "scripted"

    def __init__(self, verdict: str = "known", fail_recall: bool = False) -> None:
        self._verdict = verdict
        self._fail_recall = fail_recall

    def validate(self, settings: object) -> list[str]:
        return []

    def build(self, settings: object) -> LLMBackend:
        fail_recall = self._fail_recall
        verdict = self._verdict

        def responder(request: LLMRequest) -> LLMResult:
            if request.system == RECALL_SYSTEM_PROMPT:
                if fail_recall:
                    return LLMResult.failed(ErrorKind.TRANSIENT, "recall down", None)
                return LLMResult.ok_structured({"answer": "an answer"}, "claude-sonnet-5")
            if request.system == JUDGE_SYSTEM_PROMPT:
                return LLMResult.ok_structured(
                    {"verdict": verdict, "reason": "because"}, "claude-haiku-5"
                )
            return LLMResult.ok_text("pong", "claude-sonnet-5")

        return FakeBackend(responder)


def registry_with(factory: ScriptedLLMFactory) -> Registry:
    registry = Registry()
    registry.register(factory)
    return registry


def run(argv: list[str], env: dict[str, str], registry: Registry) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err, registry=registry)
    return code, out.getvalue(), err.getvalue()


def test_probe_is_a_builtin_command() -> None:
    assert "probe" in [command.name for command in builtin_commands()]


def test_probe_command_probes_unprobed_claims_and_prints_report(tmp_path: Path) -> None:
    env = environment(tmp_path)
    claim_id, _ = seed(env["INFOVORE_DB_PATH"])
    code, out, _ = run(["probe"], env, registry_with(ScriptedLLMFactory(verdict="known")))
    assert code == ExitCode.OK
    assert "probed: 1" in out
    assert "failed: 0" in out
    assert "pauses: 0" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    claim = get_claim(conn, claim_id)
    assert claim is not None
    assert claim.novelty is Novelty.KNOWN
    assert claim.probe_model == "claude-sonnet-5"


def test_probe_command_exits_failure_when_a_claim_fails(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    code, out, _ = run(["probe"], env, registry_with(ScriptedLLMFactory(fail_recall=True)))
    assert code == ExitCode.FAILURE
    assert "failed: 1" in out


def test_probe_command_limit_flag(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    code, out, _ = run(
        ["probe", "--limit", "1"], env, registry_with(ScriptedLLMFactory(verdict="partial"))
    )
    assert code == ExitCode.OK
    assert "probed: 1" in out


def test_probe_command_probe_model_flag_idempotent(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    registry = registry_with(ScriptedLLMFactory(verdict="known"))
    code, out, _ = run(["probe", "--probe-model", "claude-sonnet-5"], env, registry)
    assert code == ExitCode.OK
    assert "probed: 1" in out
    code, out, _ = run(["probe", "--probe-model", "claude-sonnet-5"], env, registry)
    assert code == ExitCode.OK
    assert "probed: 0" in out


def test_probe_command_run_id_flag_scopes_to_run(tmp_path: Path) -> None:
    env = environment(tmp_path)
    _, run_id_a = seed(env["INFOVORE_DB_PATH"], exchange_id=1, message_id=1, statement="widget A")
    claim_b, _ = seed(env["INFOVORE_DB_PATH"], exchange_id=2, message_id=2, statement="widget B")
    code, out, _ = run(
        ["probe", "--run-id", str(run_id_a)],
        env,
        registry_with(ScriptedLLMFactory(verdict="known")),
    )
    assert code == ExitCode.OK
    assert "probed: 1" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    claim = get_claim(conn, claim_b)
    assert claim is not None
    assert claim.novelty is Novelty.UNPROBED
