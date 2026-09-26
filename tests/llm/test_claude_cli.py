from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.config import StageSettings
from infovore.llm.claude_cli import ClaudeCliBackend, ClaudeCliFactory, SubprocessRunner
from infovore.llm.process import ProcessResult, ProcessRunner
from infovore.llm.protocol import Capabilities, ErrorKind, LLMRequest


@dataclass
class FakeProcessRunner:
    result: ProcessResult
    calls: list[tuple[list[str], str, str, float]] = field(default_factory=list)
    cwd_states: list[tuple[bool, list[Path]]] = field(default_factory=list)

    async def run(self, argv: Sequence[str], stdin: str, cwd: str, timeout: float) -> ProcessResult:
        self.calls.append((list(argv), stdin, cwd, timeout))
        path = Path(cwd)
        entries = list(path.iterdir()) if path.is_dir() else []
        self.cwd_states.append((path.is_dir(), entries))
        return self.result


@dataclass
class ExplodingRunner:
    calls: list[str] = field(default_factory=list)

    async def run(self, argv: Sequence[str], stdin: str, cwd: str, timeout: float) -> ProcessResult:
        self.calls.append(cwd)
        raise RuntimeError("runner exploded")


def make_backend(
    runner: ProcessRunner,
    scratch_dir: Path,
    model: str = "sonnet",
    binary: str = "claude",
    timeout: float = 30.0,
    concurrency: int = 3,
) -> ClaudeCliBackend:
    return ClaudeCliBackend(
        runner,
        model,
        binary=binary,
        timeout=timeout,
        scratch_dir=scratch_dir,
        concurrency=concurrency,
    )


def text_request(schema: Mapping[str, object] | None = None) -> LLMRequest:
    return LLMRequest(system="be terse", prompt="say hi", json_schema=schema, max_output_tokens=100)


def ok(**fields: object) -> ProcessResult:
    import json

    return ProcessResult(exit_code=0, stdout=json.dumps(fields), stderr="", timed_out=False)


def test_capabilities_reports_native_schema_and_stage_concurrency(tmp_path: Path) -> None:
    backend = make_backend(FakeProcessRunner(ok(result="x")), tmp_path, concurrency=7)
    assert backend.capabilities() == Capabilities(native_json_schema=True, max_concurrency=7)


async def test_argv_never_contains_bare_and_matches_exact_shape(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(result="hi", modelUsage={"claude-sonnet-5": {"outputTokens": 3}}))
    backend = make_backend(runner, tmp_path, model="sonnet", binary="claude")
    await backend.complete(text_request())
    argv, stdin, _cwd, timeout = runner.calls[0]
    assert argv == [
        "claude",
        "-p",
        "--model",
        "sonnet",
        "--system-prompt",
        "be terse",
        "--tools",
        "",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--no-session-persistence",
        "--output-format",
        "json",
    ]
    assert "--bare" not in argv
    assert stdin == "say hi"
    assert timeout == 30.0


async def test_argv_appends_json_schema_flag_when_schema_present(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    runner = FakeProcessRunner(ok(structured_output={"a": "b"}))
    backend = make_backend(runner, tmp_path)
    await backend.complete(text_request(schema=schema))
    argv = runner.calls[0][0]
    assert argv[-2] == "--json-schema"
    assert "--bare" not in argv
    import json as jsonlib

    assert jsonlib.loads(argv[-1]) == schema


async def test_cwd_is_a_fresh_empty_directory_under_scratch_dir(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    runner = FakeProcessRunner(ok(result="hi"))
    backend = make_backend(runner, scratch)
    await backend.complete(text_request())
    cwd = Path(runner.calls[0][2])
    assert cwd.is_relative_to(scratch)
    was_dir, entries = runner.cwd_states[0]
    assert was_dir
    assert entries == []


async def test_cwd_is_unique_per_call(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(result="hi"))
    backend = make_backend(runner, tmp_path)
    await backend.complete(text_request())
    await backend.complete(text_request())
    first_cwd, second_cwd = runner.calls[0][2], runner.calls[1][2]
    assert first_cwd != second_cwd


async def test_cwd_is_removed_after_a_successful_call(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    runner = FakeProcessRunner(ok(result="hi"))
    backend = make_backend(runner, scratch)
    await backend.complete(text_request())
    cwd = Path(runner.calls[0][2])
    assert not cwd.exists()
    assert list(scratch.iterdir()) == []


async def test_scratch_dir_has_no_leftover_entries_after_an_error_result(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    runner = FakeProcessRunner(ok(is_error=True, result="boom"))
    backend = make_backend(runner, scratch)
    result = await backend.complete(text_request())
    assert result.error is not None
    cwd = Path(runner.calls[0][2])
    assert not cwd.exists()
    assert list(scratch.iterdir()) == []


async def test_scratch_dir_has_no_leftover_entries_after_a_runner_exception(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    runner = ExplodingRunner()
    backend = make_backend(runner, scratch)
    with pytest.raises(RuntimeError, match="runner exploded"):
        await backend.complete(text_request())
    cwd = Path(runner.calls[0])
    assert not cwd.exists()
    assert list(scratch.iterdir()) == []


async def test_scratch_dir_is_created_when_missing(tmp_path: Path) -> None:
    scratch = tmp_path / "does" / "not" / "exist" / "yet"
    runner = FakeProcessRunner(ok(result="hi"))
    backend = make_backend(runner, scratch)
    assert not scratch.exists()
    await backend.complete(text_request())
    assert scratch.is_dir()
    assert list(scratch.iterdir()) == []


async def test_text_success_reads_result_field(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(result="plain text"))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is None
    assert result.text == "plain text"
    assert result.structured is None


async def test_text_success_defaults_to_empty_string_when_result_missing(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(modelUsage={"claude-sonnet-5": {"outputTokens": 1}}))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is None
    assert result.text == ""


async def test_structured_success_reads_structured_output(tmp_path: Path) -> None:
    schema = {"type": "object"}
    runner = FakeProcessRunner(ok(structured_output={"answer": "42"}))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request(schema=schema))
    assert result.error is None
    assert result.structured == {"answer": "42"}
    assert result.text is None


async def test_schema_requested_but_structured_output_missing_is_fatal(tmp_path: Path) -> None:
    schema = {"type": "object"}
    runner = FakeProcessRunner(ok(result="oops, no structured output"))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request(schema=schema))
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL
    assert "structured_output" in result.error.message


async def test_model_usage_picks_highest_output_token_entry(tmp_path: Path) -> None:
    runner = FakeProcessRunner(
        ok(
            result="hi",
            modelUsage={
                "claude-haiku-5": {"outputTokens": 5},
                "claude-sonnet-5": {"outputTokens": 50},
            },
        )
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.model == "claude-sonnet-5"


async def test_model_usage_falls_back_to_alias_when_absent(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(result="hi"))
    backend = make_backend(runner, tmp_path, model="sonnet")
    result = await backend.complete(text_request())
    assert result.model == "sonnet"


async def test_model_usage_treats_missing_output_tokens_as_zero(tmp_path: Path) -> None:
    runner = FakeProcessRunner(
        ok(
            result="hi",
            modelUsage={
                "claude-sonnet-5": {"outputTokens": 1},
                "claude-legacy": {},
            },
        )
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.model == "claude-sonnet-5"


async def test_model_usage_treats_non_mapping_entry_as_zero(tmp_path: Path) -> None:
    runner = FakeProcessRunner(
        ok(
            result="hi",
            modelUsage={
                "claude-sonnet-5": {"outputTokens": 1},
                "claude-weird": "not-a-mapping",
            },
        )
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.model == "claude-sonnet-5"


async def test_model_usage_falls_back_to_alias_when_empty(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(result="hi", modelUsage={}))
    backend = make_backend(runner, tmp_path, model="sonnet")
    result = await backend.complete(text_request())
    assert result.model == "sonnet"


async def test_usage_is_read_from_payload(tmp_path: Path) -> None:
    runner = FakeProcessRunner(
        ok(result="hi", usage={"input_tokens": 12, "output_tokens": 34}, total_cost_usd=0.05)
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 34
    assert result.usage.cost_usd == 0.05


async def test_usage_defaults_when_payload_has_none(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(result="hi"))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert result.usage.cost_usd is None


async def test_timeout_is_transient(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ProcessResult(exit_code=None, stdout="", stderr="", timed_out=True))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.TRANSIENT


async def test_nonzero_exit_with_empty_stdout_and_network_stderr_is_transient(
    tmp_path: Path,
) -> None:
    runner = FakeProcessRunner(
        ProcessResult(exit_code=1, stdout="", stderr="Connection refused", timed_out=False)
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.TRANSIENT


async def test_nonzero_exit_with_non_json_stdout_and_plain_stderr_is_fatal(
    tmp_path: Path,
) -> None:
    runner = FakeProcessRunner(
        ProcessResult(exit_code=1, stdout="not json", stderr="boom", timed_out=False)
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL


async def test_nonzero_exit_with_empty_everything_uses_exit_code_message(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ProcessResult(exit_code=7, stdout="", stderr="", timed_out=False))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL
    assert "7" in result.error.message


async def test_is_error_with_429_status_is_usage_limit(tmp_path: Path) -> None:
    runner = FakeProcessRunner(
        ok(is_error=True, api_error_status=429, subtype="rate_limit", result="too many requests")
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.USAGE_LIMIT


async def test_is_error_with_usage_limit_phrase_and_no_status_is_usage_limit(
    tmp_path: Path,
) -> None:
    runner = FakeProcessRunner(ok(is_error=True, result="you have hit your usage limit for today"))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.USAGE_LIMIT


async def test_usage_limit_defaults_retry_after_to_300_when_no_reset_time(
    tmp_path: Path,
) -> None:
    runner = FakeProcessRunner(ok(is_error=True, result="usage limit reached"))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.retry_after == 300.0


async def test_usage_limit_parses_retry_after_from_iso_timestamp(tmp_path: Path) -> None:
    target = datetime.now(UTC) + timedelta(seconds=120)
    runner = FakeProcessRunner(
        ok(is_error=True, result=f"usage limit reached, resets at {target.isoformat()}")
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.retry_after == 120.0


async def test_usage_limit_parses_retry_after_from_epoch_seconds(tmp_path: Path) -> None:
    target_epoch = round(datetime.now(UTC).timestamp()) + 120
    runner = FakeProcessRunner(
        ok(is_error=True, result=f"usage limit reached, resets at {target_epoch}")
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.retry_after is not None
    assert abs(result.error.retry_after - 120.0) <= 1.0


async def test_usage_limit_parses_retry_after_from_naive_iso_timestamp(tmp_path: Path) -> None:
    target = datetime.now(UTC).replace(tzinfo=None) + timedelta(seconds=120)
    runner = FakeProcessRunner(
        ok(is_error=True, result=f"usage limit reached, resets at {target.isoformat()}")
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.retry_after is not None
    assert abs(result.error.retry_after - 120.0) <= 1.0


async def test_usage_limit_falls_back_to_default_on_unparseable_iso_timestamp(
    tmp_path: Path,
) -> None:
    runner = FakeProcessRunner(
        ok(is_error=True, result="usage limit reached, resets at 2026-13-40T99:99:00Z")
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.retry_after == 300.0


async def test_is_error_529_is_transient(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(is_error=True, api_error_status=529, result="overloaded_error"))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.TRANSIENT


async def test_is_error_5xx_is_transient(tmp_path: Path) -> None:
    runner = FakeProcessRunner(
        ok(is_error=True, api_error_status=503, result="internal_server_error")
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.TRANSIENT


async def test_is_error_401_is_fatal(tmp_path: Path) -> None:
    runner = FakeProcessRunner(
        ok(is_error=True, api_error_status=401, result="authentication_error")
    )
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL


async def test_is_error_403_is_fatal(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(is_error=True, api_error_status=403, result="forbidden"))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL


async def test_is_error_not_logged_in_phrase_without_status_is_fatal(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(is_error=True, result="you are not logged in"))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL


async def test_is_error_unrecognized_defaults_to_fatal(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(is_error=True, result="something odd happened"))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.kind is ErrorKind.FATAL


async def test_is_error_with_missing_result_uses_subtype_message(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(is_error=True, subtype="error_max_turns"))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.message


async def test_is_error_with_no_result_or_subtype_uses_default_message(tmp_path: Path) -> None:
    runner = FakeProcessRunner(ok(is_error=True))
    backend = make_backend(runner, tmp_path)
    result = await backend.complete(text_request())
    assert result.error is not None
    assert result.error.message == "claude cli reported an error"


def test_factory_name_is_claude_cli() -> None:
    assert ClaudeCliFactory().name == "claude_cli"


def test_factory_validate_requires_model() -> None:
    settings = StageSettings(backend="claude_cli", model="", concurrency=1, timeout_seconds=1.0)
    assert ClaudeCliFactory().validate(settings) == ["model is required"]


def test_factory_validate_accepts_nonempty_model() -> None:
    settings = StageSettings(
        backend="claude_cli", model="sonnet", concurrency=1, timeout_seconds=1.0
    )
    assert ClaudeCliFactory().validate(settings) == []


def test_factory_build_uses_subprocess_runner_and_stage_settings() -> None:
    settings = StageSettings(
        backend="claude_cli", model="sonnet", concurrency=5, timeout_seconds=12.0
    )
    backend = ClaudeCliFactory().build(settings)
    assert isinstance(backend, ClaudeCliBackend)
    assert backend.capabilities() == Capabilities(native_json_schema=True, max_concurrency=5)


def test_factory_build_honors_binary_option() -> None:
    settings = StageSettings(
        backend="claude_cli",
        model="sonnet",
        concurrency=1,
        timeout_seconds=1.0,
        options={"binary": "/opt/claude/bin/claude"},
    )
    backend = ClaudeCliFactory().build(settings)
    assert isinstance(backend, ClaudeCliBackend)
    assert backend._binary == "/opt/claude/bin/claude"


def test_factory_build_default_binary_is_claude() -> None:
    settings = StageSettings(
        backend="claude_cli", model="sonnet", concurrency=1, timeout_seconds=1.0
    )
    backend = ClaudeCliFactory().build(settings)
    assert isinstance(backend, ClaudeCliBackend)
    assert backend._binary == "claude"


def test_factory_build_honors_scratch_dir_option() -> None:
    settings = StageSettings(
        backend="claude_cli",
        model="sonnet",
        concurrency=1,
        timeout_seconds=1.0,
        options={"scratch_dir": "/tmp/somewhere"},
    )
    backend = ClaudeCliFactory().build(settings)
    assert isinstance(backend, ClaudeCliBackend)
    assert backend._scratch_dir == Path("/tmp/somewhere")


def test_subprocess_runner_satisfies_process_runner_shape() -> None:
    runner = SubprocessRunner()
    assert hasattr(runner, "run")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("connection refused", True),
        ("connection reset by peer", True),
        ("network is unreachable", True),
        ("temporary failure in name resolution via getaddrinfo", True),
        ("permission denied", False),
    ],
)
def test_network_error_detection(text: str, expected: bool) -> None:
    from infovore.llm.claude_cli import _looks_like_network_error

    assert _looks_like_network_error(text) is expected
