import asyncio
import json
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from infovore.config import DEFAULT_SCRATCH_DIR, StageSettings
from infovore.llm.process import ProcessResult, ProcessRunner
from infovore.llm.protocol import Capabilities, ErrorKind, LLMBackend, LLMRequest, LLMResult, Usage

DEFAULT_BINARY = "claude"
DEFAULT_RETRY_AFTER = 300.0

_USAGE_LIMIT_RE = re.compile(r"usage limit|rate limit|limit reached", re.IGNORECASE)
_AUTH_RE = re.compile(r"not logged in|authentication", re.IGNORECASE)
_NETWORK_RE = re.compile(
    r"connection refused|connection reset|network is unreachable|"
    r"getaddrinfo|dns|econnrefused|socket",
    re.IGNORECASE,
)
_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?")
_EPOCH_RE = re.compile(r"\b(1\d{9})\b")


def _looks_like_network_error(text: str) -> bool:
    return _NETWORK_RE.search(text) is not None


def _parse_retry_after(text: str) -> float:
    iso_match = _ISO_RE.search(text)
    if iso_match is not None:
        raw = iso_match.group(0)
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            target = datetime.fromisoformat(normalized)
        except ValueError:
            return DEFAULT_RETRY_AFTER
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        delta = (target - datetime.now(UTC)).total_seconds()
        return max(0.0, float(round(delta)))
    epoch_match = _EPOCH_RE.search(text)
    if epoch_match is not None:
        target_epoch = float(epoch_match.group(1))
        delta = target_epoch - datetime.now(UTC).timestamp()
        return max(0.0, float(round(delta)))
    return DEFAULT_RETRY_AFTER


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _output_tokens(entry: object) -> int:
    if isinstance(entry, Mapping):
        value = entry.get("outputTokens")
        if isinstance(value, int):
            return value
    return 0


class ClaudeCliBackend:
    def __init__(
        self,
        runner: ProcessRunner,
        model: str,
        *,
        binary: str = DEFAULT_BINARY,
        timeout: float,
        scratch_dir: Path,
        concurrency: int,
    ) -> None:
        self._runner = runner
        self._model = model
        self._binary = binary
        self._timeout = timeout
        self._scratch_dir = scratch_dir
        self._concurrency = concurrency

    def capabilities(self) -> Capabilities:
        return Capabilities(native_json_schema=True, max_concurrency=self._concurrency)

    async def complete(self, request: LLMRequest) -> LLMResult:
        argv = self._build_argv(request)
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(self._scratch_dir)) as cwd:
            result = await self._runner.run(argv, request.prompt, cwd, self._timeout)
        return self._map_result(result, request)

    def _build_argv(self, request: LLMRequest) -> list[str]:
        argv = [
            self._binary,
            "-p",
            "--model",
            self._model,
            "--system-prompt",
            request.system,
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--no-session-persistence",
            "--output-format",
            "json",
        ]
        if request.json_schema is not None:
            argv.extend(["--json-schema", json.dumps(request.json_schema, sort_keys=True)])
        return argv

    def _map_result(self, result: ProcessResult, request: LLMRequest) -> LLMResult:
        if result.timed_out:
            return LLMResult.failed(ErrorKind.TRANSIENT, "claude cli timed out", None)
        payload = self._parse_json(result.stdout)
        if payload is None:
            if _looks_like_network_error(result.stderr):
                kind = ErrorKind.TRANSIENT
            else:
                kind = ErrorKind.FATAL
            message = result.stderr.strip() or f"claude cli exited with code {result.exit_code}"
            return LLMResult.failed(kind, message, None)
        return self._map_payload(payload, request, result.stderr)

    def _parse_json(self, stdout: str) -> Mapping[str, object] | None:
        stripped = stdout.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return None
        return parsed if isinstance(parsed, Mapping) else None

    def _map_payload(
        self, payload: Mapping[str, object], request: LLMRequest, stderr: str
    ) -> LLMResult:
        if payload.get("is_error"):
            return self._map_error(payload, stderr)
        model = self._select_model(payload)
        usage = self._extract_usage(payload)
        if request.json_schema is not None:
            structured = payload.get("structured_output")
            if not isinstance(structured, Mapping):
                return LLMResult.failed(
                    ErrorKind.FATAL, "structured_output missing from claude cli response", None
                )
            return LLMResult.ok_structured(structured, model, usage)
        text = payload.get("result")
        return LLMResult.ok_text(text if isinstance(text, str) else "", model, usage)

    def _select_model(self, payload: Mapping[str, object]) -> str:
        model_usage = payload.get("modelUsage")
        if isinstance(model_usage, Mapping) and model_usage:
            best = max(model_usage.items(), key=lambda pair: _output_tokens(pair[1]))
            return str(best[0])
        return self._model

    def _extract_usage(self, payload: Mapping[str, object]) -> Usage:
        usage_payload = payload.get("usage")
        input_tokens = None
        output_tokens = None
        if isinstance(usage_payload, Mapping):
            input_tokens = _as_int(usage_payload.get("input_tokens"))
            output_tokens = _as_int(usage_payload.get("output_tokens"))
        cost = payload.get("total_cost_usd")
        cost_usd = float(cost) if isinstance(cost, int | float) else None
        return Usage(input_tokens, output_tokens, cost_usd)

    def _map_error(self, payload: Mapping[str, object], stderr: str) -> LLMResult:
        status = payload.get("api_error_status")
        status_int = status if isinstance(status, int) else None
        result_text = payload.get("result")
        if isinstance(result_text, str) and result_text:
            message = result_text
        else:
            subtype = payload.get("subtype")
            if isinstance(subtype, str) and subtype:
                message = subtype
            else:
                message = "claude cli reported an error"
        haystack = f"{message}\n{stderr}"
        if status_int == 429 or _USAGE_LIMIT_RE.search(haystack) is not None:
            return LLMResult.failed(ErrorKind.USAGE_LIMIT, message, _parse_retry_after(haystack))
        if status_int == 529 or (status_int is not None and 500 <= status_int < 600):
            return LLMResult.failed(ErrorKind.TRANSIENT, message, None)
        if status_int in (401, 403) or _AUTH_RE.search(haystack) is not None:
            return LLMResult.failed(ErrorKind.FATAL, message, None)
        return LLMResult.failed(ErrorKind.FATAL, message, None)


class SubprocessRunner:
    async def run(
        self, argv: Sequence[str], stdin: str, cwd: str, timeout: float
    ) -> ProcessResult:  # pragma: no cover
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(stdin.encode()), timeout=timeout
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return ProcessResult(exit_code=None, stdout="", stderr="", timed_out=True)
        return ProcessResult(
            exit_code=process.returncode,
            stdout=stdout_bytes.decode(),
            stderr=stderr_bytes.decode(),
            timed_out=False,
        )


class ClaudeCliFactory:
    name = "claude_cli"

    def validate(self, settings: StageSettings) -> list[str]:
        return [] if settings.model else ["model is required"]

    def build(self, settings: StageSettings) -> LLMBackend:
        return ClaudeCliBackend(
            SubprocessRunner(),
            settings.model,
            binary=settings.options.get("binary", DEFAULT_BINARY),
            timeout=settings.timeout_seconds,
            scratch_dir=Path(settings.options.get("scratch_dir", DEFAULT_SCRATCH_DIR)),
            concurrency=settings.concurrency,
        )
