import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Any, Final

from infovore.claims.redact import Redacted, RenderedLine, leaks
from infovore.db.claims_v2 import ClaimIn, Rejection
from infovore.llm.gateway_key import headers, refusal

RECIPE: Final = "claims-v2"
WINDOW_CHARS: Final = 6000
TIMEOUT: Final = 300.0
INFO_TIMEOUT: Final = 10.0
CLAIM_MAX_CHARS: Final = 220
MAX_TOKENS: Final = 400
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "c": {
            "type": "array",
            "items": {
                "type": "array",
                "prefixItems": [
                    {"type": "string"},
                    {"type": "string", "maxLength": CLAIM_MAX_CHARS},
                    {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                ],
                "minItems": 3,
                "maxItems": 3,
            },
        }
    },
    "required": ["c"],
    "additionalProperties": False,
}
SYSTEM: Final = (
    "Lines are '[ref] user-id: text'. Extract standalone factual claims about technology,"
    " hardware, software, SGI and IRIX, retrocomputing, first-hand experience, or where to"
    " find things. Rewrite each in your own words as one short sentence, never copy a message."
    " It must make sense alone: resolve pronouns ('it' becomes the thing meant)."
    " No questions, opinions, jokes, banter or greetings."
    " First-hand experience is stated as the speaker's own ('they run ...')."
    ' Reply {"c":[[user-id, claim, [refs]], ...]}: the exact user id of the line the claim'
    " rests on, the claim without any user id in it, the ref numbers of its lines."
    " Say only what the lines say."
    ' With zero claims, reply {"c":[]}.'
    " Bad: 'It can even have an r12k. Mine is 400mhz' (copied, 'It' unresolved)."
    " Good: 'The O2 takes an R12000 CPU; theirs runs at 400 MHz.'"
    " Bad: 'isn't it just yoinking the chip and putting in a new one?' (a question)."
    " Good: nothing."
    " Bad: 'lol my O2 is the best machine ever' (joke, opinion)."
    " Good: nothing."
)

Transport = Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], float]]
Getter = Callable[[str], Any]


class ClaimsReplyError(Exception):
    pass


@dataclass(frozen=True)
class RawClaim:
    speaker: str
    statement: str
    refs: tuple[int, ...]


@dataclass(frozen=True)
class Reply:
    claims: list[RawClaim]
    input_tokens: int
    output_tokens: int
    seconds: float


@dataclass(frozen=True)
class ExchangeResult:
    claims: list[ClaimIn]
    rejected: list[Rejection]
    windows: int
    input_tokens: int
    output_tokens: int
    seconds: float


def prompt_hash() -> str:
    blob = json.dumps([SYSTEM, SCHEMA], sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def build_request(model: str, window: str, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Conversation:\n{window}"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "claims", "schema": SCHEMA, "strict": True},
        },
    }


def _render_line(line: RenderedLine) -> str:
    return f"[{line.ref}] {line.speaker}: {line.text}"


def render_window(lines: Sequence[RenderedLine]) -> str:
    return "\n".join(_render_line(line) for line in lines)


def windows(lines: Sequence[RenderedLine], max_chars: int) -> list[list[RenderedLine]]:
    out: list[list[RenderedLine]] = []
    size = 0
    for line in lines:
        overflow = len(_render_line(line)) - max_chars
        if overflow > 0:
            line = replace(line, text=line.text[: max(0, len(line.text) - overflow)])
        length = len(_render_line(line)) + 1
        if out and size + length <= max_chars:
            out[-1].append(line)
            size += length
        else:
            out.append([line])
            size = length
    return out


def parse_reply(reply: Mapping[str, Any], seconds: float) -> Reply:
    try:
        choice = reply["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ClaimsReplyError("truncated: reply hit max_tokens")
        body = json.loads(choice["message"]["content"])
        claims = [RawClaim(str(c[0]), str(c[1]), tuple(int(r) for r in c[2])) for c in body["c"]]
        usage = reply.get("usage") or {}
        return Reply(
            claims,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            seconds,
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ClaimsReplyError(f"unusable reply: {error!r}") from error


def validate(
    claims: Sequence[RawClaim], redacted: Redacted, window_refs: frozenset[int] | None = None
) -> tuple[list[ClaimIn], list[Rejection]]:
    by_ref = {line.ref: line.message_id for line in redacted.lines}
    accepted: list[ClaimIn] = []
    rejected: list[Rejection] = []
    for claim in claims:
        reason = _problem(claim, redacted, by_ref, window_refs)
        if reason is None:
            ids = tuple(dict.fromkeys(by_ref[r] for r in claim.refs))
            accepted.append(ClaimIn(claim.speaker, claim.statement.strip(), ids))
        else:
            cited = json.dumps(list(claim.refs))
            rejected.append(Rejection(claim.speaker, claim.statement, cited, reason))
    return accepted, rejected


def _problem(
    claim: RawClaim,
    redacted: Redacted,
    by_ref: Mapping[int, int],
    window_refs: frozenset[int] | None,
) -> str | None:
    if not claim.statement.strip():
        return "empty statement"
    if not claim.refs:
        return "no refs"
    for ref in claim.refs:
        if ref not in by_ref:
            return f"unknown ref {ref}"
        if window_refs is not None and ref not in window_refs:
            return f"ref {ref} not in window"
    if claim.speaker not in redacted.speakers:
        return "unknown speaker"
    if leaks(claim.statement, redacted.names):
        return "name leak"
    return None


def extract_exchange(
    post: Transport, model: str, redacted: Redacted, max_chars: int, max_tokens: int = MAX_TOKENS
) -> ExchangeResult:
    claims: list[ClaimIn] = []
    rejected: list[Rejection] = []
    tokens_in = tokens_out = 0
    seconds = 0.0
    parts = windows(redacted.lines, max_chars)
    for part in parts:
        body, elapsed = post(build_request(model, render_window(part), max_tokens))
        reply = parse_reply(body, elapsed)
        good, bad = validate(reply.claims, redacted, frozenset(line.ref for line in part))
        claims += good
        rejected += bad
        tokens_in += reply.input_tokens
        tokens_out += reply.output_tokens
        seconds += reply.seconds
    return ExchangeResult(claims, rejected, len(parts), tokens_in, tokens_out, seconds)


def extract_concurrently(
    items: Iterable[tuple[int, Redacted]],
    post: Transport,
    model: str,
    max_chars: int,
    concurrency: int,
    stop: Callable[[], bool],
    max_tokens: int = MAX_TOKENS,
) -> Iterator[tuple[int, ExchangeResult | ClaimsReplyError]]:
    """Yield (id, result or error) in completion order on the caller's thread; only
    the HTTP work runs in workers. Once stop() is true nothing new starts and
    in-flight work drains."""
    source = iter(items)
    pending: dict[Future[ExchangeResult], int] = {}
    with ThreadPoolExecutor(concurrency) as pool:
        exhausted = False
        while True:
            while not exhausted and not stop() and len(pending) < concurrency:
                item = next(source, None)
                if item is None:
                    exhausted = True
                    break
                future = pool.submit(extract_exchange, post, model, item[1], max_chars, max_tokens)
                pending[future] = item[0]
            if not pending:
                return
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=pending.__getitem__):
                eid = pending.pop(future)
                try:
                    yield eid, future.result()
                except ClaimsReplyError as error:
                    yield eid, error


def fetch_model_id(endpoint: str, alias: str, get: Getter) -> tuple[str, str]:
    root = endpoint.rstrip("/").removesuffix("/v1")
    try:
        for entry in get(f"{root}/model/info")["data"]:
            if entry.get("model_name") == alias:
                return str(entry["litellm_params"]["model"]), "model_info"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return alias, "alias"


def http_get(url: str) -> Any:
    request = urllib.request.Request(url, headers=headers())
    try:
        with urllib.request.urlopen(request, timeout=INFO_TIMEOUT) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        refused = refusal(error)
        if refused is None:
            raise
        raise refused from error


def http_post(endpoint: str, timeout: float = TIMEOUT) -> Transport:
    url = endpoint.rstrip("/") + "/chat/completions"
    sent = {"Content-Type": "application/json", **headers()}

    def send(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], float]:
        request = urllib.request.Request(url, json.dumps(payload).encode("utf-8"), sent)
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise refusal(error) or ClaimsReplyError(str(error)) from error
        return body, time.monotonic() - start

    return send
