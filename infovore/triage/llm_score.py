import argparse
import hashlib
import json
import sqlite3
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from infovore.config import ConfigError
from infovore.db.annotations import Annotation, record_annotation
from infovore.db.batch import exchange_inputs_for_ids
from infovore.eval.slices import slice_ids, slice_names
from infovore.rows import Label, MessageRow
from infovore.triage.bayes import auc
from infovore.triage.cascade import (
    RESIDUE,
    current_exchange_ids,
    run_cascade,
    try_fit,
    tune_high,
    tuning_samples,
)
from infovore.triage.human import training_labels
from infovore.triage.lexicon import load_lexicon

if TYPE_CHECKING:
    from infovore.cli import AppContext

SCORER_PREFIX: Final = "p_relevant_llm_"
WINDOW_CHARS: Final = 6000
MAX_WINDOWS: Final = 8
TIMEOUT: Final = 120.0
RELEVANT: Final = "relevant"
IRRELEVANT: Final = "irrelevant"
SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": [RELEVANT, IRRELEVANT]},
        "probability": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["decision", "probability"],
    "additionalProperties": False,
}
SYSTEM: Final = (
    "You label chat exchanges from a community Discord. An exchange is relevant if it touches"
    " technology, computers, SGI, IRIX or retrocomputing at all, however briefly. It is"
    " irrelevant if it is off-topic chatter. Reply with JSON: decision (relevant or"
    " irrelevant) and probability, your confidence in that decision from 0 to 1."
)

Transport = Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], float]]


class LlmCallError(Exception):
    pass


@dataclass(frozen=True)
class Call:
    p_relevant: float
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    server_model: str
    fingerprint: str


@dataclass(frozen=True)
class Scored:
    exchange_id: int
    score: float
    windows: int
    prompt_tokens: int
    completion_tokens: int
    seconds: float


def prompt_hash() -> str:
    blob = json.dumps([SYSTEM, SCHEMA], sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def render(messages: Sequence[MessageRow]) -> list[str]:
    return [f"{m.author_name_at_time}: {m.content}".strip() for m in messages if m.content.strip()]


def windows(lines: Sequence[str], max_chars: int, max_windows: int) -> list[str]:
    pieces: list[str] = []
    for line in lines:
        pieces.extend(line[i : i + max_chars] for i in range(0, len(line), max_chars))
    out: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            out.append(current)
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        out.append(current)
    return out[:max_windows]


def build_request(model: str, window: str) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Exchange:\n{window}"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "decision", "schema": SCHEMA, "strict": True},
        },
    }


def parse_reply(reply: Mapping[str, Any], seconds: float) -> Call:
    try:
        decision = json.loads(reply["choices"][0]["message"]["content"])
        label, p = decision["decision"], float(decision["probability"])
        usage = reply.get("usage") or {}
        if label not in (RELEVANT, IRRELEVANT):
            raise ValueError(label)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise LlmCallError(f"unusable reply: {error!r}") from error
    p = min(1.0, max(0.0, p))
    return Call(
        p if label == RELEVANT else 1.0 - p,
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        seconds,
        str(reply.get("model") or ""),
        str(reply.get("system_fingerprint") or ""),
    )


def http_transport(endpoint: str, timeout: float = TIMEOUT) -> Transport:
    url = endpoint.rstrip("/") + "/chat/completions"

    def send(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], float]:
        request = urllib.request.Request(
            url,
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json", "Authorization": "Bearer not-needed"},
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise LlmCallError(str(error)) from error
        return body, time.monotonic() - start

    return send


def score_windows(transport: Transport, model: str, parts: Sequence[str]) -> list[Call]:
    calls = []
    for part in parts:
        reply, seconds = transport(build_request(model, part))
        calls.append(parse_reply(reply, seconds))
    return calls


def select_ids(
    conn: sqlite3.Connection,
    slices: str | None,
    residue: bool,
    labelled_only: bool,
    exclude: frozenset[str],
) -> list[int]:
    if slices:
        names = slices.split(",")
        unknown = [n for n in names if n not in slice_names(conn)]
        if unknown:
            raise ConfigError(f"unknown slice(s): {', '.join(unknown)}")
        ids = sorted({i for n in names for i in slice_ids(conn, n)})
    else:
        ids = current_exchange_ids(conn)
    if residue:
        lexicon = load_lexicon()
        t_high = tune_high(tuning_samples(conn, lexicon, exclude))
        fit, _ = try_fit(conn, exclude, None)
        outcomes = run_cascade(conn, ids, lexicon, t_high, fit, exclude)
        ids = [o.exchange_id for o in outcomes if o.stage == RESIDUE]
    if labelled_only:
        labels, _ = training_labels(conn)
        ids = [i for i in ids if i in labels]
    return ids


def configure(sub: Any) -> None:
    p = sub.add_parser("llm-score", help="score exchanges with a served typed-decision model")
    p.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. .../v1")
    p.add_argument("--model", required=True, help="model alias on the server")
    p.add_argument("--residue", action="store_true", help="only what the cascade abstains on")
    p.add_argument("--slices", default=None, help="comma-separated slice names")
    p.add_argument("--labelled-only", action="store_true", dest="labelled_only")
    p.add_argument("--limit", type=int, default=None, metavar="N")
    p.add_argument("--all-residue", action="store_true", dest="all_residue")
    p.add_argument("--write", action="store_true")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--window-chars", type=int, default=WINDOW_CHARS, dest="window_chars")
    p.add_argument("--max-windows", type=int, default=MAX_WINDOWS, dest="max_windows")
    p.add_argument("--timeout", type=float, default=TIMEOUT)


def _version(conn: sqlite3.Connection, scorer: str) -> int:
    row = conn.execute(
        "SELECT MAX(scorer_version) AS v FROM annotations WHERE scorer = ?", (scorer,)
    ).fetchone()
    return int(row["v"] or 0) + 1


def run_llm_score(
    context: "AppContext", args: argparse.Namespace, transport: Transport | None = None
) -> int:
    from infovore.cli import ExitCode

    if args.limit is None and not args.all_residue:
        raise ConfigError("llm-score needs --limit N (try 50) or --all-residue")
    if args.all_residue and not args.residue:
        raise ConfigError("--all-residue only makes sense with --residue")
    if not args.residue and not args.slices:
        raise ConfigError("--residue or --slices is required")
    conn, out = context.conn, context.stdout
    ids = select_ids(
        conn, args.slices, args.residue, args.labelled_only, context.settings.exclude_channels
    )
    if args.limit is not None:
        ids = ids[: args.limit]
    inputs = exchange_inputs_for_ids(conn, ids)
    plan = {
        eid: windows(render(inputs[eid].messages), args.window_chars, args.max_windows)
        for eid in ids
    }
    scorer = SCORER_PREFIX + args.model
    if args.dry_run:
        for eid in ids:
            for part in plan[eid]:
                out.write(f"exchange {eid}\t{json.dumps(build_request(args.model, part))}\n")
        total = sum(len(w) for w in plan.values())
        out.write(f"dry-run: {total} requests for {len(ids)} exchanges, none sent\n")
        return int(ExitCode.OK)
    send = transport or http_transport(args.endpoint, args.timeout)
    version = _version(conn, scorer)
    scored: list[Scored] = []
    server = ("", "")
    failed = 0
    for eid in ids:
        if not plan[eid]:
            continue
        try:
            calls = score_windows(send, args.model, plan[eid])
        except LlmCallError as error:
            failed += 1
            out.write(f"exchange {eid}\tERROR\t{error}\n")
            continue
        server = server if server[0] else (calls[0].server_model, calls[0].fingerprint)
        one = Scored(
            eid,
            max(c.p_relevant for c in calls),
            len(calls),
            sum(c.prompt_tokens for c in calls),
            sum(c.completion_tokens for c in calls),
            sum(c.seconds for c in calls),
        )
        scored.append(one)
        out.write(
            f"exchange {eid}\tscore={one.score:.3f}\twindows={one.windows}"
            f"\ttokens={one.prompt_tokens}+{one.completion_tokens}\tseconds={one.seconds:.2f}\n"
        )
        if args.write:
            recipe: dict[str, object] = {
                "endpoint": args.endpoint,
                "model": args.model,
                "server_model": server[0],
                "server_fingerprint": server[1],
                "prompt_sha": prompt_hash(),
                "window_chars": args.window_chars,
                "max_windows": args.max_windows,
                "windows": one.windows,
                "aggregate": "max",
                "prompt_tokens": one.prompt_tokens,
                "completion_tokens": one.completion_tokens,
                "seconds": round(one.seconds, 3),
            }
            note = Annotation(
                "exchange",
                eid,
                scorer,
                version,
                "derived",
                score=one.score,
                label=RELEVANT if one.score >= 0.5 else IRRELEVANT,
                recipe=recipe,
                source_ref="relevance-llm-score",
            )
            record_annotation(conn, note, context.clock.now())
    conn.commit()
    out.write(
        f"scored={len(scored)} failed={failed} tokens={sum(s.prompt_tokens for s in scored)}"
        f"+{sum(s.completion_tokens for s in scored)} seconds={sum(s.seconds for s in scored):.2f}"
        f" server_model={server[0] or '-'}\n"
    )
    if args.write and scored:
        out.write(f"wrote {len(scored)}: {scorer} v{version}\n")
    out.write(_evaluation(conn, scored))
    return int(ExitCode.BACKEND if failed and not scored else ExitCode.OK)


def _evaluation(conn: sqlite3.Connection, scored: Sequence[Scored]) -> str:
    labels, _ = training_labels(conn)
    pairs = [(s.score, labels[s.exchange_id]) for s in scored if s.exchange_id in labels]
    rel = sum(1 for _, label in pairs if label is Label.LORE)
    value = auc(pairs)
    text = "n/a" if value is None else f"{value:.3f}"
    return f"labelled: relevant={rel} irrelevant={len(pairs) - rel} auc={text} (no training)\n"
