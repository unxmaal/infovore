import argparse
import hashlib
import json
import signal
import sqlite3
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from infovore.claims.check import DEFAULT_THRESHOLD, run_check, show_checks
from infovore.claims.export_cases import export_cases
from infovore.claims.extract import (
    MAX_TOKENS,
    RECIPE,
    TIMEOUT,
    WINDOW_CHARS,
    ClaimsReplyError,
    ExchangeResult,
    build_request,
    extract_concurrently,
    fetch_model_id,
    http_get,
    http_post,
    prompt_hash,
    render_window,
    windows,
)
from infovore.claims.httpd import DEFAULT_CLAIMS_PORT, listening_url, shutdown_all, start_all
from infovore.claims.redact import WIDTH, Redacted, redact_conversation, require_salt
from infovore.config import ConfigError, normalize_channel_names
from infovore.db.batch import exchange_inputs_for_ids
from infovore.db.channel_filter import known_channel_names
from infovore.db.claims_v2 import (
    ExchangeOutcome,
    RunReport,
    archived_exchange_ids,
    create_run,
    processed_ok,
    record_exchange,
    rejection_rows,
    report_rows,
    review_rows,
    run_ids,
)
from infovore.eval.slices import slice_ids, slice_names

if TYPE_CHECKING:
    from infovore.cli import AppContext


def _select(context: "AppContext", args: argparse.Namespace) -> tuple[list[int], str]:
    conn = context.conn
    if args.channels:
        wanted = normalize_channel_names(args.channels)
        known = known_channel_names(conn)
        if unknown := sorted(wanted - known):
            raise ConfigError(
                f"unknown channel(s) in --channels: {', '.join(unknown)};"
                f" known channels: {', '.join(sorted(known)) or 'none'}"
            )
        ids = archived_exchange_ids(conn, wanted, context.settings.exclude_channels)
        return ids, f"channels={args.channels}"
    if args.slices:
        names = args.slices.split(",")
        unknown_slices = [n for n in names if n not in slice_names(conn)]
        if unknown_slices:
            raise ConfigError(f"unknown slice(s): {', '.join(unknown_slices)}")
        ids = list(dict.fromkeys(i for n in names for i in slice_ids(conn, n)))
        return ids, f"slices={args.slices}"
    try:
        ids = list(dict.fromkeys(int(part) for part in args.ids.split(",")))
    except ValueError as error:
        raise ConfigError("--ids must be comma-separated exchange ids") from error
    marks = ",".join("?" for _ in ids)
    found = {
        r[0] for r in conn.execute(f"SELECT id FROM current_exchanges WHERE id IN ({marks})", ids)
    }
    if missing := [i for i in ids if i not in found]:
        raise ConfigError(f"unknown exchange id(s): {', '.join(map(str, missing))}")
    return ids, f"ids={args.ids}"


def _outcome(result: ExchangeResult) -> ExchangeOutcome:
    return ExchangeOutcome(
        "ok",
        None,
        result.windows,
        result.input_tokens,
        result.output_tokens,
        result.seconds,
    )


class StopFlag:
    """First SIGINT/SIGTERM asks the run to wind down; a second one aborts."""

    def __init__(self) -> None:
        self.requested = False

    def request(self) -> None:
        self.requested = True

    def handle(self, signum: int, frame: object) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.request()


@contextmanager
def _stop_on_signals(flag: StopFlag) -> Iterator[None]:
    signums = (signal.SIGINT, signal.SIGTERM)
    previous = [signal.signal(n, flag.handle) for n in signums]
    try:
        yield
    finally:
        for number, handler in zip(signums, previous, strict=True):
            signal.signal(number, handler)


MAX_CONCURRENCY = 4


def progress_line(done: int, total: int, claims: int, elapsed: float) -> str:
    rate = done / elapsed * 3600 if elapsed > 0 else 0.0
    if rate > 0:
        left = round((total - done) / rate * 3600)
        eta = f"{left // 3600}:{left % 3600 // 60:02d}:{left % 60:02d}"
    else:
        eta = "n/a"
    return (
        f"progress: {done}/{total} conversations, claims={claims}, {rate:.1f} conv/hour, eta {eta}"
    )


def _redacted_stream(
    conn: sqlite3.Connection, ids: Sequence[int], salt: str
) -> Iterator[tuple[int, Redacted]]:
    for eid in ids:
        messages = exchange_inputs_for_ids(conn, [eid])[eid].messages
        yield eid, redact_conversation(messages, salt)


def _check_args(args: argparse.Namespace, salt: str | None) -> str:
    if args.limit is None:
        raise ConfigError("claims extract needs --limit N (try 5)")
    if args.limit < 1:
        raise ConfigError("--limit must be positive")
    if args.concurrency < 1:
        raise ConfigError("--concurrency must be positive")
    if args.max_tokens < 1:
        raise ConfigError("--max-tokens must be positive")
    if args.progress_every < 1:
        raise ConfigError("--progress-every must be positive")
    salt = require_salt(salt)
    if not (args.slices or args.ids or args.channels):
        raise ConfigError("--slices, --ids or --channels is required")
    if args.resume and not args.write:
        raise ConfigError("--resume needs --write")
    return salt


def _dry_run(context: "AppContext", args: argparse.Namespace, ids: list[int], salt: str) -> int:
    from infovore.cli import ExitCode

    out, total = context.stdout, 0
    for eid, redacted in _redacted_stream(context.conn, ids, salt):
        for part in windows(redacted.lines, args.window_chars):
            total += 1
            request = build_request(args.model, render_window(part), args.max_tokens)
            out.write(f"exchange {eid}\t{json.dumps(request)}\n")
    out.write(f"dry-run: {total} requests for {len(ids)} conversations, none sent\n")
    return int(ExitCode.OK)


def _extract(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode

    conn, out = context.conn, context.stdout
    salt = _check_args(args, context.settings.pseudonym_salt)
    all_ids, selection = _select(context, args)
    if args.dry_run:
        return _dry_run(context, args, all_ids[: args.limit], salt)
    model_id, source = fetch_model_id(args.endpoint, args.model, http_get)
    note = "" if source == "model_info" else " (alias; /model/info unavailable)"
    out.write(f"model_id={model_id}{note}\n")
    if args.resume:
        skip = processed_ok(conn, model_id, prompt_hash())
        out.write(f"resume: skipping {len(skip.intersection(all_ids))} already done\n")
        all_ids = [i for i in all_ids if i not in skip]
    workers = min(args.concurrency, MAX_CONCURRENCY)
    if workers < args.concurrency:
        out.write(f"concurrency capped at {MAX_CONCURRENCY}\n")
    ids = all_ids[: args.limit]
    if not ids:
        out.write("nothing to do\n")
        return int(ExitCode.OK)
    run_id = None
    if args.write:
        recipe = {
            "recipe": RECIPE,
            "window_chars": args.window_chars,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "limit": args.limit,
            "pseudonym_width": WIDTH,
            "salt_fingerprint": hashlib.sha256(f"fp:{salt}".encode()).hexdigest()[:8],
        }
        run_id = create_run(
            conn,
            endpoint=args.endpoint,
            model_alias=args.model,
            model_id=model_id,
            model_id_source=source,
            prompt_hash=prompt_hash(),
            selection=selection,
            recipe=recipe,
            now=context.clock.now(),
        )
    send = http_post(args.endpoint, args.timeout)
    done = failed = claims = rejected = tokens_in = tokens_out = 0
    seconds = 0.0
    started = time.monotonic()
    stop = StopFlag()
    with _stop_on_signals(stop):
        results = extract_concurrently(
            _redacted_stream(conn, ids, salt),
            send,
            args.model,
            args.window_chars,
            workers,
            lambda: stop.requested,
            args.max_tokens,
        )
        for eid, result in results:
            if isinstance(result, ClaimsReplyError):
                failed += 1
                out.write(f"exchange {eid}\tERROR\t{result}\n")
                if run_id is not None:
                    failure = ExchangeOutcome("failed", str(result), 0, 0, 0, 0.0)
                    record_exchange(conn, run_id, eid, failure, [], [])
            else:
                done += 1
                claims += len(result.claims)
                rejected += len(result.rejected)
                tokens_in += result.input_tokens
                tokens_out += result.output_tokens
                seconds += result.seconds
                out.write(
                    f"exchange {eid}\tclaims={len(result.claims)}\trejected={len(result.rejected)}"
                    f"\ttokens={result.input_tokens}+{result.output_tokens}"
                    f"\tseconds={result.seconds:.2f}\n"
                )
                for claim in result.claims:
                    out.write(f"  {claim.statement}\n")
                if run_id is not None:
                    record_exchange(
                        conn, run_id, eid, _outcome(result), result.claims, result.rejected
                    )
            if (done + failed) % args.progress_every == 0:
                line = progress_line(done + failed, len(ids), claims, time.monotonic() - started)
                sys.stderr.write(f"{line}\n")
    out.write(
        f"conversations={done} failed={failed} claims={claims} rejected={rejected}"
        f" tokens={tokens_in}+{tokens_out} seconds={seconds:.2f}\n"
    )
    out.write(f"run {run_id} written\n" if run_id else "not written (use --write)\n")
    if stop.requested:
        out.write(f"interrupted after {done + failed} of {len(ids)}; rerun with --resume\n")
        return 130
    return int(ExitCode.BACKEND if failed and not done else ExitCode.OK)


def _require_run(conn: sqlite3.Connection, run_id: int) -> None:
    if run_id not in run_ids(conn):
        raise ConfigError(f"unknown run {run_id}")


def _serve(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode
    from infovore.sift.httpd import block_until_interrupted

    _require_run(context.conn, args.run)
    salt = require_salt(context.settings.pseudonym_salt)
    rows = review_rows(context.conn, args.run)
    reviewed = sum(1 for r in rows if r.verdict is not None)
    context.stdout.write(f"{len(rows)} claims, {reviewed} reviewed\n")
    servers = start_all(
        args.hosts or ["127.0.0.1"], args.port, context.conn, context.clock, args.run, salt
    )
    try:
        for server in servers:
            context.stdout.write(f"listening on {listening_url(server)}\n")
        context.stdout.flush()
        block_until_interrupted(threading.Event())
    finally:
        shutdown_all(servers)
    return int(ExitCode.OK)


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def _format(r: RunReport) -> str:
    reviewed = sum(r.verdicts.values())
    processed = r.conversations + r.failed
    per_conv = f"{r.claims / r.conversations:.2f}" if r.conversations else "n/a"
    verdicts = ", ".join(f"{k} {v}" for k, v in r.verdicts.items())
    by_interface = "".join(
        f"    {name}: {', '.join(f'{k} {v}' for k, v in counts.items())}\n"
        for name, counts in r.interfaces.items()
        if sum(counts.values())
    )
    zero = f"{r.zero_claim_conversations} ({_pct(r.zero_claim_conversations, r.conversations)})"
    made_up = (
        f"{_pct(r.verdicts['made_up'], reviewed)} ({r.verdicts['made_up']} of {reviewed} reviewed)"
        if reviewed
        else "n/a"
    )
    seconds = f"{r.seconds / processed:.2f}" if processed else "n/a"
    return (
        f"run {r.run_id}: {r.model_id} ({r.model_id_source}) prompt {r.prompt_hash}\n"
        f"  conversations: {r.conversations} (failed {r.failed})\n"
        f"  claims: {r.claims} ({per_conv} per conversation, rejected {r.rejected})\n"
        f"  zero-claim conversations: {zero}\n"
        f"  reviews: {verdicts}, unreviewed {r.unreviewed}\n"
        f"{by_interface}"
        f"  made-up rate: {made_up}\n"
        f"  tokens: {r.input_tokens} in, {r.output_tokens} out\n"
        f"  seconds per conversation: {seconds}\n"
    )


def _report(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode

    if args.run is not None:
        _require_run(context.conn, args.run)
    reports = report_rows(context.conn, args.run)
    context.stdout.write("".join(_format(r) for r in reports) or "no runs\n")
    return int(ExitCode.OK)


def _show(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode

    _require_run(context.conn, args.run)
    out = context.stdout
    if args.check:
        show_checks(context.conn, args.run, out.write)
    elif args.rejected:
        for rejection in rejection_rows(context.conn, args.run):
            out.write(f"{rejection.speaker}: {rejection.statement} [{rejection.reason}]\n")
    else:
        for row in review_rows(context.conn, args.run):
            out.write(f"{row.claim_id}\t{row.speaker}: {row.statement}\n")
    return int(ExitCode.OK)


class ClaimsCommand:
    name = "claims"
    help = "claim extraction trial: extract with a local model, review, report"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(
            dest="claims_action", required=True, parser_class=argparse.ArgumentParser
        )
        extract = sub.add_parser("extract", help="extract claims from conversations")
        extract.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
        extract.add_argument("--model", required=True, help="model alias on the server")
        pick = extract.add_mutually_exclusive_group()
        pick.add_argument("--slices", default=None, help="comma-separated slice names")
        pick.add_argument("--ids", default=None, help="comma-separated exchange ids")
        pick.add_argument("--channels", default=None, help="comma-separated channel names")
        extract.add_argument("--limit", type=int, default=None, metavar="N")
        extract.add_argument("--resume", action="store_true")
        extract.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY, metavar="N")
        extract.add_argument("--progress-every", type=int, default=10, dest="progress_every")
        extract.add_argument("--write", action="store_true")
        extract.add_argument("--dry-run", action="store_true", dest="dry_run")
        extract.add_argument("--window-chars", type=int, default=WINDOW_CHARS, dest="window_chars")
        extract.add_argument("--max-tokens", type=int, default=MAX_TOKENS, dest="max_tokens")
        extract.add_argument("--timeout", type=float, default=TIMEOUT)
        serve = sub.add_parser("serve", help="serve the claim review page until Ctrl-C")
        serve.add_argument("--run", type=int, required=True)
        serve.add_argument("--host", action="append", default=None, dest="hosts")
        serve.add_argument("--port", type=int, default=DEFAULT_CLAIMS_PORT)
        show = sub.add_parser("show", help="print a run's claims or rejections as text")
        show.add_argument("--run", type=int, required=True)
        show.add_argument("--rejected", action="store_true")
        show.add_argument("--check", action="store_true", help="show check verdicts")
        check = sub.add_parser("check", help="verify claim facts against cited messages")
        check.add_argument("--run", type=int, action="append", required=True)
        check.add_argument("--write", action="store_true")
        check.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
        export = sub.add_parser("export-cases", help="write reviewed claims as JSONL eval cases")
        which = export.add_mutually_exclusive_group(required=True)
        which.add_argument("--runs", help="comma-separated run ids")
        which.add_argument(
            "--negatives", action="store_true", help="human-irrelevant conversations, no claims"
        )
        export.add_argument("--out", required=True, help="output path, outside any git work tree")
        export.add_argument("--window-chars", type=int, default=WINDOW_CHARS, dest="window_chars")
        report = sub.add_parser("report", help="per-run trial numbers")
        report.add_argument("--run", type=int, default=None)

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        if args.claims_action == "extract":
            return _extract(context, args)
        if args.claims_action == "serve":
            return _serve(context, args)
        if args.claims_action == "check":
            return run_check(context, args)
        if args.claims_action == "export-cases":
            return export_cases(context, args)
        if args.claims_action == "show":
            return _show(context, args)
        return _report(context, args)
