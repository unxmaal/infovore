import argparse
import hashlib
import json
import sqlite3
import threading
from typing import TYPE_CHECKING

from infovore.claims.extract import (
    RECIPE,
    TIMEOUT,
    WINDOW_CHARS,
    ClaimsReplyError,
    ExchangeResult,
    build_request,
    extract_exchange,
    fetch_model_id,
    http_get,
    http_post,
    prompt_hash,
    render_window,
    windows,
)
from infovore.claims.httpd import DEFAULT_CLAIMS_PORT, listening_url, shutdown_all, start_all
from infovore.claims.redact import WIDTH, redact_conversation
from infovore.config import ConfigError
from infovore.db.batch import exchange_inputs_for_ids
from infovore.db.claims_v2 import (
    ExchangeOutcome,
    RunReport,
    create_run,
    record_exchange,
    rejection_rows,
    report_rows,
    review_rows,
    run_ids,
)
from infovore.eval.slices import slice_ids, slice_names

if TYPE_CHECKING:
    from infovore.cli import AppContext


def _select(conn: sqlite3.Connection, args: argparse.Namespace) -> tuple[list[int], str]:
    if args.slices:
        names = args.slices.split(",")
        unknown = [n for n in names if n not in slice_names(conn)]
        if unknown:
            raise ConfigError(f"unknown slice(s): {', '.join(unknown)}")
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


def _extract(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode

    conn, out, salt = context.conn, context.stdout, context.settings.pseudonym_salt
    if args.limit is None:
        raise ConfigError("claims extract needs --limit N (try 5)")
    if args.limit < 1:
        raise ConfigError("--limit must be positive")
    if not salt:
        raise ConfigError("INFOVORE_PSEUDONYM_SALT is required: names are redacted with it")
    if not (args.slices or args.ids):
        raise ConfigError("--slices or --ids is required")
    ids, selection = _select(conn, args)
    ids = ids[: args.limit]
    inputs = exchange_inputs_for_ids(conn, ids)
    redacted = {eid: redact_conversation(inputs[eid].messages, salt) for eid in ids}
    if args.dry_run:
        total = 0
        for eid in ids:
            for part in windows(redacted[eid].lines, args.window_chars):
                total += 1
                request = build_request(args.model, render_window(part))
                out.write(f"exchange {eid}\t{json.dumps(request)}\n")
        out.write(f"dry-run: {total} requests for {len(ids)} conversations, none sent\n")
        return int(ExitCode.OK)
    model_id, source = fetch_model_id(args.endpoint, args.model, http_get)
    note = "" if source == "model_info" else " (alias; /model/info unavailable)"
    out.write(f"model_id={model_id}{note}\n")
    run_id = None
    if args.write:
        recipe = {
            "recipe": RECIPE,
            "window_chars": args.window_chars,
            "temperature": 0,
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
    for eid in ids:
        try:
            result = extract_exchange(send, args.model, redacted[eid], args.window_chars)
        except ClaimsReplyError as error:
            failed += 1
            out.write(f"exchange {eid}\tERROR\t{error}\n")
            if run_id is not None:
                record_exchange(
                    conn, run_id, eid, ExchangeOutcome("failed", str(error), 0, 0, 0, 0.0), [], []
                )
            continue
        done += 1
        claims += len(result.claims)
        rejected += len(result.rejected)
        tokens_in += result.input_tokens
        tokens_out += result.output_tokens
        seconds += result.seconds
        out.write(
            f"exchange {eid}\tclaims={len(result.claims)}\trejected={len(result.rejected)}"
            f"\ttokens={result.input_tokens}+{result.output_tokens}\tseconds={result.seconds:.2f}\n"
        )
        for claim in result.claims:
            out.write(f"  {claim.statement}\n")
        if run_id is not None:
            record_exchange(conn, run_id, eid, _outcome(result), result.claims, result.rejected)
    out.write(
        f"conversations={done} failed={failed} claims={claims} rejected={rejected}"
        f" tokens={tokens_in}+{tokens_out} seconds={seconds:.2f}\n"
    )
    out.write(f"run {run_id} written\n" if run_id else "not written (use --write)\n")
    return int(ExitCode.BACKEND if failed and not done else ExitCode.OK)


def _require_run(conn: sqlite3.Connection, run_id: int) -> None:
    if run_id not in run_ids(conn):
        raise ConfigError(f"unknown run {run_id}")


def _serve(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode
    from infovore.sift.httpd import block_until_interrupted

    _require_run(context.conn, args.run)
    rows = review_rows(context.conn, args.run)
    reviewed = sum(1 for r in rows if r.verdict is not None)
    context.stdout.write(f"{len(rows)} claims, {reviewed} reviewed\n")
    servers = start_all(
        args.hosts or ["127.0.0.1"], args.port, context.conn, context.clock, args.run
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
    if args.rejected:
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
        extract.add_argument("--limit", type=int, default=None, metavar="N")
        extract.add_argument("--write", action="store_true")
        extract.add_argument("--dry-run", action="store_true", dest="dry_run")
        extract.add_argument("--window-chars", type=int, default=WINDOW_CHARS, dest="window_chars")
        extract.add_argument("--timeout", type=float, default=TIMEOUT)
        serve = sub.add_parser("serve", help="serve the claim review page until Ctrl-C")
        serve.add_argument("--run", type=int, required=True)
        serve.add_argument("--host", action="append", default=None, dest="hosts")
        serve.add_argument("--port", type=int, default=DEFAULT_CLAIMS_PORT)
        show = sub.add_parser("show", help="print a run's claims or rejections as text")
        show.add_argument("--run", type=int, required=True)
        show.add_argument("--rejected", action="store_true")
        report = sub.add_parser("report", help="per-run trial numbers")
        report.add_argument("--run", type=int, default=None)

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        if args.claims_action == "extract":
            return _extract(context, args)
        if args.claims_action == "serve":
            return _serve(context, args)
        if args.claims_action == "show":
            return _show(context, args)
        return _report(context, args)
