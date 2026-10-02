import argparse
import threading
from typing import TYPE_CHECKING

from infovore.config import ConfigError
from infovore.eval.judge import fact_counts, progress, self_agreement
from infovore.eval.judge_httpd import DEFAULT_JUDGE_PORT, listening_url, shutdown_all, start_all
from infovore.eval.slices import (
    SliceExistsError,
    SliceTooSmallError,
    freeze_slices,
    slice_names,
    slice_summary,
)

if TYPE_CHECKING:
    from infovore.cli import AppContext


def _bucket_label(bucket: tuple[int, int]) -> str:
    low, high = bucket
    return f"{low}+" if high >= 10**9 else f"{low}-{high}"


class SliceCommand:
    name = "slice"
    help = "freeze and inspect the evaluation slices (#190)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(
            dest="slice_action", required=True, parser_class=argparse.ArgumentParser
        )
        sub.add_parser("freeze", help="freeze s1, s2, c1 and the gold set, once")
        show = sub.add_parser("show", help="exchanges and messages per size bucket")
        show.add_argument("name", nargs="?", default=None)

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        if args.slice_action == "freeze":
            settings = context.settings
            try:
                frozen = freeze_slices(
                    context.conn,
                    max_retries=settings.max_retries,
                    min_score=settings.triage_min_score,
                    min_p_lore=settings.triage_min_p_lore,
                    exclude_channels=settings.exclude_channels,
                    at=context.clock.now(),
                )
            except (SliceExistsError, SliceTooSmallError) as error:
                raise ConfigError(str(error)) from error
            for name, ids in frozen.items():
                context.stdout.write(f"froze {name}: {len(ids)} exchanges\n")
            return int(ExitCode.OK)

        names = [args.name] if args.name else slice_names(context.conn)
        if not names:
            context.stdout.write("no slices frozen yet; run `infovore slice freeze`\n")
            return int(ExitCode.OK)
        for name in names:
            rows = slice_summary(context.conn, name)
            exchanges = sum(row.exchanges for row in rows)
            messages = sum(row.messages for row in rows)
            context.stdout.write(f"{name}: {exchanges} exchanges, {messages} messages\n")
            for row in rows:
                context.stdout.write(
                    f"  {_bucket_label(row.bucket):<6} {row.exchanges:>4} exchanges"
                    f" {row.messages:>6} messages\n"
                )
        return int(ExitCode.OK)


class JudgeCommand:
    name = "judge"
    help = "Eric's fact-marking page over the gold set, and its report (#190)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(
            dest="judge_action", required=True, parser_class=argparse.ArgumentParser
        )
        serve = sub.add_parser("serve", help="serve the judging page until Ctrl-C")
        serve.add_argument("--host", action="append", default=None, dest="hosts")
        serve.add_argument("--port", type=int, default=DEFAULT_JUDGE_PORT)
        sub.add_parser("report", help="progress, facts marked, and self-agreement")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode
        from infovore.sift.httpd import block_until_interrupted

        if args.judge_action == "report":
            done, total = progress(context.conn)
            facts, no_facts = fact_counts(context.conn)
            agreement = self_agreement(context.conn)
            context.stdout.write(f"judged: {done} of {total} queue items\n")
            context.stdout.write(
                f"first pass: {facts} fact messages, {no_facts} no-fact messages\n"
            )
            rate = "n/a" if agreement.rate is None else f"{agreement.rate:.1%}"
            context.stdout.write(
                f"self-agreement: {rate} ({agreement.agreed} of {agreement.messages} messages"
                f" across {agreement.exchanges} repeated exchanges)\n"
            )
            return int(ExitCode.OK)

        if progress(context.conn)[1] == 0:
            raise ConfigError("no gold set frozen yet; run `infovore slice freeze` first")
        hosts = args.hosts or ["127.0.0.1"]
        servers = start_all(hosts, args.port, context.conn, context.clock)
        try:
            for server in servers:
                context.stdout.write(f"listening on {listening_url(server)}\n")
            context.stdout.flush()
            block_until_interrupted(threading.Event())
        finally:
            shutdown_all(servers)
        return int(ExitCode.OK)
