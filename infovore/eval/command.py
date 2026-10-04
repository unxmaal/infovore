import argparse
import threading
from typing import TYPE_CHECKING

from infovore.config import ConfigError
from infovore.eval.channel_report import channel_report, format_channel_report
from infovore.eval.judge import (
    DEFAULT_UNCERTAINTY_SCORER,
    LABELS,
    LIKELY_IRRELEVANT,
    RELEVANCE_TARGET,
    UNCERTAIN,
    UNDECIDED,
    UNDECIDED_RELEVANT_TARGET,
    IRRELEVANT,
    RELEVANT,
    QueueBuilder,
    c1_queue,
    cached_queue,
    frozen_queue,
    label_counts,
    likely_irrelevant_queue,
    undecided_counts,
    undecided_queue,
    self_agreement,
    slice_progress,
    trainable_needed,
    uncertain_judged,
    uncertain_queue,
)
from infovore.eval.judge_httpd import DEFAULT_JUDGE_PORT, listening_url, shutdown_all, start_all
from infovore.eval.slices import (
    SliceExistsError,
    SliceTooSmallError,
    freeze_slices,
    slice_names,
    slice_summary,
)
from infovore.triage.human import trainable_counts

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
    help = "Eric's exchange-judging page and its report (#190)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(
            dest="judge_action", required=True, parser_class=argparse.ArgumentParser
        )
        serve = sub.add_parser("serve", help="serve the judging page until Ctrl-C")
        serve.add_argument("--host", action="append", default=None, dest="hosts")
        serve.add_argument("--port", type=int, default=DEFAULT_JUDGE_PORT)
        serve.add_argument(
            "--queue", choices=["frozen", UNCERTAIN, "c1", LIKELY_IRRELEVANT, UNDECIDED], default="frozen"
        )
        serve.add_argument(
            "--scorer",
            default=DEFAULT_UNCERTAINTY_SCORER,
            help="uncertainty source for --queue uncertain",
        )
        report = sub.add_parser(
            "report", help="labels, slice progress, self-agreement, labels still needed"
        )
        report.add_argument("--by-channel", action="store_true", help="per-channel label breakdown")
        report.add_argument("--min-labels", type=int, default=1)

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode
        from infovore.sift.httpd import block_until_interrupted

        if args.judge_action == "report" and args.by_channel:
            rows = channel_report(context.conn, context.settings.exclude_channels, args.min_labels)
            if rows:
                context.stdout.write(format_channel_report(rows))
            else:
                context.stdout.write(f"no channels with at least {args.min_labels} labels\n")
            return int(ExitCode.OK)
        if args.judge_action == "report":
            counts = label_counts(context.conn)
            for label in LABELS:
                context.stdout.write(f"{label}: {counts[label]}\n")
            for entry in slice_progress(context.conn):
                context.stdout.write(f"slice {entry.name}: {entry.done} of {entry.total} judged\n")
            context.stdout.write(f"uncertain: {uncertain_judged(context.conn)} judged\n")
            agreement = self_agreement(context.conn)
            rate = "n/a" if agreement.rate is None else f"{agreement.rate:.1%}"
            context.stdout.write(
                f"self-agreement: {rate} ({agreement.agreed} of {agreement.exchanges}"
                " repeated exchanges)\n"
            )
            excluded = context.settings.exclude_channels
            relevant, irrelevant = trainable_counts(context.conn, excluded)
            context.stdout.write(f"trainable relevant: {relevant}\n")
            context.stdout.write(f"trainable irrelevant: {irrelevant}\n")
            for label, need in trainable_needed(context.conn, excluded).items():
                context.stdout.write(f"needed: {need} more {label} to reach {RELEVANCE_TARGET}\n")
            pile = undecided_counts(context.conn)
            context.stdout.write(
                f"undecided pile labels: {pile[RELEVANT]} relevant, {pile[IRRELEVANT]} irrelevant\n"
            )
            more = max(0, UNDECIDED_RELEVANT_TARGET - pile[RELEVANT])
            context.stdout.write(
                f"undecided: {more} more relevant to reach {UNDECIDED_RELEVANT_TARGET}\n"
            )
            return int(ExitCode.OK)

        build_queue: QueueBuilder = frozen_queue
        if args.queue == UNCERTAIN:
            settings = context.settings
            build_queue = uncertain_queue(
                settings.max_retries,
                settings.exclude_channels,
                args.scorer,
            )
        elif args.queue == LIKELY_IRRELEVANT:
            build_queue = likely_irrelevant_queue(context.settings.exclude_channels)
        elif args.queue == UNDECIDED:
            build_queue = undecided_queue(context.settings.exclude_channels)
        elif args.queue == "c1":
            build_queue = c1_queue(context.settings.exclude_channels)
        elif not frozen_queue(context.conn):
            raise ConfigError("no gold set frozen yet; run `infovore slice freeze` first")
        if args.queue != "frozen":
            build_queue = cached_queue(build_queue)
        hosts = args.hosts or ["127.0.0.1"]
        servers = start_all(
            hosts,
            args.port,
            context.conn,
            context.clock,
            build_queue,
            None if args.queue == "frozen" else args.queue,
            context.settings.exclude_channels,
        )
        try:
            for server in servers:
                context.stdout.write(f"listening on {listening_url(server)}\n")
            context.stdout.flush()
            block_until_interrupted(threading.Event())
        finally:
            shutdown_all(servers)
        return int(ExitCode.OK)
