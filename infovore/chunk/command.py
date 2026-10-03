import argparse
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from infovore.chunk.grouper import GroupingEvent, GroupingStarted, group_pending
from infovore.chunk.measure import MeasureRule, measure, render
from infovore.timing import FixedClock

if TYPE_CHECKING:
    from infovore.cli import AppContext

DEFAULT_MEASURE_GAPS = (30, 60, 120, 240)


def aware_datetime(text: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not an ISO-8601 time: {text!r}") from error
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError(f"time must include a timezone: {text!r}")
    return moment


def _describe_grouping_event(event: GroupingEvent) -> str:
    match event:
        case GroupingStarted(channels=channels):
            return f"grouping: {channels} channels with ungrouped messages"
        case _:
            return (
                f"channel {event.channel_id}: +{event.exchanges_created} exchanges,"
                f" {event.groups_deferred} deferred"
            )


def measure_rules(args: argparse.Namespace) -> list[MeasureRule]:
    fold = args.fold
    suffix = f" fold x{fold:g}/{args.fold_size}" if fold else ""
    rules = [
        MeasureRule(
            f"gap {minutes}{suffix}",
            gap=timedelta(minutes=minutes),
            fold_factor=fold,
            fold_size=args.fold_size,
        )
        for minutes in args.gap or DEFAULT_MEASURE_GAPS
    ]
    if args.adaptive:
        rules += [
            MeasureRule(
                f"adaptive p{percentile:g}{suffix}",
                percentile=percentile,
                floor=timedelta(minutes=args.min_gap),
                ceiling=timedelta(minutes=args.max_gap),
                fold_factor=fold,
                fold_size=args.fold_size,
            )
            for percentile in args.percentile or [90.0]
        ]
    return rules


class ChunkCommand:
    name = "chunk"
    help = "group ingested messages into closed exchanges"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--now", type=aware_datetime, default=None)
        parser.add_argument("--measure", action="store_true")
        parser.add_argument("--gap", type=int, action="append", metavar="MINUTES")
        parser.add_argument("--adaptive", action="store_true")
        parser.add_argument("--percentile", type=float, action="append")
        parser.add_argument("--min-gap", type=int, default=30, metavar="MINUTES")
        parser.add_argument("--max-gap", type=int, default=360, metavar="MINUTES")
        parser.add_argument("--fold", type=float, default=0, metavar="FACTOR")
        parser.add_argument("--fold-size", type=int, default=1, metavar="MESSAGES")
        parser.add_argument("--channels", type=int, default=10)

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import _say

        settings = context.settings
        if args.measure:
            rules = measure_rules(args)
            results = measure(
                context.conn, rules, settings.exchange_max_messages, settings.include_bot_messages
            )
            context.stdout.write(render(results, args.channels))
            return 0
        clock = FixedClock(args.now) if args.now is not None else context.clock
        report = group_pending(
            context.conn,
            clock,
            quiet_gap=timedelta(minutes=settings.quiet_gap_minutes),
            max_messages=settings.exchange_max_messages,
            include_bots=settings.include_bot_messages,
            progress=lambda event: _say(context.stdout, _describe_grouping_event(event)),
        )
        context.stdout.write(
            f"exchanges created: {report.exchanges_created}\n"
            f"messages grouped: {report.messages_grouped}\n"
            f"groups deferred: {report.groups_deferred}\n"
        )
        return 0
