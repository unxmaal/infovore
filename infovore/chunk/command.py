import argparse
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from infovore.chunk.grouper import group_pending
from infovore.timing import FixedClock

if TYPE_CHECKING:
    from infovore.cli import AppContext


def aware_datetime(text: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not an ISO-8601 time: {text!r}") from error
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError(f"time must include a timezone: {text!r}")
    return moment


class ChunkCommand:
    name = "chunk"
    help = "group ingested messages into closed exchanges"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--now", type=aware_datetime, default=None)

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        settings = context.settings
        clock = FixedClock(args.now) if args.now is not None else context.clock
        report = group_pending(
            context.conn,
            clock,
            quiet_gap=timedelta(minutes=settings.quiet_gap_minutes),
            max_messages=settings.exchange_max_messages,
            include_bots=settings.include_bot_messages,
        )
        context.stdout.write(
            f"exchanges created: {report.exchanges_created}\n"
            f"messages grouped: {report.messages_grouped}\n"
            f"groups deferred: {report.groups_deferred}\n"
        )
        return 0
