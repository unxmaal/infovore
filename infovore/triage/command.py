import argparse
from typing import TYPE_CHECKING

from infovore.triage.report import TriageStats, compute_triage_stats
from infovore.triage.runner import (
    TriageEvent,
    TriageExchangeScored,
    TriagePriorsApplied,
    TriageStarted,
    triage_pending,
)

if TYPE_CHECKING:
    from infovore.cli import AppContext


def _describe_triage_event(event: TriageEvent) -> str:
    match event:
        case TriageStarted(total=total):
            return f"triage: {total} exchanges to score"
        case TriageExchangeScored(exchange_id=exchange_id, score=score, index=index, total=total):
            return f"exchange {exchange_id}: score={score} ({index}/{total})"
        case _:
            assert isinstance(event, TriagePriorsApplied)
            return f"channel priors applied: {event.channels} channels"


def _render_report(stats: TriageStats) -> str:
    lines = ["score histogram:"]
    for index in range(10):
        bucket = f"{index / 10:.1f}-{(index + 1) / 10:.1f}"
        lines.append(f"  {bucket}: {stats.histogram.get(bucket, 0)}")
    lines.append("channels:")
    for channel_id in sorted(stats.channel_stats):
        mean, count = stats.channel_stats[channel_id]
        lines.append(f"  {channel_id}: mean={mean:.4f} count={count}")
    lines.append(f"above threshold: {stats.above_threshold}")
    lines.append(f"below threshold: {stats.below_threshold}")
    lines.append("top reasons:")
    for name, count in stats.top_reasons:
        lines.append(f"  {name}: {count}")
    return "\n".join(lines) + "\n"


class TriageCommand:
    name = "triage"
    help = "score exchanges with rule-based triage and channel priors"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--report", action="store_true")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode, _say

        report = triage_pending(
            context.conn,
            progress=lambda event: _say(context.stdout, _describe_triage_event(event)),
        )
        context.stdout.write(
            f"scored={report.scored} channels_adjusted={report.channels_adjusted}\n"
        )
        if args.report:
            stats = compute_triage_stats(context.conn, context.settings.triage_min_score)
            context.stdout.write(_render_report(stats))
        return ExitCode.OK
