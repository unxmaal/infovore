import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING

from infovore.config import ConfigError
from infovore.triage.bayes import Metrics
from infovore.triage.human import (
    MIN_PER_CLASS,
    SCORER,
    InsufficientHumanLabelsError,
    NoScorerAnnotationsError,
    evaluate_scorer,
    fit_human,
    score_human,
)
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


def _render_metrics_table(metrics: Sequence[Metrics]) -> str:
    lines = ["threshold  tp  fp  fn  tn  precision  recall  f1"]
    for metric in metrics:
        lines.append(
            f"{metric.threshold:.1f}  {metric.tp}  {metric.fp}  {metric.fn}  {metric.tn} "
            f" {metric.precision:.3f}  {metric.recall:.3f}  {metric.f1:.3f}"
        )
    return "\n".join(lines)


class TriageCommand:
    name = "triage"
    help = "score exchanges with rule-based triage and fit the human-label classifier"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--report", action="store_true")
        parser.add_argument("--human-report", action="store_true", dest="human_report")
        parser.add_argument("--train-human", action="store_true", dest="train_human")
        parser.add_argument(
            "--min-per-class", type=int, default=MIN_PER_CLASS, dest="min_per_class"
        )
        parser.add_argument("--scorer", default=None)
        parser.add_argument("--include-training", action="store_true", dest="include_training")
        parser.add_argument("--scorer-version", type=int, default=None, dest="scorer_version")
        parser.add_argument("--human-limit", type=int, default=None, dest="human_limit")
        parser.add_argument("--all-exchanges", action="store_true", dest="all_exchanges")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode, _say

        if args.human_report or args.train_human:
            return self._human(context, args)

        report = triage_pending(
            context.conn,
            progress=lambda event: _say(context.stdout, _describe_triage_event(event)),
            rules=context.settings.triage_rules,
            workers=context.settings.workers,
        )
        context.stdout.write(
            f"scored={report.scored} channels_adjusted={report.channels_adjusted}\n"
        )
        if args.report:
            stats = compute_triage_stats(
                context.conn, context.settings.triage_min_score, rules=context.settings.triage_rules
            )
            context.stdout.write(_render_report(stats))
        return ExitCode.OK

    def _human(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        if args.human_report and args.scorer is not None:
            return self._scorer_report(context, args)
        if args.train_human and args.human_limit is None and not args.all_exchanges:
            raise ConfigError("--train-human needs --human-limit N or --all-exchanges")
        try:
            fit = fit_human(
                context.conn,
                context.settings.triage_rules,
                args.min_per_class,
                context.settings.exclude_channels,
            )
        except InsufficientHumanLabelsError as error:
            if args.train_human:
                raise ConfigError(str(error)) from error
            context.stdout.write(f"human model: {error}\n")
            return ExitCode.OK
        report = fit.report
        auc_text = f"{report.auc:.3f}" if report.auc is not None else "n/a"
        context.stdout.write(
            f"human model: relevant={report.relevant} irrelevant={report.irrelevant}"
            f" holdout_size={report.holdout_size} auc={auc_text}\n"
        )
        context.stdout.write(_render_metrics_table(report.metrics) + "\n")
        if args.train_human:
            version, written = score_human(
                context.conn,
                fit,
                context.clock.now(),
                context.settings.triage_rules,
                args.human_limit,
            )
            context.stdout.write(
                f"wrote {written} derived annotations as {SCORER} v{version}"
                " (not activated; exchanges.p_lore unchanged)\n"
            )
        return ExitCode.OK

    def _scorer_report(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        try:
            result = evaluate_scorer(
                context.conn, args.scorer, args.scorer_version, args.include_training
            )
        except NoScorerAnnotationsError as error:
            raise ConfigError(str(error)) from error
        auc_text = f"{result.auc:.3f}" if result.auc is not None else "n/a"
        context.stdout.write(f"population: {result.population} n={result.evaluated}\n")
        context.stdout.write(
            f"scorer {result.scorer} v{result.version}: evaluated={result.evaluated}"
            f" relevant={result.relevant} irrelevant={result.irrelevant} auc={auc_text}\n"
        )
        context.stdout.write(_render_metrics_table(result.metrics) + "\n")
        return ExitCode.OK
