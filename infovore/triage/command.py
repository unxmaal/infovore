import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING

from infovore.config import ConfigError
from infovore.triage.bayes import Metrics
from infovore.triage.report import TriageStats, compute_triage_stats
from infovore.triage.runner import (
    TriageEvent,
    TriageExchangeScored,
    TriagePriorsApplied,
    TriageStarted,
    triage_pending,
)
from infovore.triage.train import (
    RECALL_TARGETS,
    InsufficientLabelsError,
    NoTrainedModelError,
    RecommendationRow,
    TokenInfo,
    load_latest_model,
    model_rules_version,
    recommend_table,
    score_all,
    train_and_store,
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


def _render_confusion(metric: Metrics, threshold: float) -> str:
    return (
        f"confusion at p_lore>={threshold:.2f}: tp={metric.tp} fp={metric.fp} fn={metric.fn}"
        f" tn={metric.tn} precision={metric.precision:.3f} recall={metric.recall:.3f}"
        f" f1={metric.f1:.3f}"
    )


def _render_top_tokens(tokens: Sequence[TokenInfo]) -> str:
    lines = ["top tokens:"]
    for token in tokens:
        lines.append(
            f"  {token.token}: p={token.probability:.3f}"
            f" lore={token.lore_count} noise={token.noise_count}"
        )
    return "\n".join(lines)


def _rules_staleness_warning(context: "AppContext") -> str | None:
    """A one-line warning when the latest trained model's rules version differs
    from the currently active rules, or `None` when they match or there is no
    trained model yet. Rules edits change `SIG_<signal>` virtual tokens, so a
    model trained under different rules should be retrained, not silently
    reused (issue #95)."""
    trained_version = model_rules_version(context.conn)
    if trained_version is None:
        return None
    active_version = context.settings.triage_rules.version
    if trained_version == active_version:
        return None
    return (
        f"warning: the latest trained triage model was trained under rules {trained_version!r},"
        f" but the active rules are {active_version!r}; run `infovore triage --train` to retrain\n"
    )


def _ordered_unique(values: Sequence[float]) -> tuple[float, ...]:
    seen: list[float] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _render_recall_table(rows: Sequence[RecommendationRow]) -> str:
    lines = ["recall table (target recall -> threshold, corpus share):"]
    for row in rows:
        if row.metric is None:
            lines.append(f"  {row.min_recall:.2f}  unreachable")
        else:
            lines.append(
                f"  {row.min_recall:.2f}  threshold={row.formatted_threshold} share={row.share:.3f}"
            )
    return "\n".join(lines) + "\n"


class TriageCommand:
    name = "triage"
    help = "score exchanges with rule-based triage, train the classifier, and gate on it"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--report", action="store_true")
        parser.add_argument("--train", action="store_true")
        parser.add_argument(
            "--recommend-threshold", action="store_true", dest="recommend_threshold"
        )
        parser.add_argument("--min-recall", type=float, default=0.9, dest="min_recall")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode, _say

        if args.recommend_threshold:
            return self._recommend(context, args)
        if args.train:
            return self._train(context)

        report = triage_pending(
            context.conn,
            progress=lambda event: _say(context.stdout, _describe_triage_event(event)),
            rules=context.settings.triage_rules,
        )
        context.stdout.write(
            f"scored={report.scored} channels_adjusted={report.channels_adjusted}\n"
        )
        warning = _rules_staleness_warning(context)
        if warning is not None:
            context.stdout.write(warning)
        if args.report:
            stats = compute_triage_stats(
                context.conn, context.settings.triage_min_score, rules=context.settings.triage_rules
            )
            context.stdout.write(_render_report(stats))
        return ExitCode.OK

    def _train(self, context: "AppContext") -> int:
        from infovore.cli import ExitCode

        try:
            report = train_and_store(
                context.conn,
                context.clock,
                confusion_threshold=context.settings.triage_min_p_lore,
                rules=context.settings.triage_rules,
            )
        except InsufficientLabelsError as error:
            raise ConfigError(str(error)) from error

        context.stdout.write(
            f"trained model v{report.version}: labels_used={report.labels_used}"
            f" holdout_size={report.holdout_size}\n"
        )
        context.stdout.write(_render_metrics_table(report.metrics) + "\n")
        context.stdout.write(
            _render_confusion(report.confusion, context.settings.triage_min_p_lore) + "\n"
        )
        context.stdout.write(_render_top_tokens(report.top_tokens) + "\n")

        loaded = load_latest_model(context.conn)
        assert loaded is not None
        version, model = loaded
        scored = score_all(context.conn, model, version, rules=context.settings.triage_rules)
        context.stdout.write(f"scored {scored} exchanges with p_lore\n")
        return ExitCode.OK

    def _recommend(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        try:
            min_recalls = _ordered_unique((args.min_recall, *RECALL_TARGETS))
            main_row, *table_rows = recommend_table(
                context.conn, min_recalls, rules=context.settings.triage_rules
            )
        except NoTrainedModelError as error:
            raise ConfigError("no trained model; run `infovore triage --train` first") from error

        warning = _rules_staleness_warning(context)
        if warning is not None:
            context.stdout.write(warning)
        if main_row.metric is None:
            context.stdout.write(f"no threshold meets recall >= {args.min_recall}\n")
        else:
            assert main_row.share is not None
            assert main_row.formatted_threshold is not None
            metric = main_row.metric
            context.stdout.write(
                f"recommended INFOVORE_TRIAGE_MIN_P_LORE={main_row.formatted_threshold}"
                f" (recall={metric.recall:.3f} precision={metric.precision:.3f})\n"
                f"expected share of exchanges sent to the LLM: {main_row.share:.3f}\n"
            )
        context.stdout.write(_render_recall_table(table_rows))
        return ExitCode.OK
