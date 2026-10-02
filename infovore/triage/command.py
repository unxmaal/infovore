import argparse
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.config import ConfigError
from infovore.triage.bayes import Metrics
from infovore.triage.gain_curve import compute_gain_curve, format_gain_report
from infovore.triage.human import (
    MIN_PER_CLASS,
    SCORER,
    InsufficientHumanLabelsError,
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
from infovore.triage.tuning import (
    MAX_CORPUS_DF,
    MIN_RECALL_FOR_REPORT_CARD,
    MIN_SIGNAL_SUPPORT,
    FitWeightsResult,
    NoLabelsError,
    ReportCard,
    ScoreCard,
    SignalStats,
    SuggestTermsResult,
    TermCandidate,
    compute_report_card,
    fit_weights,
    sampling_bias_warning,
    signal_report,
    suggest_terms,
)
from infovore.triage.yield_report import YieldBand, compute_yield_by_band

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


def _band_name(band: YieldBand) -> str:
    if band.lower is None:
        return "unscored"
    if band.upper is None:
        return f">={band.lower:g}"
    return f"{band.lower:g}-{band.upper:g}"


def _optional(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _render_yield(bands: list[YieldBand]) -> str:
    lines = [
        "",
        "realised yield by p_lore band (live runs only):",
        f"  {'band':<12} {'runs':>7} {'barren':>7} {'barren%':>8}"
        f" {'claims':>8} {'per run':>8} {'cost/claim':>11}",
    ]
    for band in bands:
        lines.append(
            f"  {_band_name(band):<12} {band.exchanges:>7} {band.barren:>7}"
            f" {_optional(band.barren_rate, '.1%'):>8}"
            f" {band.claims:>8} {_optional(band.claims_per_exchange, '.2f'):>8}"
            f" {_optional(band.cost_per_claim, '.4f'):>11}"
        )
    return "\n".join(lines) + "\n"


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


def _render_signal_report(stats: Sequence[SignalStats]) -> str:
    lines = [
        "signal                fires_lore  fires_noise  precision  lift    current  fitted   flag"
    ]
    for stat in stats:
        lines.append(
            f"  {stat.name:<20} {stat.fires_lore:>9}  {stat.fires_noise:>10}"
            f"  {stat.precision:>9.3f}  {stat.lift:>6.3f}  {stat.current_weight:>7.4f}"
            f"  {stat.fitted_weight:>7.4f}  {stat.flag}"
        )
    return "\n".join(lines) + "\n"


def _render_term_candidates(header: str, candidates: Sequence[TermCandidate]) -> list[str]:
    lines = [header]
    if not candidates:
        lines.append("  (none)")
        return lines
    for candidate in candidates:
        lines.append(
            f"  {candidate.token}: p={candidate.probability:.3f}"
            f" lore={candidate.lore_count} noise={candidate.noise_count}"
            f" corpus_df={candidate.corpus_df:.4f}"
        )
    return lines


def _render_suggest_terms(result: SuggestTermsResult) -> str:
    lines = _render_term_candidates(
        "candidate additions (strong lore evidence, rare corpus-wide, no current rule matches):",
        result.additions,
    )
    lines += _render_term_candidates(
        "drop candidates (current domain term, doesn't predict lore):", result.drops
    )
    lines.append(
        "insert these lines into the existing domain_terms = [...] list in rules.toml"
        " (never paste as a full replacement; review drop candidates above before"
        " removing them by hand):"
    )
    lines.append(result.snippet)
    return "\n".join(lines) + "\n"


def _render_sampling_bias_warning(
    class_imbalance_warning: bool,
    uncertain_sampling_share: float | None,
    uncertain_sampling_warning: bool,
) -> str | None:
    """One warning line (or `None`) for `--fit-weights`/`--signal-report`
    (issue #107): prefer the recorded-sampling-origin check when any labeled
    exchange's provenance is known, falling back to the class-imbalance
    check (issue #95) when it isn't — hand labels, or labels from before
    `extraction_runs.sampled_by` existed."""
    if uncertain_sampling_share is not None:
        if not uncertain_sampling_warning:
            return None
        return (
            f"warning: {uncertain_sampling_share:.3f} of labeled exchanges' sampling origin is"
            " `uncertain` (over 70%); a label set built mostly from `--strategy uncertain`"
            " rounds isn't representative. Mix in `--strategy random` rounds, or use"
            " `--strategy mixed`, so labels stay representative — see the README's Tuning loop"
            " section."
        )
    if class_imbalance_warning:
        return (
            "warning: over 70% of labels are one class; a rules edit fit from this data may not"
            " generalize. No sampling origin is recorded for these labels (hand labels, or labels"
            " from before `extraction_runs.sampled_by` existed), so this checks class imbalance"
            " instead — see the README's Tuning loop section."
        )
    return None


def _render_fit_weights(result: FitWeightsResult, out_path: Path) -> str:
    lines = [f"wrote fitted rules to {out_path}", "signal weights (before -> after):"]
    for change in result.changes:
        lines.append(f"  {change.name} ({change.field}): {change.before:.4f} -> {change.after:.4f}")
    lines.append(f"label class imbalance: {result.class_imbalance:.3f}")
    warning = _render_sampling_bias_warning(
        result.class_imbalance_warning,
        result.uncertain_sampling_share,
        result.uncertain_sampling_warning,
    )
    if warning is not None:
        lines.append(warning)
    return "\n".join(lines) + "\n"


def _shipped_rules_path() -> Path:
    return Path(str(resources.files("infovore.triage").joinpath("rules.toml"))).resolve()


def _render_score_card(card: ScoreCard) -> str:
    auc_text = f"{card.auc:.3f}" if card.auc is not None else "n/a (needs both labels)"
    if card.threshold is None:
        threshold_text = "unreachable at the target recall"
    else:
        assert card.recall_at_threshold is not None
        assert card.corpus_share is not None
        threshold_text = (
            f"threshold={card.threshold:.4f} recall={card.recall_at_threshold:.3f}"
            f" corpus_share={card.corpus_share:.3f}"
        )
    return f"  {card.label:<20} auc={auc_text}  {threshold_text}"


def _render_report_card(card: ReportCard, min_recall: float) -> str:
    lines = [f"report card (labels vs. score, corpus share at recall >= {min_recall:.2f}):"]
    lines.append(_render_score_card(card.rule_score))
    if card.p_lore is None:
        lines.append("  p_lore               no trained model yet, or it hasn't scored any label")
    else:
        lines.append(_render_score_card(card.p_lore))
    lines.append(_render_score_card(card.fitted_combination))
    return "\n".join(lines) + "\n"


class TriageCommand:
    name = "triage"
    help = "score exchanges with rule-based triage, train the classifier, and gate on it"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--report", action="store_true")
        parser.add_argument("--train", action="store_true")
        parser.add_argument("--human-report", action="store_true", dest="human_report")
        parser.add_argument("--train-human", action="store_true", dest="train_human")
        parser.add_argument(
            "--min-per-class", type=int, default=MIN_PER_CLASS, dest="min_per_class"
        )
        parser.add_argument("--human-limit", type=int, default=None, dest="human_limit")
        parser.add_argument("--all-exchanges", action="store_true", dest="all_exchanges")
        parser.add_argument(
            "--recommend-threshold", action="store_true", dest="recommend_threshold"
        )
        parser.add_argument("--min-recall", type=float, default=0.9, dest="min_recall")
        parser.add_argument("--signal-report", action="store_true", dest="signal_report")
        parser.add_argument(
            "--gain-curve",
            choices=["trial", "live"],
            default=None,
            dest="gain_curve",
        )
        parser.add_argument(
            "--compare-scorers",
            default=None,
            dest="compare_scorers",
            metavar="V1,V2",
            help="rank one population by several p_lore model versions",
        )
        parser.add_argument(
            "--compare-mode", choices=["trial", "live"], default="trial", dest="compare_mode"
        )
        parser.add_argument("--suggest-terms", action="store_true", dest="suggest_terms")
        parser.add_argument(
            "--min-support", type=int, default=MIN_SIGNAL_SUPPORT, dest="min_support"
        )
        parser.add_argument(
            "--max-corpus-df", type=float, default=MAX_CORPUS_DF, dest="max_corpus_df"
        )
        parser.add_argument("--fit-weights", action="store_true", dest="fit_weights")
        parser.add_argument("--out", type=str, default=None, dest="out")
        parser.add_argument("--force", action="store_true")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode, _say

        if args.recommend_threshold:
            return self._recommend(context, args)
        if args.train:
            return self._train(context)
        if args.human_report or args.train_human:
            return self._human(context, args)
        if args.signal_report:
            return self._signal_report(context)
        if args.gain_curve:
            return self._gain_curve(context, args.gain_curve)
        if args.compare_scorers:
            return self._compare_scorers(context, args.compare_scorers, args.compare_mode)
        if args.suggest_terms:
            return self._suggest_terms(context, args)
        if args.fit_weights:
            return self._fit_weights(context, args)

        report = triage_pending(
            context.conn,
            progress=lambda event: _say(context.stdout, _describe_triage_event(event)),
            rules=context.settings.triage_rules,
            workers=context.settings.workers,
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
            card = compute_report_card(context.conn, context.settings.triage_rules)
            if card is not None:
                context.stdout.write(_render_report_card(card, MIN_RECALL_FOR_REPORT_CARD))
            context.stdout.write(_render_yield(compute_yield_by_band(context.conn)))
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
        scored = score_all(
            context.conn,
            model,
            version,
            rules=context.settings.triage_rules,
            workers=context.settings.workers,
        )
        context.stdout.write(f"scored {scored} exchanges with p_lore\n")
        return ExitCode.OK

    def _human(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        if args.train_human and args.human_limit is None and not args.all_exchanges:
            raise ConfigError("--train-human needs --human-limit N or --all-exchanges")
        try:
            fit = fit_human(context.conn, context.settings.triage_rules, args.min_per_class)
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
                f"recommended threshold (recall={metric.recall:.3f}"
                f" precision={metric.precision:.3f}); add this line to your environment:\n"
                f"INFOVORE_TRIAGE_MIN_P_LORE={main_row.formatted_threshold}\n"
                f"expected share of exchanges sent to the LLM: {main_row.share:.3f}\n"
            )
        context.stdout.write(_render_recall_table(table_rows))
        return ExitCode.OK

    def _gain_curve(self, context: "AppContext", mode: str) -> int:
        from infovore.cli import ExitCode

        report = compute_gain_curve(context.conn, mode=mode)
        for line in format_gain_report(report):
            context.stdout.write(f"{line}\n")
        return int(ExitCode.OK)

    def _compare_scorers(self, context: "AppContext", versions: str, mode: str) -> int:
        """Rank one population by several scorer versions, reading each from
        `annotations`. Needs no labels and makes no LLM calls: the ground
        truth is the recorded extraction outcome."""
        from infovore.cli import ExitCode
        from infovore.triage.scorer_compare import (
            bootstrap_difference,
            candidates_for,
            compare_scorers,
            format_comparison,
            scores_from_annotations,
        )

        try:
            wanted = [int(part) for part in versions.split(",") if part.strip()]
        except ValueError:
            raise ConfigError(f"--compare-scorers wants versions, got {versions!r}") from None
        if not wanted:
            raise ConfigError("--compare-scorers needs at least one version")

        for line in format_comparison(compare_scorers(context.conn, mode, wanted)):
            context.stdout.write(f"{line}\n")

        # The curve alone cannot say whether a gap is real. A paired bootstrap
        # against the last version named gives each gap an interval.
        baseline = f"p_lore v{wanted[-1]}"
        arms = {f"p_lore v{v}": scores_from_annotations(context.conn, "p_lore", v) for v in wanted}
        candidates = candidates_for(context.conn, mode)
        context.stdout.write(f"\npaired against {baseline}, 95% CI over 2000 resamples\n")
        for fraction in (0.10, 0.25):
            context.stdout.write(f"  at {int(fraction * 100)}% of budget\n")
            for interval in bootstrap_difference(candidates, arms, baseline, fraction=fraction):
                verdict = "SEPARABLE" if interval.separable_from_zero else "not separable"
                context.stdout.write(
                    f"    {interval.label:<28} {interval.point:+6.2f} pts"
                    f"  [{interval.low:+6.2f}, {interval.high:+6.2f}]  {verdict}\n"
                )
        return int(ExitCode.OK)

    def _signal_report(self, context: "AppContext") -> int:
        from infovore.cli import ExitCode

        try:
            stats = signal_report(context.conn, context.settings.triage_rules)
            bias = sampling_bias_warning(context.conn, context.settings.triage_rules)
        except NoLabelsError as error:
            raise ConfigError(
                "no labeled exchanges; run `infovore label` first (see the README's Tuning loop)"
            ) from error
        context.stdout.write(_render_signal_report(stats))
        warning = _render_sampling_bias_warning(
            bias.class_imbalance_warning,
            bias.uncertain_sampling_share,
            bias.uncertain_sampling_warning,
        )
        if warning is not None:
            context.stdout.write(warning + "\n")
        return ExitCode.OK

    def _suggest_terms(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode, _say

        try:
            result = suggest_terms(
                context.conn,
                context.settings.triage_rules,
                min_support=args.min_support,
                max_corpus_df=args.max_corpus_df,
                progress=lambda scanned, total: _say(
                    context.stdout, f"corpus df: scanned {scanned}/{total} exchanges"
                ),
                workers=context.settings.workers,
            )
        except NoTrainedModelError as error:
            raise ConfigError("no trained model; run `infovore triage --train` first") from error
        context.stdout.write(_render_suggest_terms(result))
        return ExitCode.OK

    def _fit_weights(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        if not args.out:
            raise ConfigError("--fit-weights requires --out PATH")
        out_path = Path(args.out)
        if out_path.resolve() == _shipped_rules_path():
            raise ConfigError(
                f"refusing to overwrite the shipped rules.toml ({out_path}); pass a different --out"
            )
        if out_path.exists() and not args.force:
            raise ConfigError(f"refusing to overwrite existing file {out_path} (pass --force)")

        try:
            result = fit_weights(context.conn, context.settings.triage_rules)
        except NoLabelsError as error:
            raise ConfigError(
                "no labeled exchanges; run `infovore label` first (see the README's Tuning loop)"
            ) from error

        out_path.write_text(result.rules_toml, encoding="utf-8")
        context.stdout.write(_render_fit_weights(result, out_path))
        return ExitCode.OK
