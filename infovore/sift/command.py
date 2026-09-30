import argparse
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.config import ConfigError, normalize_channel_names
from infovore.db.channel_filter import known_channel_names
from infovore.sift.addresses import default_hosts
from infovore.sift.citations import derive_citation_labels
from infovore.sift.export import export_batch
from infovore.sift.features import DEFAULT_FEATURE_SET, FeatureSet
from infovore.sift.httpd import block_until_interrupted, listening_url, shutdown_all, start_all
from infovore.sift.importer import (
    ChannelCounts,
    MissingManifestError,
    NoSiftResultsFoundError,
    import_batch,
    save_trash_rules,
)
from infovore.sift.sampling import (
    DEFAULT_MIX_FRACTION_UNCERTAIN,
    NoScoredMessagesError,
    SiftStrategy,
)
from infovore.sift.serve import build_serve_app
from infovore.sift.train import (
    DEFAULT_CONFUSION_THRESHOLD,
    DEFAULT_MIN_HUMAN_LABELS_PER_CLASS,
    DiscardRow,
    FeatureSetAblation,
    InsufficientLabelsError,
    MessageTokenInfo,
    load_latest_message_model,
    score_all,
    train_and_store,
)
from infovore.triage.bayes import Metrics

if TYPE_CHECKING:
    from infovore.cli import AppContext

DEFAULT_SIFT_SIZE = 1000
DEFAULT_SIFT_SERVE_PORT = 8765

_NO_SIFT_RESULTS_MESSAGE = (
    "no kept.csv or trash-regexes.csv in DIR; in lnav, after filtering, run either"
    " `;SELECT msg FROM infovore_sift` then `:write-csv-to DIR/kept.csv`, or"
    " `;SELECT pattern FROM lnav_view_filters WHERE view_name='log' AND type='out'"
    " AND language='regex' AND enabled=1` then `:write-csv-to DIR/trash-regexes.csv`"
    " (see the README's 'Sifting' section)"
)


def _render_channel_counts(by_channel: dict[str, ChannelCounts]) -> str:
    return "".join(
        f"  #{channel}: keep={counts.keep} trash={counts.trash}\n"
        for channel, counts in sorted(by_channel.items())
    )


def _render_message_metrics_table(metrics: Sequence[Metrics]) -> str:
    lines = ["threshold  tp  fp  fn  tn  precision  recall  f1"]
    for metric in metrics:
        lines.append(
            f"{metric.threshold:.1f}  {metric.tp}  {metric.fp}  {metric.fn}  {metric.tn} "
            f" {metric.precision:.3f}  {metric.recall:.3f}  {metric.f1:.3f}"
        )
    return "\n".join(lines)


def _render_message_confusion(metric: Metrics, threshold: float) -> str:
    return (
        f"confusion at p_trash>={threshold:.2f}: tp={metric.tp} fp={metric.fp} fn={metric.fn}"
        f" tn={metric.tn} precision={metric.precision:.3f} recall={metric.recall:.3f}"
        f" f1={metric.f1:.3f}"
    )


def _render_message_auc(label: str, value: float | None) -> str:
    text = f"{value:.3f}" if value is not None else "n/a (needs both labels)"
    return f"  {label}: auc={text}"


def _render_channel_stats(channel_stats: dict[str, Metrics]) -> str:
    lines = ["citation-holdout per channel (by name, at the confusion threshold):"]
    for channel_name in sorted(channel_stats):
        metric = channel_stats[channel_name]
        lines.append(
            f"  #{channel_name}: precision={metric.precision:.3f} recall={metric.recall:.3f}"
            f" (tp={metric.tp} fp={metric.fp} fn={metric.fn} tn={metric.tn})"
        )
    return "\n".join(lines)


def _render_discard_pile(rows: Sequence[DiscardRow]) -> str:
    lines = ["discard pile (combined score, out-of-fold, against human labels only):"]
    for row in rows:
        keep_lost = f"{row.keep_lost:.3f}" if row.keep_lost is not None else "n/a"
        trash_caught = f"{row.trash_caught:.3f}" if row.trash_caught is not None else "n/a"
        lines.append(
            f"  p_trash>={row.threshold:.2f}: human_keep_lost={keep_lost}"
            f" human_trash_caught={trash_caught}"
        )
    return "\n".join(lines)


def _render_feature_set_ablation(
    ablations: Sequence[FeatureSetAblation], persisted: FeatureSet
) -> str:
    """The issue #144 three-way ablation: the combined model's out-of-fold
    AUC and discard pile for `plain`/`structural`/`context`, fit on the
    exact same folds -- so the gain or loss any one set brings over the
    others is measured, not assumed, on every run. The persisted/scored
    set (`--features`, default `plain`) is marked with `*`; its row's
    figures are exactly the ones already printed above this table."""
    lines = [
        "ablation (combined model, out-of-fold against human labels, same folds):",
        "  feature set  combined_auc  discard pile (p_trash>=thresh: keep_lost/trash_caught)",
    ]
    for row in ablations:
        marker = "*" if row.feature_set is persisted else " "
        auc_text = f"{row.combined_auc:.3f}" if row.combined_auc is not None else "n/a"
        cells = []
        for discard in row.discard_pile:
            keep_lost = f"{discard.keep_lost:.3f}" if discard.keep_lost is not None else "n/a"
            trash_caught = (
                f"{discard.trash_caught:.3f}" if discard.trash_caught is not None else "n/a"
            )
            cells.append(f"{discard.threshold:.1f}:{keep_lost}/{trash_caught}")
        lines.append(f"  {row.feature_set.value:<11}{marker} {auc_text:>10}    " + "  ".join(cells))
    lines.append("  (* = persisted/scored this run)")
    return "\n".join(lines)


def _render_message_top_tokens(label: str, tokens: Sequence[MessageTokenInfo]) -> str:
    lines = [f"top tokens ({label} model):"]
    for token in tokens:
        lines.append(
            f"  {token.token}: p_trash={token.probability:.3f}"
            f" trash={token.trash_count} keep={token.keep_count}"
        )
    return "\n".join(lines)


def _resolve_include_channels(context: "AppContext", raw: str | None) -> frozenset[str]:
    """`--channels` -> a validated, normalized include set (issue #138):
    empty/unset means no restriction. An unknown channel name exits `2`
    listing every known channel name, so a typo doesn't silently sample
    nothing."""
    include_channels = normalize_channel_names(raw) if raw else frozenset()
    if not include_channels:
        return include_channels
    known = known_channel_names(context.conn)
    unknown = sorted(include_channels - known)
    if unknown:
        raise ConfigError(
            f"unknown channel(s) in --channels: {', '.join(unknown)};"
            f" known channels: {', '.join(sorted(known)) or 'none'}"
        )
    return include_channels


class SiftCommand:
    name = "sift"
    help = "message-level trash sifting in lnav: export a batch, import human labels"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest="sift_command", required=True, parser_class=argparse.ArgumentParser
        )
        export_parser = subparsers.add_parser(
            "export", help="write an lnav-ready batch of messages to sift"
        )
        export_parser.add_argument("--size", type=int, default=DEFAULT_SIFT_SIZE)
        export_parser.add_argument(
            "--strategy",
            choices=[strategy.value for strategy in SiftStrategy],
            default=SiftStrategy.RANDOM.value,
        )
        export_parser.add_argument("--seed", type=int, default=0)
        export_parser.add_argument("--repeat", type=int, default=0)
        export_parser.add_argument(
            "--mix", type=float, default=DEFAULT_MIX_FRACTION_UNCERTAIN, dest="mix"
        )
        export_parser.add_argument("--out", type=str, default=None, dest="out")
        export_parser.add_argument(
            "--channels",
            type=str,
            default=None,
            dest="channels",
            help="comma-separated channel names to restrict sampling to (and their threads);"
            " the channel denylist (INFOVORE_EXCLUDE_CHANNELS) always wins",
        )

        import_parser = subparsers.add_parser(
            "import", help="record human labels from a sifted lnav batch"
        )
        import_parser.add_argument("dir", type=str)
        import_parser.add_argument("--save-rules", type=str, default=None, dest="save_rules")

        subparsers.add_parser(
            "citations",
            help="derive weak citation-based message labels from ok live extraction runs"
            " (issue #128 PR 3)",
        )

        train_parser = subparsers.add_parser(
            "train", help="train the message-level trash classifier and score p_trash"
        )
        train_parser.add_argument(
            "--human-weight",
            type=int,
            default=None,
            dest="human_weight",
            help="removed in #135 (message-level classes are now combined by a fitted"
            " logistic combiner, not by weighted duplication); passing this is an error",
        )
        train_parser.add_argument(
            "--min-human-labels",
            type=int,
            default=DEFAULT_MIN_HUMAN_LABELS_PER_CLASS,
            dest="min_human_labels",
            help="minimum human labels of each class needed to fit the combiner;"
            " below it, sift train falls back to the citation model alone",
        )
        train_parser.add_argument(
            "--features",
            choices=[fs.value for fs in FeatureSet],
            default=DEFAULT_FEATURE_SET.value,
            dest="features",
            help="which named feature set (issue #144) the persisted/scored model actually"
            f" uses (default: {DEFAULT_FEATURE_SET.value}); `structural` adds only the context"
            " signals (POS_/EXSIZE_/CTX_/*_FACT_*, never a neighbour's own word tokens),"
            " `context` adds everything including PREV_/NEXT_/REPLYTO_ word tokens; the"
            " report's ablation compares all three on the same folds regardless of this flag",
        )
        serve_parser = subparsers.add_parser(
            "serve", help="serve a keyboard-driven browser UI for sifting a batch"
        )
        serve_parser.add_argument("dir", type=str, nargs="?", default=None)
        serve_parser.add_argument("--new", action="store_true")
        serve_parser.add_argument("--size", type=int, default=DEFAULT_SIFT_SIZE)
        serve_parser.add_argument(
            "--strategy",
            choices=[strategy.value for strategy in SiftStrategy],
            default=SiftStrategy.RANDOM.value,
        )
        serve_parser.add_argument("--seed", type=int, default=0)
        serve_parser.add_argument("--repeat", type=int, default=0)
        serve_parser.add_argument(
            "--mix", type=float, default=DEFAULT_MIX_FRACTION_UNCERTAIN, dest="mix"
        )
        serve_parser.add_argument("--out", type=str, default=None, dest="out")
        serve_parser.add_argument("--host", action="append", default=None, dest="hosts")
        serve_parser.add_argument("--port", type=int, default=DEFAULT_SIFT_SERVE_PORT)
        serve_parser.add_argument(
            "--channels",
            type=str,
            default=None,
            dest="channels",
            help="comma-separated channel names to restrict sampling to (and their threads,"
            " --new only); the channel denylist (INFOVORE_EXCLUDE_CHANNELS) always wins",
        )

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        if args.sift_command == "import":
            return self._import(context, args)
        if args.sift_command == "citations":
            return self._citations(context, args)
        if args.sift_command == "train":
            return self._train(context, args)
        if args.sift_command == "serve":
            return self._serve(context, args)
        return self._export(context, args)

    def _citations(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        report = derive_citation_labels(context.conn, context.clock.now())
        context.stdout.write(
            f"considered {report.exchanges_considered} successfully extracted (live, ok)"
            f" exchanges: keep={report.keep} trash={report.trash}\n"
        )
        return ExitCode.OK

    def _train(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        if args.human_weight is not None:
            raise ConfigError(
                "--human-weight was removed in #135: the message classifier now trains a"
                " citation model and a human model separately and combines them with a"
                " fitted logistic regression, so there is no manual weight to set. Just"
                " run `infovore sift train` (optionally with --min-human-labels)."
            )

        try:
            report = train_and_store(
                context.conn,
                context.clock,
                min_human_labels_per_class=args.min_human_labels,
                feature_set=FeatureSet(args.features),
            )
        except InsufficientLabelsError as error:
            raise ConfigError(str(error)) from error

        fallback_note = (
            f" (fallback: citation-only, need >={report.min_human_labels_per_class} human"
            " labels per class to fit a combiner)"
            if report.fallback
            else ""
        )
        context.stdout.write(
            f"trained message model v{report.version} (features={report.feature_set.value}):"
            f" citation_labels={report.citation_labels_used}"
            f" human_labels={report.human_labels_used}{fallback_note}\n"
        )
        context.stdout.write("out-of-fold evaluation against human labels:\n")
        context.stdout.write(_render_message_auc("citation-only", report.citation_auc) + "\n")
        context.stdout.write(_render_message_auc("human-only", report.human_auc) + "\n")
        context.stdout.write(_render_message_auc("combined", report.combined_auc) + "\n")
        context.stdout.write(_render_discard_pile(report.discard_pile) + "\n")
        context.stdout.write("citation-holdout metrics (secondary):\n")
        context.stdout.write(_render_message_metrics_table(report.citation_holdout_metrics) + "\n")
        context.stdout.write(
            _render_message_confusion(
                report.citation_holdout_confusion, DEFAULT_CONFUSION_THRESHOLD
            )
            + "\n"
        )
        context.stdout.write(
            _render_message_auc("citation-holdout", report.citation_holdout_auc) + "\n"
        )
        context.stdout.write(_render_channel_stats(report.channel_stats) + "\n")
        context.stdout.write(
            _render_message_top_tokens("citation", report.citation_top_tokens) + "\n"
        )
        context.stdout.write(_render_message_top_tokens("human", report.human_top_tokens) + "\n")
        context.stdout.write(
            _render_feature_set_ablation(report.ablations, report.feature_set) + "\n"
        )

        loaded = load_latest_message_model(context.conn)
        assert loaded is not None
        # `train_and_store` above always stores the current feature set
        # version and `loaded` is that same freshly trained ensemble, so
        # `FeatureSetMismatchError` (infovore.sift.train) can't fire here --
        # it guards a caller that scores a *stale* ensemble without
        # retraining first, which this command never does.
        scored = score_all(context.conn, loaded, workers=context.settings.workers)
        context.stdout.write(f"scored {scored} messages with p_trash\n")
        return ExitCode.OK

    def _export(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        if not args.out:
            raise ConfigError("`sift export` requires --out DIR")
        if not 0.0 <= args.mix <= 1.0:
            raise ConfigError("--mix must be between 0 and 1")

        out_dir = Path(args.out)
        strategy = SiftStrategy(args.strategy)
        include_channels = _resolve_include_channels(context, args.channels)
        try:
            report = export_batch(
                context.conn,
                size=args.size,
                strategy=strategy,
                seed=args.seed,
                mix=args.mix,
                out_dir=out_dir,
                now=context.clock.now(),
                exclude_channels=context.settings.exclude_channels,
                include_channels=include_channels,
                repeat=args.repeat,
            )
        except NoScoredMessagesError as error:
            raise ConfigError(
                "--strategy uncertain requires messages already scored with p_trash;"
                " run `infovore sift train` first once it exists (issue #128 PR 3)"
            ) from error

        context.stdout.write(
            f"wrote {report.count} messages to {report.out_dir}\n"
            f"batch log: {report.batch_log_path}\n"
            f"lnav format: {report.format_path}\n"
            f"manifest: {report.manifest_path}\n"
            f"install the format once: lnav -i {report.format_path}\n"
            f"then sift: lnav {report.batch_log_path}\n"
        )
        return ExitCode.OK

    def _import(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        dir_ = Path(args.dir)
        try:
            report = import_batch(context.conn, dir_, context.clock.now())
        except MissingManifestError as error:
            raise ConfigError(
                f"no manifest.json in {dir_}; run `infovore sift export` first"
            ) from error
        except NoSiftResultsFoundError as error:
            raise ConfigError(_NO_SIFT_RESULTS_MESSAGE) from error

        if args.save_rules is not None:
            if report.path != "b":
                raise ConfigError(
                    "--save-rules requires trash-regexes.csv (path b); this batch was"
                    " imported from kept.csv (path a), which has no regexes to save"
                )
            saved_path = save_trash_rules(
                context.settings.scratch_dir, args.save_rules, report.patterns, context.clock.now()
            )
            context.stdout.write(f"saved trash rules {args.save_rules!r} to {saved_path}\n")

        context.stdout.write(
            f"imported {report.keep + report.trash} (path {report.path}): keep={report.keep}"
            f" trash={report.trash} (source_ref={report.source_ref})\n"
            f"{_render_channel_counts(report.by_channel)}"
        )
        return ExitCode.OK

    def _serve(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        if bool(args.dir) == bool(args.new):
            raise ConfigError("`sift serve` requires exactly one of DIR or --new")
        if args.new and not args.out:
            raise ConfigError("`sift serve --new` requires --out DIR")
        if not 0.0 <= args.mix <= 1.0:
            raise ConfigError("--mix must be between 0 and 1")

        strategy = SiftStrategy(args.strategy)
        include_channels = _resolve_include_channels(context, args.channels)
        try:
            app = build_serve_app(
                context.conn,
                dir_=Path(args.dir) if args.dir else None,
                new=args.new,
                size=args.size,
                strategy=strategy,
                seed=args.seed,
                mix=args.mix,
                out_dir=Path(args.out) if args.out else None,
                scratch_dir=context.settings.scratch_dir,
                clock=context.clock,
                exclude_channels=context.settings.exclude_channels,
                include_channels=include_channels,
                repeat=args.repeat,
            )
        except NoScoredMessagesError as error:
            raise ConfigError(
                "--strategy uncertain requires messages already scored with p_trash;"
                " run `infovore sift train` first once it exists"
            ) from error
        except MissingManifestError as error:
            raise ConfigError(
                f"no manifest.json in {error}; run `infovore sift export` first"
            ) from error

        hosts = args.hosts if args.hosts else default_hosts()
        try:
            servers = start_all(hosts, args.port, app)
        except OSError as error:
            raise ConfigError(f"could not bind port {args.port}: {error}") from error

        for server in servers:
            context.stdout.write(f"listening on {listening_url(server)}\n")
        context.stdout.write(
            f"serving batch {app.batch_name!r}: {len(app.messages())} messages"
            " (Ctrl-C to stop; run `infovore sift train` once everything is labeled)\n"
        )
        context.stdout.flush()
        try:
            block_until_interrupted(threading.Event())
        finally:
            shutdown_all(servers)
        return ExitCode.OK
