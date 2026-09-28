import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.config import ConfigError
from infovore.sift.citations import derive_citation_labels
from infovore.sift.export import export_batch
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
from infovore.sift.train import (
    DEFAULT_CONFUSION_THRESHOLD,
    DEFAULT_HUMAN_WEIGHT,
    DiscardRow,
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


def _render_channel_stats(channel_stats: dict[int, Metrics]) -> str:
    lines = ["per channel (at the confusion threshold):"]
    for channel_id in sorted(channel_stats):
        metric = channel_stats[channel_id]
        lines.append(
            f"  channel {channel_id}: precision={metric.precision:.3f} recall={metric.recall:.3f}"
            f" (tp={metric.tp} fp={metric.fp} fn={metric.fn} tn={metric.tn})"
        )
    return "\n".join(lines)


def _render_discard_pile(rows: Sequence[DiscardRow]) -> str:
    lines = ["discard pile (share of held-out keep messages that would be discarded):"]
    for row in rows:
        human = f"{row.human_keep_discarded:.3f}" if row.human_keep_discarded is not None else "n/a"
        citation = (
            f"{row.citation_keep_discarded:.3f}"
            if row.citation_keep_discarded is not None
            else "n/a"
        )
        lines.append(
            f"  p_trash>={row.threshold:.2f}: human_keep_lost={human} citation_keep_lost={citation}"
        )
    return "\n".join(lines)


def _render_message_top_tokens(tokens: Sequence[MessageTokenInfo]) -> str:
    lines = ["top tokens:"]
    for token in tokens:
        lines.append(
            f"  {token.token}: p_trash={token.probability:.3f}"
            f" trash={token.trash_count} keep={token.keep_count}"
        )
    return "\n".join(lines)


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
        export_parser.add_argument(
            "--mix", type=float, default=DEFAULT_MIX_FRACTION_UNCERTAIN, dest="mix"
        )
        export_parser.add_argument("--out", type=str, default=None, dest="out")

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
            "--human-weight", type=int, default=DEFAULT_HUMAN_WEIGHT, dest="human_weight"
        )

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        if args.sift_command == "import":
            return self._import(context, args)
        if args.sift_command == "citations":
            return self._citations(context, args)
        if args.sift_command == "train":
            return self._train(context, args)
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

        try:
            report = train_and_store(context.conn, context.clock, human_weight=args.human_weight)
        except InsufficientLabelsError as error:
            raise ConfigError(str(error)) from error

        context.stdout.write(
            f"trained message model v{report.version}: labels_used={report.labels_used}"
            f" holdout_size={report.holdout_size} human_weight={report.human_weight}\n"
        )
        context.stdout.write(_render_message_metrics_table(report.overall_metrics) + "\n")
        context.stdout.write(
            _render_message_confusion(report.confusion, DEFAULT_CONFUSION_THRESHOLD) + "\n"
        )
        context.stdout.write(_render_message_auc("overall", report.overall_auc) + "\n")
        context.stdout.write(_render_message_auc("human-labeled holdout", report.human_auc) + "\n")
        context.stdout.write(
            _render_message_auc("citation-labeled holdout", report.citation_auc) + "\n"
        )
        context.stdout.write(_render_channel_stats(report.channel_stats) + "\n")
        context.stdout.write(_render_discard_pile(report.discard_pile) + "\n")
        context.stdout.write(_render_message_top_tokens(report.top_tokens) + "\n")

        loaded = load_latest_message_model(context.conn)
        assert loaded is not None
        version, model = loaded
        scored = score_all(context.conn, model, version, workers=context.settings.workers)
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
        try:
            report = export_batch(
                context.conn,
                size=args.size,
                strategy=strategy,
                seed=args.seed,
                mix=args.mix,
                out_dir=out_dir,
                now=context.clock.now(),
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
