import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.config import ConfigError
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

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        if args.sift_command == "import":
            return self._import(context, args)
        return self._export(context, args)

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
