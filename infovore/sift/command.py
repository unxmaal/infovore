import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.config import ConfigError
from infovore.sift.export import export_batch
from infovore.sift.sampling import (
    DEFAULT_MIX_FRACTION_UNCERTAIN,
    NoScoredMessagesError,
    SiftStrategy,
)

if TYPE_CHECKING:
    from infovore.cli import AppContext

DEFAULT_SIFT_SIZE = 1000


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

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
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
