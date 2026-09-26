import argparse
from typing import TYPE_CHECKING

from infovore.config import ConfigError, Stage
from infovore.extract.llm_extractor import LLMClaimExtractor
from infovore.extract.runner import PromptNotPromotedError, run_extraction, select_trial_sample
from infovore.rows import RunMode

if TYPE_CHECKING:
    from infovore.cli import AppContext


class ExtractCommand:
    name = "extract"
    help = "extract claims from pending and stale exchanges"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--mode", choices=["trial", "live"], default="live")
        parser.add_argument("--sample", type=int, default=None)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--exchange-id", type=int, action="append", default=[])

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode, stage_backend

        mode = RunMode(args.mode)
        if mode is RunMode.TRIAL and args.sample is None and not args.exchange_id:
            raise ConfigError("--mode trial requires --sample or --exchange-id")

        exchange_ids: list[int] | None = None
        if mode is RunMode.TRIAL:
            ids: list[int] = []
            if args.sample is not None:
                ids.extend(select_trial_sample(context.conn, args.sample, args.seed))
            ids.extend(args.exchange_id)
            exchange_ids = sorted(set(ids))

        stage_settings = context.settings.stages[Stage.EXTRACT]
        backend = await stage_backend(context, Stage.EXTRACT)
        extractor = LLMClaimExtractor(backend)

        try:
            report = await run_extraction(
                context.conn,
                extractor,
                context.clock,
                context.sleeper,
                mode=mode,
                model_label=stage_settings.model,
                batch_size=context.settings.batch_size,
                max_retries=context.settings.max_retries,
                concurrency=stage_settings.concurrency,
                exchange_ids=exchange_ids,
            )
        except PromptNotPromotedError as error:
            raise ConfigError(
                f"prompt version {error.version} is not promoted;"
                f" run `infovore promote --prompt-version {error.version}`"
            ) from error

        context.stdout.write(
            f"processed={report.processed} succeeded={report.succeeded}"
            f" failed={report.failed} skipped={report.skipped}"
            f" claims_recorded={report.claims_recorded} pauses={report.pauses}\n"
            f"run ids: {', '.join(str(run_id) for run_id in report.run_ids) or 'none'}\n"
        )
        return ExitCode.FAILURE if report.failed else ExitCode.OK
