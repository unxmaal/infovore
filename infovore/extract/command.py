import argparse
from typing import TYPE_CHECKING

from infovore.config import ConfigError, Stage
from infovore.db.codec import to_db_time
from infovore.extract.llm_extractor import LLMClaimExtractor
from infovore.extract.runner import (
    ExchangeClaimed,
    ExchangeFailed,
    ExchangeSkipped,
    ExtractionEvent,
    ExtractionStarted,
    NoScoredExchangesError,
    PromptNotPromotedError,
    TrialSampleStrategy,
    UntriagedExchangesError,
    run_extraction,
    select_trial_sample,
)
from infovore.rows import RunMode

if TYPE_CHECKING:
    from infovore.cli import AppContext


def _counter(index: int, total: int | None) -> str:
    return f"{index}/{total}" if total is not None else f"{index} done"


def _describe_extraction_event(event: ExtractionEvent) -> str:
    match event:
        case ExtractionStarted(mode=RunMode.TRIAL, total=total):
            return f"extract: trial mode, {total} exchanges queued"
        case ExtractionStarted():
            return "extract: live mode, draining queued exchanges"
        case ExchangeClaimed(exchange_id=exchange_id, claims=claims, index=index, total=total):
            return f"exchange {exchange_id}: {claims} claims ({_counter(index, total)})"
        case ExchangeSkipped(exchange_id=exchange_id, index=index, total=total):
            return f"exchange {exchange_id}: skipped ({_counter(index, total)})"
        case ExchangeFailed(exchange_id=exchange_id, kind=kind, index=index, total=total):
            return f"exchange {exchange_id}: failed: {kind} ({_counter(index, total)})"
        case _:
            return (
                f"exchange {event.exchange_id}: paused {event.retry_after}s (usage limit)"
                f" ({_counter(event.index, event.total)})"
            )


class ExtractCommand:
    name = "extract"
    help = "extract claims from pending and stale exchanges"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--mode", choices=["trial", "live"], default="live")
        parser.add_argument("--sample", type=int, default=None)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--exchange-id", type=int, action="append", default=[])
        parser.add_argument("--min-score", type=float, default=None)
        parser.add_argument("--max-score", type=float, default=None)
        parser.add_argument(
            "--strategy",
            choices=[strategy.value for strategy in TrialSampleStrategy],
            default=TrialSampleStrategy.STRATIFIED.value,
        )

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode, _say, stage_backend

        mode = RunMode(args.mode)
        if mode is RunMode.TRIAL and args.sample is None and not args.exchange_id:
            raise ConfigError("--mode trial requires --sample or --exchange-id")

        exchange_ids: list[int] | None = None
        if mode is RunMode.TRIAL:
            ids: list[int] = []
            if args.sample is not None:
                try:
                    ids.extend(
                        select_trial_sample(
                            context.conn,
                            args.sample,
                            args.seed,
                            min_score=args.min_score,
                            max_score=args.max_score,
                            strategy=TrialSampleStrategy(args.strategy),
                        )
                    )
                except NoScoredExchangesError as error:
                    raise ConfigError(
                        "--strategy uncertain requires a trained model;"
                        " run `infovore triage --train` first"
                    ) from error
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
                min_score=context.settings.triage_min_score,
                min_p_lore=context.settings.triage_min_p_lore,
                exchange_ids=exchange_ids,
                progress=lambda event: _say(context.stdout, _describe_extraction_event(event)),
                batch_id=to_db_time(context.clock.now()),
            )
        except PromptNotPromotedError as error:
            raise ConfigError(
                f"prompt version {error.version} is not promoted;"
                f" run `infovore promote --prompt-version {error.version}`"
            ) from error
        except UntriagedExchangesError as error:
            raise ConfigError(
                "pending exchanges are untriaged; run `infovore triage` first"
            ) from error

        context.stdout.write(
            f"processed={report.processed} succeeded={report.succeeded}"
            f" failed={report.failed} skipped={report.skipped}"
            f" claims_recorded={report.claims_recorded} pauses={report.pauses}\n"
            f"run ids: {', '.join(str(run_id) for run_id in report.run_ids) or 'none'}\n"
        )
        return ExitCode.FAILURE if report.failed else ExitCode.OK
