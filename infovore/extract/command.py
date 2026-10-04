import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.config import ConfigError, Stage
from infovore.db.batches import record_extraction_batch
from infovore.db.codec import to_db_time
from infovore.extract.llm_extractor import LLMClaimExtractor
from infovore.extract.prompt import LIVE_PROMPT_VERSION, PROMPTS
from infovore.extract.prompt_compare import (
    DEFAULT_COMPARE_LIMIT,
    PromptComparison,
    format_comparison,
    run_arm,
    sample_extracted_exchanges,
)
from infovore.extract.runner import (
    ExchangeClaimed,
    ExchangeFailed,
    ExchangeSkipped,
    ExtractionEvent,
    ExtractionStarted,
    PromptNotPromotedError,
    TrialSampleStrategy,
    UntriagedExchangesError,
    run_extraction,
    select_trial_sample_origins,
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
        parser.add_argument(
            "--compare-prompt",
            type=str,
            default=None,
            dest="compare_prompt",
            help="re-extract done exchanges under the live prompt and this one; writes nothing",
        )
        parser.add_argument("--compare-limit", type=int, default=DEFAULT_COMPARE_LIMIT)
        parser.add_argument(
            "--compare-dump",
            type=str,
            default=None,
            dest="compare_dump",
            help="directory to write each arm's claims as JSONL; the aggregate is not the evidence",
        )
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

    async def _compare_prompt(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode, stage_backend

        candidate = args.compare_prompt
        if candidate not in PROMPTS:
            raise ConfigError(f"unknown prompt version {candidate}; have {sorted(PROMPTS)}")
        exchange_ids = sample_extracted_exchanges(context.conn, args.compare_limit)
        if not exchange_ids:
            raise ConfigError("no already-extracted exchanges to compare against")

        backend = await stage_backend(context, Stage.EXTRACT)
        dump_dir = Path(args.compare_dump) if args.compare_dump else None
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
        arms = []
        for version in (LIVE_PROMPT_VERSION, candidate):
            extractor = LLMClaimExtractor(backend, prompt_version=version)
            arms.append(
                await run_arm(
                    context.conn,
                    extractor,
                    version,
                    exchange_ids,
                    dump=dump_dir / f"{version}.jsonl" if dump_dir else None,
                )
            )
        comparison = PromptComparison(exchange_ids=tuple(exchange_ids), arms=tuple(arms))
        for line in format_comparison(comparison):
            context.stdout.write(f"{line}\n")
        return int(ExitCode.OK)

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode, _say, stage_backend

        if args.compare_prompt:
            return await self._compare_prompt(context, args)

        mode = RunMode(args.mode)
        if mode is RunMode.TRIAL and args.sample is None and not args.exchange_id:
            raise ConfigError("--mode trial requires --sample or --exchange-id")

        now = context.clock.now()
        batch_id = to_db_time(now)
        sample = args.sample if mode is RunMode.TRIAL else None
        record_extraction_batch(
            context.conn,
            batch_id,
            mode,
            args.strategy if sample is not None else None,
            args.seed,
            sample,
            now,
        )

        exchange_ids: list[int] | None = None
        sampled_origins: dict[int, str] = {}
        if mode is RunMode.TRIAL:
            ids: list[int] = []
            if args.sample is not None:
                sampled_origins = select_trial_sample_origins(
                    context.conn,
                    args.sample,
                    args.seed,
                    min_score=args.min_score,
                    max_score=args.max_score,
                    strategy=TrialSampleStrategy(args.strategy),
                    exclude_channels=context.settings.exclude_channels,
                )
                ids.extend(sampled_origins)
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
                progress=lambda event: _say(context.stdout, _describe_extraction_event(event)),
                batch_id=batch_id,
                rules=context.settings.triage_rules,
                sampled_by=sampled_origins,
                exclude_channels=context.settings.exclude_channels,
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
