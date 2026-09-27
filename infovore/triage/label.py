import argparse
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, TextIO

from infovore.config import ConfigError
from infovore.db.claims import claims_for_run, get_run
from infovore.db.labels import set_label
from infovore.db.run_selection import (
    InvalidRunSelectorError,
    NoTrialBatchError,
    resolve_run_selector,
)
from infovore.rows import Label, LabelSource, Novelty, RunOutcome

if TYPE_CHECKING:
    from infovore.cli import AppContext

_LORE_VERDICTS = (Novelty.UNKNOWN, Novelty.PARTIAL, Novelty.CONTRADICTS)


class UnknownRunError(Exception):
    def __init__(self, run_id: int) -> None:
        super().__init__(f"unknown run id: {run_id}")
        self.run_id = run_id


@dataclass(frozen=True)
class DeriveReport:
    lore: int = 0
    noise: int = 0
    skipped_reasons: dict[int, str] = field(default_factory=dict)

    @property
    def skipped(self) -> int:
        return len(self.skipped_reasons)


def derive_labels_from_runs(
    conn: sqlite3.Connection,
    run_ids: Sequence[int],
    at: datetime,
    progress: Callable[[int, str], None] | None = None,
) -> DeriveReport:
    lore = 0
    noise = 0
    skipped_reasons: dict[int, str] = {}
    for run_id in run_ids:
        run = get_run(conn, run_id)
        if run is None:
            raise UnknownRunError(run_id)
        if run.outcome is RunOutcome.FAILED:
            skipped_reasons[run_id] = "failed run"
            if progress is not None:
                progress(run_id, "skipped (failed run)")
            continue
        claims = claims_for_run(conn, run_id)
        if any(claim.novelty in _LORE_VERDICTS for claim in claims):
            set_label(
                conn,
                run.exchange_id,
                Label.LORE,
                LabelSource.LLM,
                f"run:{run_id} model:{run.model}",
                at,
            )
            lore += 1
            if progress is not None:
                progress(run_id, "lore")
        elif all(claim.novelty is Novelty.KNOWN for claim in claims):
            set_label(
                conn,
                run.exchange_id,
                Label.NOISE,
                LabelSource.LLM,
                f"run:{run_id} model:{run.model}",
                at,
            )
            noise += 1
            if progress is not None:
                progress(run_id, "noise")
        else:
            skipped_reasons[run_id] = "unprobed claims"
            if progress is not None:
                progress(run_id, "skipped (unprobed claims)")
    return DeriveReport(lore=lore, noise=noise, skipped_reasons=skipped_reasons)


def _say(stdout: TextIO, line: str) -> None:
    stdout.write(line + "\n")
    stdout.flush()


def _format_skips(skipped_reasons: dict[int, str]) -> str:
    return " ".join(f"{run_id}:{reason}" for run_id, reason in sorted(skipped_reasons.items()))


class LabelCommand:
    name = "label"
    help = "record human or LLM-derived lore/noise labels on exchanges"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--from-runs",
            nargs="*",
            dest="from_runs",
            default=None,
            metavar="RUN_ID_OR_RANGE",
        )
        parser.add_argument("--exchange-id", type=int, default=None)
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--lore", action="store_true")
        group.add_argument("--noise", action="store_true")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        if args.from_runs is not None:
            if args.exchange_id is not None:
                raise ConfigError("--from-runs and --exchange-id are mutually exclusive")
            if args.lore or args.noise:
                raise ConfigError("--lore/--noise apply only with --exchange-id")
            try:
                run_ids = resolve_run_selector(context.conn, args.from_runs)
            except NoTrialBatchError as error:
                raise ConfigError(
                    "no trial batch found; pass --from-runs explicitly, or run"
                    " `infovore extract --mode trial` first"
                ) from error
            except InvalidRunSelectorError as error:
                raise ConfigError(str(error)) from error
            try:
                report = derive_labels_from_runs(
                    context.conn,
                    run_ids,
                    context.clock.now(),
                    progress=lambda run_id, outcome: _say(
                        context.stdout, f"run {run_id}: {outcome}"
                    ),
                )
            except UnknownRunError as error:
                raise ConfigError(str(error)) from error
            summary = f"labeled: lore={report.lore} noise={report.noise} skipped={report.skipped}"
            if report.skipped_reasons:
                summary += f" ({_format_skips(report.skipped_reasons)})"
            _say(context.stdout, summary)
            return ExitCode.OK

        if args.exchange_id is not None:
            if not (args.lore or args.noise):
                raise ConfigError("--exchange-id requires --lore or --noise")
            label = Label.LORE if args.lore else Label.NOISE
            set_label(
                context.conn,
                args.exchange_id,
                label,
                LabelSource.HUMAN,
                None,
                context.clock.now(),
            )
            _say(
                context.stdout,
                f"labeled exchange {args.exchange_id} as {label.value} (human)",
            )
            return ExitCode.OK

        raise ConfigError("label requires --from-runs or --exchange-id")
