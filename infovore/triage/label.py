import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from infovore.db.claims import claims_for_run, get_run
from infovore.db.labels import set_label
from infovore.rows import Label, LabelSource, Novelty, RunOutcome

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
