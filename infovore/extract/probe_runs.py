"""Turn one call pair's usage into a persistable `probe_runs` row."""

from datetime import datetime

from infovore.extract.protocol import ProbeUsage
from infovore.rows import ProbeRunRow, RunOutcome


def probe_run_row(
    usage: ProbeUsage,
    *,
    started_at: datetime,
    finished_at: datetime,
    claim_count: int,
    batched: bool,
    error: str | None,
) -> ProbeRunRow:
    return ProbeRunRow(
        id=None,
        probe_model=usage.probe_model,
        judge_model=usage.judge_model,
        started_at=started_at,
        finished_at=finished_at,
        claim_count=claim_count,
        batched=batched,
        recall_input_tokens=usage.recall_input_tokens,
        recall_output_tokens=usage.recall_output_tokens,
        judge_input_tokens=usage.judge_input_tokens,
        judge_output_tokens=usage.judge_output_tokens,
        cost_usd=usage.cost_usd,
        outcome=RunOutcome.FAILED if error is not None else RunOutcome.OK,
        error=error,
    )
