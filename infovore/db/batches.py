import sqlite3
from datetime import datetime

from infovore.db.codec import to_db_time
from infovore.rows import RunMode


def record_extraction_batch(
    conn: sqlite3.Connection,
    batch_id: str,
    mode: RunMode,
    strategy: str | None,
    seed: int | None,
    sample: int | None,
    created_at: datetime,
) -> None:
    """Describe one `extract` invocation (issue #107), written before any
    exchange is processed so an interrupted run still leaves its batch
    described: an `extraction_runs.batch_id` row only exists once a run is
    actually recorded, which a crash right after startup can prevent
    entirely. `strategy`/`sample` are `NULL` for a live-mode batch, or a
    trial batch with no `--sample` (explicit `--exchange-id` only, so no
    sampling strategy actually applied). `ON CONFLICT ... DO NOTHING`
    tolerates re-describing the same `batch_id` (harmless in practice, since
    a `batch_id` is a timestamp) rather than raising on the rare exact
    collision."""
    conn.execute(
        "INSERT INTO extraction_batches (batch_id, mode, strategy, seed, sample, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (batch_id) DO NOTHING",
        (batch_id, mode.value, strategy, seed, sample, to_db_time(created_at)),
    )
