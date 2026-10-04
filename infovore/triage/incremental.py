import sqlite3
from datetime import datetime
from pathlib import Path

from infovore.triage.cascade import (
    run_cascade,
    tune_high,
    tuning_samples,
    write_outcomes,
)
from infovore.triage.embed_stage import build_embed_stage
from infovore.triage.lexicon import load_lexicon
from infovore.triage.short_report import short_limit


def unscored_exchange_ids(conn: sqlite3.Connection) -> list[int]:
    return [
        row["id"]
        for row in conn.execute(
            "SELECT e.id FROM current_exchanges e WHERE NOT EXISTS ("
            " SELECT 1 FROM annotations a WHERE a.subject_kind = 'exchange'"
            " AND a.subject_id = e.id AND a.scorer LIKE 'relevance\\_%' ESCAPE '\\')"
            " ORDER BY e.id"
        )
    ]


def cascade_new_exchanges(
    conn: sqlite3.Connection, exclude_channels: frozenset[str], cache_path: Path, at: datetime
) -> int:
    ids = unscored_exchange_ids(conn)
    if not ids:
        return 0
    lexicon = load_lexicon(conn)
    t_high = tune_high(tuning_samples(conn, lexicon, exclude_channels))
    stage = build_embed_stage(conn, exclude_channels, cache_path)
    limit = short_limit(conn, lexicon, t_high, exclude_channels)
    outcomes = run_cascade(conn, ids, lexicon, t_high, stage, exclude_channels, limit)
    write_outcomes(conn, outcomes, lexicon, t_high, stage, at, limit)
    return len(ids)
