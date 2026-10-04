import sqlite3
from datetime import datetime

from infovore.triage.cascade import (
    run_cascade,
    try_fit,
    tune_high,
    tuning_samples,
    write_outcomes,
)
from infovore.triage.lexicon import load_lexicon


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
    conn: sqlite3.Connection, exclude_channels: frozenset[str], at: datetime
) -> int:
    ids = unscored_exchange_ids(conn)
    if not ids:
        return 0
    lexicon = load_lexicon()
    t_high = tune_high(tuning_samples(conn, lexicon, exclude_channels))
    fit, _ = try_fit(conn, exclude_channels)
    outcomes = run_cascade(conn, ids, lexicon, t_high, fit, exclude_channels)
    write_outcomes(conn, outcomes, lexicon, t_high, fit, at)
    return len(ids)
