import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from infovore.db.archived import archived_clause, residue_clause
from infovore.db.channel_filter import exclude_channels_clause, include_channels_clause
from infovore.db.codec import to_db_time
from infovore.db.connection import transaction

VERDICTS: Final = ("good", "wrong", "made_up", "not_useful")
INTERFACES: Final = ("cited-only", "conversation")


@dataclass(frozen=True)
class ClaimIn:
    speaker: str
    statement: str
    message_ids: tuple[int, ...]


@dataclass(frozen=True)
class Rejection:
    speaker: str
    statement: str
    cited: str
    reason: str


@dataclass(frozen=True)
class ExchangeOutcome:
    outcome: str
    error: str | None
    windows: int
    input_tokens: int
    output_tokens: int
    seconds: float


@dataclass(frozen=True)
class ReviewRow:
    claim_id: int
    exchange_id: int
    speaker: str
    statement: str
    sources: list[tuple[int, str, str]]
    verdict: str | None


@dataclass(frozen=True)
class RunReport:
    run_id: int
    model_alias: str
    model_id: str
    model_id_source: str
    prompt_hash: str
    conversations: int
    failed: int
    claims: int
    rejected: int
    zero_claim_conversations: int
    verdicts: dict[str, int]
    interfaces: dict[str, dict[str, int]]
    unreviewed: int
    input_tokens: int
    output_tokens: int
    seconds: float


def create_run(
    conn: sqlite3.Connection,
    *,
    endpoint: str,
    model_alias: str,
    model_id: str,
    model_id_source: str,
    prompt_hash: str,
    selection: str,
    recipe: Mapping[str, object],
    now: datetime,
) -> int:
    cursor = conn.execute(
        "INSERT INTO claim_runs (started_at, endpoint, model_alias, model_id, model_id_source,"
        " prompt_hash, selection, recipe_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            to_db_time(now),
            endpoint,
            model_alias,
            model_id,
            model_id_source,
            prompt_hash,
            selection,
            json.dumps(recipe, sort_keys=True),
        ),
    )
    return int(cursor.lastrowid or 0)


def record_exchange(
    conn: sqlite3.Connection,
    run_id: int,
    exchange_id: int,
    outcome: ExchangeOutcome,
    claims: Sequence[ClaimIn],
    rejected: Sequence[Rejection],
) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT INTO claim_run_exchanges (run_id, exchange_id, outcome, error, windows,"
            " input_tokens, output_tokens, seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                exchange_id,
                outcome.outcome,
                outcome.error,
                outcome.windows,
                outcome.input_tokens,
                outcome.output_tokens,
                outcome.seconds,
            ),
        )
        for claim in claims:
            cursor = conn.execute(
                "INSERT INTO claims_v2 (run_id, exchange_id, speaker, statement)"
                " VALUES (?, ?, ?, ?)",
                (run_id, exchange_id, claim.speaker, claim.statement),
            )
            conn.executemany(
                "INSERT INTO claims_v2_sources (claim_id, message_id) VALUES (?, ?)",
                [(cursor.lastrowid, message_id) for message_id in claim.message_ids],
            )
        conn.executemany(
            "INSERT INTO claim_rejections (run_id, exchange_id, speaker, statement, cited, reason)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [(run_id, exchange_id, r.speaker, r.statement, r.cited, r.reason) for r in rejected],
        )


def record_review(
    conn: sqlite3.Connection,
    claim_id: int,
    verdict: str,
    now: datetime,
    interface: str = "conversation",
) -> None:
    conn.execute(
        "INSERT INTO claim_reviews (claim_id, verdict, reviewed_at, interface) VALUES (?, ?, ?, ?)",
        (claim_id, verdict, to_db_time(now), interface),
    )


def claim_exists(conn: sqlite3.Connection, claim_id: int) -> bool:
    return conn.execute("SELECT 1 FROM claims_v2 WHERE id = ?", (claim_id,)).fetchone() is not None


def run_ids(conn: sqlite3.Connection) -> list[int]:
    return [row[0] for row in conn.execute("SELECT id FROM claim_runs ORDER BY id")]


def latest_run_id(conn: sqlite3.Connection) -> int | None:
    ids = run_ids(conn)
    return ids[-1] if ids else None


def review_rows(conn: sqlite3.Connection, run_id: int) -> list[ReviewRow]:
    sources: dict[int, list[tuple[int, str, str]]] = {}
    for row in conn.execute(
        "SELECT s.claim_id, m.id, m.author_name_at_time, m.content FROM claims_v2_sources s"
        " JOIN claims_v2 c ON c.id = s.claim_id JOIN messages m ON m.id = s.message_id"
        " WHERE c.run_id = ? ORDER BY s.claim_id, m.id",
        (run_id,),
    ):
        sources.setdefault(row[0], []).append((row[1], row[2], row[3]))
    return [
        ReviewRow(
            row["id"],
            row["exchange_id"],
            row["speaker"],
            row["statement"],
            sources.get(row["id"], []),
            row["verdict"],
        )
        for row in conn.execute(
            "SELECT c.id, c.exchange_id, c.speaker, c.statement, r.verdict FROM claims_v2 c"
            " LEFT JOIN current_claim_reviews r ON r.claim_id = c.id"
            " WHERE c.run_id = ? ORDER BY c.id",
            (run_id,),
        )
    ]


def rejection_rows(conn: sqlite3.Connection, run_id: int) -> list[Rejection]:
    return [
        Rejection(r["speaker"], r["statement"], r["cited"], r["reason"])
        for r in conn.execute(
            "SELECT speaker, statement, cited, reason FROM claim_rejections"
            " WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
    ]


def report_rows(conn: sqlite3.Connection, run_id: int | None) -> list[RunReport]:
    runs = conn.execute(
        "SELECT id, model_alias, model_id, model_id_source, prompt_hash FROM claim_runs"
        " WHERE (? IS NULL OR id = ?) ORDER BY id",
        (run_id, run_id),
    ).fetchall()
    return [_report(conn, run) for run in runs]


def _report(conn: sqlite3.Connection, run: sqlite3.Row) -> RunReport:
    rid = run["id"]
    totals = conn.execute(
        "SELECT COALESCE(SUM(outcome = 'ok'), 0), COALESCE(SUM(outcome = 'failed'), 0),"
        " COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),"
        " COALESCE(SUM(seconds), 0.0) FROM claim_run_exchanges WHERE run_id = ?",
        (rid,),
    ).fetchone()
    claims = conn.execute("SELECT COUNT(*) FROM claims_v2 WHERE run_id = ?", (rid,)).fetchone()[0]
    rejected = conn.execute(
        "SELECT COUNT(*) FROM claim_rejections WHERE run_id = ?", (rid,)
    ).fetchone()[0]
    zero = conn.execute(
        "SELECT COUNT(*) FROM claim_run_exchanges e WHERE e.run_id = ? AND e.outcome = 'ok'"
        " AND NOT EXISTS (SELECT 1 FROM claims_v2 c WHERE c.run_id = e.run_id"
        " AND c.exchange_id = e.exchange_id)",
        (rid,),
    ).fetchone()[0]
    verdicts = dict.fromkeys(VERDICTS, 0)
    interfaces = {name: dict.fromkeys(VERDICTS, 0) for name in INTERFACES}
    for row in conn.execute(
        "SELECT r.interface, r.verdict, COUNT(*) FROM claims_v2 c JOIN current_claim_reviews r"
        " ON r.claim_id = c.id WHERE c.run_id = ? GROUP BY r.interface, r.verdict",
        (rid,),
    ):
        verdicts[row[1]] += row[2]
        interfaces[row[0]][row[1]] = row[2]
    return RunReport(
        rid,
        run["model_alias"],
        run["model_id"],
        run["model_id_source"],
        run["prompt_hash"],
        totals[0],
        totals[1],
        claims,
        rejected,
        zero,
        verdicts,
        interfaces,
        claims - sum(verdicts.values()),
        totals[2],
        totals[3],
        totals[4],
    )


def processed_ok(conn: sqlite3.Connection, model_id: str, prompt_hash: str) -> set[int]:
    """Exchanges some run with this model id and prompt hash finished successfully."""
    rows = conn.execute(
        "SELECT DISTINCT e.exchange_id FROM claim_run_exchanges e"
        " JOIN claim_runs r ON r.id = e.run_id"
        " WHERE r.model_id = ? AND r.prompt_hash = ? AND e.outcome = 'ok'",
        (model_id, prompt_hash),
    )
    return {row[0] for row in rows}


def archived_exchange_ids(
    conn: sqlite3.Connection, include: frozenset[str], exclude: frozenset[str]
) -> list[int]:
    """Current archived exchanges in the named channels, by id; excluded channels never."""
    inc, inc_params = include_channels_clause("exchanges.channel_id", include)
    exc, exc_params = exclude_channels_clause("exchanges.channel_id", exclude)
    rows = conn.execute(
        f"SELECT id FROM current_exchanges AS exchanges WHERE {archived_clause('exchanges.id')}"
        f"{inc}{exc} ORDER BY id",
        (*inc_params, *exc_params),
    )
    return [row[0] for row in rows]


ARCHIVE_SORTS: Final = ("relevant", "undecided")


def archive_exchange_ids(
    conn: sqlite3.Connection, which: str, exclude: frozenset[str]
) -> list[int]:
    if which == "relevant":
        sort = f"{archived_clause('exchanges.id')} AND NOT ({residue_clause('exchanges.id')})"
    elif which == "undecided":
        sort = residue_clause("exchanges.id")
    else:
        raise ValueError(f"unknown archive sort: {which}")
    clause, params = exclude_channels_clause("exchanges.channel_id", exclude)
    rows = conn.execute(
        f"SELECT id FROM current_exchanges AS exchanges WHERE {sort}{clause} ORDER BY id",
        params,
    )
    return [row[0] for row in rows]


@dataclass(frozen=True)
class ReviewedClaim:
    review_id: int
    exchange_id: int
    speaker: str
    statement: str
    verdict: str
    message_ids: tuple[int, ...]
    interface: str


def reviewed_claims(conn: sqlite3.Connection, runs: Sequence[int]) -> list[ReviewedClaim]:
    marks = ",".join("?" for _ in runs)
    sources: dict[int, list[int]] = {}
    for claim_id, message_id in conn.execute(
        "SELECT s.claim_id, s.message_id FROM claims_v2_sources s JOIN claims_v2 c"
        f" ON c.id = s.claim_id WHERE c.run_id IN ({marks}) ORDER BY s.claim_id, s.message_id",
        list(runs),
    ):
        sources.setdefault(claim_id, []).append(message_id)
    return [
        ReviewedClaim(
            row["rid"],
            row["exchange_id"],
            row["speaker"],
            row["statement"],
            row["verdict"],
            tuple(sources.get(row["id"], ())),
            row["interface"],
        )
        for row in conn.execute(
            "SELECT c.id, c.exchange_id, c.speaker, c.statement, r.id AS rid, r.verdict,"
            " r.interface"
            " FROM claims_v2 c JOIN claim_reviews r ON r.claim_id = c.id"
            " WHERE r.id = (SELECT MAX(id) FROM claim_reviews WHERE claim_id = c.id)"
            f" AND c.run_id IN ({marks}) ORDER BY c.exchange_id, c.id",
            list(runs),
        )
    ]
