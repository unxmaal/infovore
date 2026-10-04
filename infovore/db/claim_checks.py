import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from infovore.db.codec import to_db_time
from infovore.db.connection import transaction


@dataclass(frozen=True)
class CheckRow:
    claim_id: int
    verdict: str
    overlap: float
    facts: list[dict[str, object]]


def record_checks(
    conn: sqlite3.Connection,
    rows: Sequence[CheckRow],
    recipe: Mapping[str, object],
    now: datetime,
) -> None:
    with transaction(conn):
        conn.executemany(
            "INSERT INTO claim_checks (claim_id, verdict, overlap, facts_json, recipe_json,"
            " checked_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    r.claim_id,
                    r.verdict,
                    r.overlap,
                    json.dumps(r.facts),
                    json.dumps(recipe, sort_keys=True),
                    to_db_time(now),
                )
                for r in rows
            ],
        )


def current_checks(conn: sqlite3.Connection, run_id: int) -> dict[int, CheckRow]:
    return {
        r["claim_id"]: CheckRow(
            r["claim_id"], r["verdict"], r["overlap"], json.loads(r["facts_json"])
        )
        for r in conn.execute(
            "SELECT k.* FROM current_claim_checks k JOIN claims_v2 c ON c.id = k.claim_id"
            " WHERE c.run_id = ?",
            (run_id,),
        )
    }
