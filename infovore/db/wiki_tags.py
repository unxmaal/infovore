import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime


def create_tag_run(
    conn: sqlite3.Connection,
    *,
    endpoint: str,
    model_alias: str,
    model_id: str,
    prompt_hash: str,
    now: datetime,
) -> int:
    cursor = conn.execute(
        "INSERT INTO tag_runs (started_at, endpoint, model_alias, model_id, prompt_hash)"
        " VALUES (?, ?, ?, ?, ?)",
        (now.isoformat(), endpoint, model_alias, model_id, prompt_hash),
    )
    return int(cursor.lastrowid or 0)


def record_tags(
    conn: sqlite3.Connection, tag_run_id: int, tags: Mapping[int, Sequence[str]]
) -> None:
    conn.executemany(
        "INSERT INTO claim_tags (tag_run_id, claim_id, tags_json) VALUES (?, ?, ?)",
        [(tag_run_id, claim_id, json.dumps(list(names))) for claim_id, names in tags.items()],
    )


def tagged_claim_ids(conn: sqlite3.Connection, model_id: str, prompt_hash: str) -> set[int]:
    rows = conn.execute(
        "SELECT ct.claim_id FROM claim_tags ct JOIN tag_runs tr ON tr.id = ct.tag_run_id"
        " WHERE tr.model_id = ? AND tr.prompt_hash = ?",
        (model_id, prompt_hash),
    )
    return {row[0] for row in rows}


def tags_for(conn: sqlite3.Connection, tag_run_ids: Sequence[int]) -> dict[int, list[str]]:
    if not tag_run_ids:
        return {}
    marks = ",".join("?" * len(tag_run_ids))
    rows = conn.execute(
        f"SELECT claim_id, tags_json FROM claim_tags WHERE tag_run_id IN ({marks})"
        " ORDER BY tag_run_id",
        list(tag_run_ids),
    )
    return {claim_id: json.loads(tags_json) for claim_id, tags_json in rows}
