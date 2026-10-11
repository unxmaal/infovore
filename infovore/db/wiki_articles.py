import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime


def create_article_run(
    conn: sqlite3.Connection,
    *,
    endpoint: str,
    model_alias: str,
    model_id: str,
    prompt_hash: str,
    tag_run_id: int,
    now: datetime,
) -> int:
    cursor = conn.execute(
        "INSERT INTO article_runs (started_at, endpoint, model_alias, model_id, prompt_hash,"
        " tag_run_id) VALUES (?, ?, ?, ?, ?, ?)",
        (now.isoformat(), endpoint, model_alias, model_id, prompt_hash, tag_run_id),
    )
    return int(cursor.lastrowid or 0)


def record_section(
    conn: sqlite3.Connection,
    article_run_id: int,
    topic: str,
    section: str,
    sentences: Sequence[tuple[str, Sequence[int]]],
    dropped: int,
) -> None:
    conn.execute(
        "INSERT INTO article_sections (article_run_id, topic, section, sentences_json, dropped)"
        " VALUES (?, ?, ?, ?, ?)",
        (
            article_run_id,
            topic,
            section,
            json.dumps([[text, list(ids)] for text, ids in sentences]),
            dropped,
        ),
    )


def latest_article_run(
    conn: sqlite3.Connection, model_id: str, prompt_hash: str, tag_run_id: int
) -> int | None:
    row = conn.execute(
        "SELECT id FROM article_runs WHERE model_id = ? AND prompt_hash = ? AND tag_run_id = ?"
        " ORDER BY id DESC LIMIT 1",
        (model_id, prompt_hash, tag_run_id),
    ).fetchone()
    return None if row is None else int(row[0])


def run_sections(conn: sqlite3.Connection, article_run_id: int) -> set[tuple[str, str]]:
    rows = conn.execute(
        "SELECT topic, section FROM article_sections WHERE article_run_id = ?", (article_run_id,)
    )
    return {(topic, section) for topic, section in rows}


def sections_for(
    conn: sqlite3.Connection, article_run_ids: Sequence[int]
) -> dict[str, dict[str, list[tuple[str, list[int]]]]]:
    if not article_run_ids:
        return {}
    marks = ",".join("?" * len(article_run_ids))
    rows = conn.execute(
        "SELECT topic, section, sentences_json FROM article_sections"
        f" WHERE article_run_id IN ({marks}) ORDER BY article_run_id",
        list(article_run_ids),
    )
    out: dict[str, dict[str, list[tuple[str, list[int]]]]] = {}
    for topic, section, sentences_json in rows:
        out.setdefault(topic, {})[section] = [
            (text, list(ids)) for text, ids in json.loads(sentences_json)
        ]
    return out
