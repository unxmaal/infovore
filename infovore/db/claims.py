import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from infovore.db.codec import from_db_time, to_db_time
from infovore.db.connection import transaction
from infovore.rows import ClaimKind, ClaimRow, ExtractionRunRow, Novelty, RunMode, RunOutcome

_WORD_RE = re.compile(r"[\w\-./]+")


class PromptVersionConflictError(Exception):
    pass


class UnknownPromptVersionError(Exception):
    pass


@dataclass(frozen=True)
class NewClaim:
    exchange_id: int
    statement: str
    subject: str
    kind: ClaimKind
    confidence: float
    probe_question: str
    permalink: str
    supersedes_claim_id: int | None
    source_message_ids: tuple[int, ...]


@dataclass(frozen=True)
class RecordedRun:
    run_id: int
    claim_ids: tuple[int, ...]


def register_prompt_version(
    conn: sqlite3.Connection, version: str, text_sha256: str, at: datetime
) -> None:
    row = conn.execute(
        "SELECT text_sha256 FROM prompt_versions WHERE version = ?", (version,)
    ).fetchone()
    if row is not None:
        if row["text_sha256"] != text_sha256:
            raise PromptVersionConflictError(version)
        return
    conn.execute(
        "INSERT INTO prompt_versions (version, text_sha256, created_at, promoted_at)"
        " VALUES (?, ?, ?, NULL)",
        (version, text_sha256, to_db_time(at)),
    )


def promote_prompt_version(conn: sqlite3.Connection, version: str, at: datetime) -> None:
    cursor = conn.execute(
        "UPDATE prompt_versions SET promoted_at = ? WHERE version = ?",
        (to_db_time(at), version),
    )
    if cursor.rowcount == 0:
        raise UnknownPromptVersionError(version)


def live_prompt_version(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT version FROM prompt_versions WHERE promoted_at IS NOT NULL"
        " ORDER BY promoted_at DESC LIMIT 1"
    ).fetchone()
    return row["version"] if row is not None else None


def record_run(
    conn: sqlite3.Connection, run: ExtractionRunRow, claims: Sequence[NewClaim]
) -> RecordedRun:
    if run.outcome is RunOutcome.FAILED and claims:
        raise ValueError("a failed run may not record claims")
    for claim in claims:
        if not claim.source_message_ids:
            raise ValueError("a claim must have at least one source message")
    with transaction(conn):
        cursor = conn.execute(
            "INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at,"
            " finished_at, input_tokens, output_tokens, mode, outcome, error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run.exchange_id,
                run.model,
                run.prompt_version,
                to_db_time(run.started_at),
                to_db_time(run.finished_at),
                run.input_tokens,
                run.output_tokens,
                run.mode.value,
                run.outcome.value,
                run.error,
            ),
        )
        run_id = cursor.lastrowid
        assert run_id is not None
        claim_ids: list[int] = []
        for claim in claims:
            claim_cursor = conn.execute(
                "INSERT INTO claims (exchange_id, extraction_run_id, statement, subject, kind,"
                " confidence, probe_question, permalink, supersedes_claim_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim.exchange_id,
                    run_id,
                    claim.statement,
                    claim.subject,
                    claim.kind.value,
                    claim.confidence,
                    claim.probe_question,
                    claim.permalink,
                    claim.supersedes_claim_id,
                ),
            )
            claim_id = claim_cursor.lastrowid
            assert claim_id is not None
            claim_ids.append(claim_id)
            for message_id in claim.source_message_ids:
                conn.execute(
                    "INSERT INTO claim_sources (claim_id, message_id) VALUES (?, ?)",
                    (claim_id, message_id),
                )
    return RecordedRun(run_id, tuple(claim_ids))


def _row_to_claim(row: sqlite3.Row) -> ClaimRow:
    return ClaimRow(
        id=row["id"],
        exchange_id=row["exchange_id"],
        extraction_run_id=row["extraction_run_id"],
        statement=row["statement"],
        subject=row["subject"],
        kind=ClaimKind(row["kind"]),
        confidence=row["confidence"],
        probe_question=row["probe_question"],
        permalink=row["permalink"],
        supersedes_claim_id=row["supersedes_claim_id"],
        novelty=Novelty(row["novelty"]),
        probe_model=row["probe_model"],
        probe_answer=row["probe_answer"],
        probed_at=from_db_time(row["probed_at"]),
        probe_error=row["probe_error"],
        retracted_at=from_db_time(row["retracted_at"]),
        retraction_reason=row["retraction_reason"],
    )


def get_claim(conn: sqlite3.Connection, claim_id: int) -> ClaimRow | None:
    row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
    return _row_to_claim(row) if row is not None else None


def claim_source_ids(conn: sqlite3.Connection, claim_id: int) -> list[int]:
    rows = conn.execute(
        "SELECT message_id FROM claim_sources WHERE claim_id = ? ORDER BY message_id",
        (claim_id,),
    ).fetchall()
    return [row["message_id"] for row in rows]


def claims_for_run(conn: sqlite3.Connection, run_id: int) -> list[ClaimRow]:
    rows = conn.execute(
        "SELECT * FROM claims WHERE extraction_run_id = ? ORDER BY id", (run_id,)
    ).fetchall()
    return [_row_to_claim(row) for row in rows]


def unprobed_claims(
    conn: sqlite3.Connection, limit: int, mode: RunMode | None = None
) -> list[ClaimRow]:
    query = (
        "SELECT c.* FROM claims c JOIN extraction_runs r ON r.id = c.extraction_run_id"
        " WHERE c.novelty = ? AND c.retracted_at IS NULL"
    )
    params: list[object] = [Novelty.UNPROBED.value]
    if mode is not None:
        query += " AND r.mode = ?"
        params.append(mode.value)
    query += " ORDER BY c.id LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [_row_to_claim(row) for row in rows]


def claims_needing_probe(conn: sqlite3.Connection, probe_model: str, limit: int) -> list[ClaimRow]:
    rows = conn.execute(
        "SELECT * FROM claims WHERE retracted_at IS NULL"
        " AND (probe_model IS NULL OR probe_model != ?) ORDER BY id LIMIT ?",
        (probe_model, limit),
    ).fetchall()
    return [_row_to_claim(row) for row in rows]


def claims_for_runs_needing_probe(
    conn: sqlite3.Connection,
    run_ids: Sequence[int],
    probe_model: str | None,
    limit: int,
) -> list[ClaimRow]:
    if not run_ids:
        return []
    placeholders = ",".join("?" for _ in run_ids)
    params: list[object] = list(run_ids)
    query = (
        f"SELECT * FROM claims WHERE extraction_run_id IN ({placeholders}) AND retracted_at IS NULL"
    )
    if probe_model is None:
        query += " AND novelty = ?"
        params.append(Novelty.UNPROBED.value)
    else:
        query += " AND (probe_model IS NULL OR probe_model != ?)"
        params.append(probe_model)
    query += " ORDER BY id LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [_row_to_claim(row) for row in rows]


def set_novelty(
    conn: sqlite3.Connection,
    claim_id: int,
    verdict: Novelty,
    probe_model: str,
    probe_answer: str,
    at: datetime,
) -> None:
    conn.execute(
        "UPDATE claims SET novelty = ?, probe_model = ?, probe_answer = ?, probed_at = ?,"
        " probe_error = NULL WHERE id = ?",
        (verdict.value, probe_model, probe_answer, to_db_time(at), claim_id),
    )


def set_probe_error(conn: sqlite3.Connection, claim_id: int, error: str) -> None:
    conn.execute("UPDATE claims SET probe_error = ? WHERE id = ?", (error, claim_id))


def retract_claim(conn: sqlite3.Connection, claim_id: int, reason: str, at: datetime) -> bool:
    row = conn.execute("SELECT retracted_at FROM claims WHERE id = ?", (claim_id,)).fetchone()
    if row is None:
        return False
    if row["retracted_at"] is not None:
        return True
    conn.execute(
        "UPDATE claims SET retracted_at = ?, retraction_reason = ? WHERE id = ?",
        (to_db_time(at), reason, claim_id),
    )
    return True


def retract_claims_with_all_sources_deleted(conn: sqlite3.Connection, at: datetime) -> list[int]:
    rows = conn.execute(
        "SELECT c.id FROM claims c WHERE c.retracted_at IS NULL"
        " AND EXISTS (SELECT 1 FROM claim_sources cs WHERE cs.claim_id = c.id)"
        " AND NOT EXISTS ("
        "  SELECT 1 FROM claim_sources cs2 JOIN messages m ON m.id = cs2.message_id"
        "  WHERE cs2.claim_id = c.id AND m.deleted_at IS NULL)"
    ).fetchall()
    ids = [row["id"] for row in rows]
    if ids:
        with transaction(conn):
            for claim_id in ids:
                conn.execute(
                    "UPDATE claims SET retracted_at = ?, retraction_reason = 'sources_deleted'"
                    " WHERE id = ?",
                    (to_db_time(at), claim_id),
                )
    return ids


def retract_claims_with_all_sources_opted_out(conn: sqlite3.Connection, at: datetime) -> list[int]:
    rows = conn.execute(
        "SELECT c.id FROM claims c WHERE c.retracted_at IS NULL"
        " AND EXISTS (SELECT 1 FROM claim_sources cs WHERE cs.claim_id = c.id)"
        " AND NOT EXISTS ("
        "  SELECT 1 FROM claim_sources cs2 JOIN messages m ON m.id = cs2.message_id"
        "  WHERE cs2.claim_id = c.id AND m.author_id NOT IN (SELECT user_id FROM opt_outs))"
    ).fetchall()
    ids = [row["id"] for row in rows]
    if ids:
        with transaction(conn):
            for claim_id in ids:
                conn.execute(
                    "UPDATE claims SET retracted_at = ?, retraction_reason = 'sources_opted_out'"
                    " WHERE id = ?",
                    (to_db_time(at), claim_id),
                )
    return ids


def _match_expression(text: str) -> str | None:
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text):
        token = raw.rstrip("-./")
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    if not tokens:
        return None
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def related_claims(conn: sqlite3.Connection, text: str, limit: int) -> list[ClaimRow]:
    expression = _match_expression(text)
    if expression is None:
        return []
    rows = conn.execute(
        "SELECT c.* FROM claims_fts"
        " JOIN claims c ON c.id = claims_fts.rowid"
        " JOIN extraction_runs r ON r.id = c.extraction_run_id"
        " WHERE claims_fts MATCH ? AND c.retracted_at IS NULL AND r.mode = ?"
        " ORDER BY bm25(claims_fts, 2.0, 1.0) LIMIT ?",
        (expression, RunMode.LIVE.value, limit),
    ).fetchall()
    return [_row_to_claim(row) for row in rows]
