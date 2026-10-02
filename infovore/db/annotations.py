import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from infovore.db.codec import to_db_time

SubjectKind = Literal["exchange", "message", "claim"]
Reproducibility = Literal["derived", "recorded"]

# The seam between this generic table and the single-valued columns it replaces
# as the write path. The columns stay so no existing reader breaks.
MATERIALISED: dict[str, tuple[str, str, str]] = {
    "p_lore": ("exchanges", "p_lore", "p_lore_model"),
    "p_trash": ("messages", "p_trash", "p_trash_model"),
}


class AnnotationConflictError(ValueError):
    pass


@dataclass(frozen=True)
class Annotation:
    subject_kind: SubjectKind
    subject_id: int
    scorer: str
    scorer_version: int
    reproducibility: Reproducibility
    score: float | None = None
    label: str | None = None
    recipe: dict[str, object] | None = None
    source_ref: str | None = None


def record_annotation(conn: sqlite3.Connection, annotation: Annotation, at: datetime) -> int:
    """Append one annotation. A derived score is rejected without its recipe,
    and re-scoring the same subject under the same version is a conflict
    rather than an overwrite: that overwrite is what destroyed fifteen message
    models' output (issue #176)."""
    recipe = json.dumps(annotation.recipe, sort_keys=True) if annotation.recipe else None
    try:
        cursor = conn.execute(
            "INSERT INTO annotations (subject_kind, subject_id, scorer, scorer_version,"
            " reproducibility, score, label, recipe_json, source_ref, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                annotation.subject_kind,
                annotation.subject_id,
                annotation.scorer,
                annotation.scorer_version,
                annotation.reproducibility,
                annotation.score,
                annotation.label,
                recipe,
                annotation.source_ref,
                to_db_time(at),
            ),
        )
    except sqlite3.IntegrityError as error:
        raise AnnotationConflictError(str(error)) from error
    if annotation.scorer in MATERIALISED and annotation.score is not None:
        _materialise(conn, annotation)
    return int(cursor.lastrowid or 0)


def _materialise(conn: sqlite3.Connection, annotation: Annotation) -> None:
    """Keep the legacy column in step, but only for the activated version. A
    stale version writing the column would make the "current" view describe
    whichever scorer ran last rather than the one in force."""
    if active_version(conn, annotation.scorer) != annotation.scorer_version:
        return
    table, score_column, version_column = MATERIALISED[annotation.scorer]
    conn.execute(
        f"UPDATE {table} SET {score_column} = ?, {version_column} = ? WHERE id = ?",
        (annotation.score, annotation.scorer_version, annotation.subject_id),
    )


def activate_scorer(
    conn: sqlite3.Connection, scorer: str, version: int, at: datetime, note: str | None = None
) -> None:
    conn.execute(
        "INSERT INTO scorer_activations (scorer, scorer_version, activated_at, note)"
        " VALUES (?, ?, ?, ?)",
        (scorer, version, to_db_time(at), note),
    )


def active_version(conn: sqlite3.Connection, scorer: str) -> int | None:
    row = conn.execute(
        "SELECT scorer_version FROM scorer_activations WHERE scorer = ?"
        " ORDER BY activated_at DESC, id DESC LIMIT 1",
        (scorer,),
    ).fetchone()
    return int(row["scorer_version"]) if row is not None else None


def version_active_at(conn: sqlite3.Connection, scorer: str, at: datetime) -> int | None:
    """Which version was live at a past moment. Unanswerable before this table
    existed except by reading git history."""
    row = conn.execute(
        "SELECT scorer_version FROM scorer_activations WHERE scorer = ? AND activated_at <= ?"
        " ORDER BY activated_at DESC, id DESC LIMIT 1",
        (scorer, to_db_time(at)),
    ).fetchone()
    return int(row["scorer_version"]) if row is not None else None


def annotation_history(
    conn: sqlite3.Connection, subject_kind: SubjectKind, subject_id: int, scorer: str
) -> list[sqlite3.Row]:
    """Every answer this scorer has given about this subject, oldest first.
    The question the old schema could not answer at all."""
    return list(
        conn.execute(
            "SELECT * FROM annotations WHERE subject_kind = ? AND subject_id = ? AND scorer = ?"
            " ORDER BY scorer_version, id",
            (subject_kind, subject_id, scorer),
        )
    )


def drop_derived(conn: sqlite3.Connection, scorer: str, version: int) -> int:
    """Discard a derived scorer version's cache. Permitted precisely because a
    derived value is recomputable from its recipe; the delete trigger refuses
    the same call for recorded annotations."""
    cursor = conn.execute(
        "DELETE FROM annotations WHERE scorer = ? AND scorer_version = ?"
        " AND reproducibility = 'derived'",
        (scorer, version),
    )
    return cursor.rowcount
