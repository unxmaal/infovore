import sqlite3
from datetime import datetime, timedelta

from infovore.chunk.recipe import ChunkRecipe
from infovore.db.codec import to_db_time

_KEY = (
    "quiet_gap_seconds = ? AND max_messages = ? AND overlap = ? AND gap_percentile = ?"
    " AND gap_floor_seconds = ? AND gap_ceiling_seconds = ? AND fold_factor = ?"
    " AND fold_size = ?"
)


def _params(recipe: ChunkRecipe) -> tuple[int, int, int, float, int, int, float, int]:
    return (
        int(recipe.quiet_gap.total_seconds()),
        recipe.max_messages,
        recipe.overlap,
        recipe.gap_percentile,
        int(recipe.gap_floor.total_seconds()),
        int(recipe.gap_ceiling.total_seconds()),
        recipe.fold_factor,
        recipe.fold_size,
    )


def register_recipe(conn: sqlite3.Connection, recipe: ChunkRecipe, at: datetime) -> int:
    """Return the version for this exact set of parameters, inserting one if
    it is new. Identity is every parameter, so moving a constant produces a
    new version rather than silently relabelling the old corpus."""
    params = _params(recipe)
    existing = conn.execute(f"SELECT version FROM chunk_recipes WHERE {_KEY}", params).fetchone()
    if existing is not None:
        return int(existing["version"])
    cursor = conn.execute(
        "INSERT INTO chunk_recipes (quiet_gap_seconds, max_messages, overlap, gap_percentile,"
        " gap_floor_seconds, gap_ceiling_seconds, fold_factor, fold_size, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (*params, to_db_time(at)),
    )
    return int(cursor.lastrowid or 0)


def recipe_for_version(conn: sqlite3.Connection, version: int) -> ChunkRecipe | None:
    row = conn.execute("SELECT * FROM chunk_recipes WHERE version = ?", (version,)).fetchone()
    if row is None:
        return None
    return ChunkRecipe(
        quiet_gap=timedelta(seconds=row["quiet_gap_seconds"]),
        max_messages=row["max_messages"],
        overlap=row["overlap"],
        gap_percentile=row["gap_percentile"],
        gap_floor=timedelta(seconds=row["gap_floor_seconds"]),
        gap_ceiling=timedelta(seconds=row["gap_ceiling_seconds"]),
        fold_factor=row["fold_factor"],
        fold_size=row["fold_size"],
    )


def current_recipe_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT MAX(chunk_recipe) AS version FROM exchanges WHERE superseded_by_recipe IS NULL"
    ).fetchone()
    return int(row["version"] or 1)
