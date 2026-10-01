import sqlite3
from datetime import datetime, timedelta

from infovore.chunk.recipe import ChunkRecipe
from infovore.db.codec import to_db_time


def register_recipe(conn: sqlite3.Connection, recipe: ChunkRecipe, at: datetime) -> int:
    """Return the version for this exact set of parameters, inserting one if
    it is new. Identity is every parameter, so moving a constant produces a
    new version rather than silently relabelling the old corpus."""
    seconds = int(recipe.quiet_gap.total_seconds())
    existing = conn.execute(
        "SELECT version FROM chunk_recipes WHERE quiet_gap_seconds = ?"
        " AND max_messages = ? AND overlap = ?",
        (seconds, recipe.max_messages, recipe.overlap),
    ).fetchone()
    if existing is not None:
        return int(existing["version"])
    cursor = conn.execute(
        "INSERT INTO chunk_recipes (quiet_gap_seconds, max_messages, overlap, created_at)"
        " VALUES (?, ?, ?, ?)",
        (seconds, recipe.max_messages, recipe.overlap, to_db_time(at)),
    )
    return int(cursor.lastrowid or 0)


def recipe_for_version(conn: sqlite3.Connection, version: int) -> ChunkRecipe | None:
    row = conn.execute(
        "SELECT quiet_gap_seconds, max_messages, overlap FROM chunk_recipes WHERE version = ?",
        (version,),
    ).fetchone()
    if row is None:
        return None
    return ChunkRecipe(
        quiet_gap=timedelta(seconds=row["quiet_gap_seconds"]),
        max_messages=row["max_messages"],
        overlap=row["overlap"],
    )
