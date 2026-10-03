import sqlite3
from collections.abc import Sequence
from datetime import timedelta

from infovore.chunk.recipe import ChunkRecipe
from infovore.chunk.rules import channel_gap
from infovore.db.raw import light_channel_messages
from infovore.rows import MessageRow


def derive_gap(
    conn: sqlite3.Connection,
    recipe: ChunkRecipe,
    channel_id: int,
    include_bots: bool,
    messages: Sequence[MessageRow] | None = None,
) -> timedelta:
    if not recipe.adaptive:
        return recipe.quiet_gap
    return channel_gap(
        light_channel_messages(conn, channel_id) if messages is None else messages,
        recipe.gap_percentile,
        recipe.gap_floor,
        recipe.gap_ceiling,
        include_bots,
    )


def recorded_gap(
    conn: sqlite3.Connection,
    version: int,
    recipe: ChunkRecipe,
    channel_id: int,
    include_bots: bool,
) -> timedelta:
    row = conn.execute(
        "SELECT gap_seconds FROM channel_chunk_gaps WHERE recipe = ? AND channel_id = ?",
        (version, channel_id),
    ).fetchone()
    if row is not None:
        return timedelta(seconds=row["gap_seconds"])
    gap = derive_gap(conn, recipe, channel_id, include_bots)
    record_gap(conn, version, channel_id, gap)
    return gap


def record_gap(conn: sqlite3.Connection, version: int, channel_id: int, gap: timedelta) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO channel_chunk_gaps (recipe, channel_id, gap_seconds)"
        " VALUES (?, ?, ?)",
        (version, channel_id, int(gap.total_seconds())),
    )


def fold_gap(recipe: ChunkRecipe, gap: timedelta) -> timedelta | None:
    if not recipe.fold_factor or not recipe.fold_size:
        return None
    return gap * recipe.fold_factor
