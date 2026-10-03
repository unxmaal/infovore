import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from infovore.chunk.gaps import derive_gap, fold_gap, recorded_gap
from infovore.chunk.grouper import group_pending
from infovore.chunk.recipe import ChunkRecipe, current_recipe, settings_recipe
from infovore.db.chunk_recipes import current_recipe_version, recipe_for_version
from infovore.db.connection import load_migrations, migrate, open_database
from infovore.timing import FixedClock
from tests.chunk.test_rechunk import NOW, count, msg

FIXED = ChunkRecipe(quiet_gap=timedelta(minutes=30), max_messages=50, overlap=3)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "g.db")
    migrate(connection)
    return connection


def test_the_migration_registers_the_adaptive_recipe_as_version_two(
    conn: sqlite3.Connection,
) -> None:
    assert recipe_for_version(conn, 2) == current_recipe()
    assert recipe_for_version(conn, 1) == FIXED
    assert current_recipe().adaptive
    assert not FIXED.adaptive


def test_settings_override_the_floor_and_the_cap() -> None:
    recipe = settings_recipe(timedelta(minutes=10), 20)
    assert (recipe.quiet_gap, recipe.gap_floor, recipe.max_messages) == (
        timedelta(minutes=10),
        timedelta(minutes=10),
        20,
    )
    assert recipe.gap_percentile == current_recipe().gap_percentile


def test_a_fixed_recipe_needs_no_history_and_folds_only_when_asked(
    conn: sqlite3.Connection,
) -> None:
    assert derive_gap(conn, FIXED, 1, False) == timedelta(minutes=30)
    assert fold_gap(FIXED, timedelta(minutes=30)) is None
    assert fold_gap(current_recipe(), timedelta(minutes=30)) == timedelta(hours=2)
    folded_without_size = ChunkRecipe(timedelta(minutes=30), 50, 3, fold_factor=2)
    assert fold_gap(folded_without_size, timedelta(minutes=30)) is None


def test_a_channels_gap_is_recorded_once_and_then_reused(conn: sqlite3.Connection) -> None:
    for id, minute in [(1, 0), (2, 100), (3, 200)]:
        msg(conn, id, minute)
    first = recorded_gap(conn, 2, current_recipe(), 1, False)
    assert first == timedelta(minutes=100)
    msg(conn, 4, 201)
    msg(conn, 5, 202)
    msg(conn, 6, 203)
    msg(conn, 7, 204)
    assert recorded_gap(conn, 2, current_recipe(), 1, False) == first
    assert count(conn, "channel_chunk_gaps") == 1


def test_the_live_chunker_records_its_recipe_and_uses_the_channels_gap(
    conn: sqlite3.Connection,
) -> None:
    for id, minute in [(1, 0), (2, 100), (3, 101), (4, 700)]:
        msg(conn, id, minute)
    report = group_pending(conn, FixedClock(NOW), recipe=current_recipe())
    assert report.exchanges_created == 2
    rows = conn.execute("SELECT chunk_recipe, message_count FROM exchanges ORDER BY id").fetchall()
    assert [(r["chunk_recipe"], r["message_count"]) for r in rows] == [(2, 3), (2, 1)]
    assert current_recipe_version(conn) == 2


def test_the_current_recipe_is_version_one_until_something_is_chunked(
    conn: sqlite3.Connection,
) -> None:
    assert current_recipe_version(conn) == 1


def test_the_migration_keeps_exchanges_pointing_at_their_recipe(tmp_path: Path) -> None:
    connection = open_database(tmp_path / "old.db")
    migrate(connection, load_migrations()[:23])
    connection.execute("INSERT INTO channels (id, guild_id, name, kind) VALUES (1, 9, 'c', 'text')")
    connection.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash, chunk_recipe)"
        " VALUES (1, 1, 1, 1, 'a', 'a', 1, 'quiet_gap', 'h', 1)"
    )
    migrate(connection)
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    row = connection.execute("SELECT chunk_recipe, superseded_by_recipe FROM exchanges").fetchone()
    assert (row["chunk_recipe"], row["superseded_by_recipe"]) == (1, None)
    recipe = connection.execute("SELECT * FROM chunk_recipes WHERE version = 1").fetchone()
    assert recipe["gap_percentile"] == 0
    assert connection.execute("SELECT COUNT(*) FROM chunk_recipes").fetchone()[0] == 2
