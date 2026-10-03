import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.chunk.recipe import ChunkRecipe, current_recipe
from infovore.db.chunk_recipes import recipe_for_version, register_recipe
from infovore.db.connection import migrate, open_database

AT = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    return connection


def test_the_recipe_is_derived_from_the_chunker_constants() -> None:
    """A hard-coded recipe rots the moment a constant moves. `grouping_rule`
    already records WHICH rule fired without the parameters it fired with,
    which is the bug (issue #176)."""
    from infovore.chunk.rules import DEFAULT_MAX_MESSAGES, DEFAULT_OVERLAP, DEFAULT_QUIET_GAP

    recipe = current_recipe()

    assert recipe.quiet_gap == DEFAULT_QUIET_GAP
    assert recipe.max_messages == DEFAULT_MAX_MESSAGES
    assert recipe.overlap == DEFAULT_OVERLAP


def test_the_migration_records_the_recipe_the_corpus_was_chunked_with(
    conn: sqlite3.Connection,
) -> None:
    """The 203,634 existing exchanges were all produced with a 30 minute gap,
    a 50 message cap and 3 messages of overlap. Those constants were
    introduced in one commit and never changed, so backfilling them recovers
    what was decided rather than inventing it."""
    recipe = recipe_for_version(conn, 1)

    assert recipe is not None
    assert recipe.quiet_gap == timedelta(minutes=30)
    assert recipe.max_messages == 50
    assert recipe.overlap == 3


def test_registering_the_same_recipe_reuses_its_version(conn: sqlite3.Connection) -> None:
    first = register_recipe(conn, current_recipe(), AT)
    second = register_recipe(conn, current_recipe(), AT)

    assert first == second


def test_a_changed_parameter_registers_a_new_version(conn: sqlite3.Connection) -> None:
    """The whole point: moving a constant must produce a new recipe rather
    than silently relabelling the old one."""
    original = register_recipe(conn, current_recipe(), AT)
    tighter = ChunkRecipe(quiet_gap=timedelta(minutes=20), max_messages=50, overlap=3)

    changed = register_recipe(conn, tighter, AT)

    assert changed != original
    assert recipe_for_version(conn, changed) == tighter


def test_every_parameter_participates_in_the_identity(conn: sqlite3.Connection) -> None:
    base = current_recipe()
    register_recipe(conn, base, AT)
    variants = [
        ChunkRecipe(timedelta(minutes=20), base.max_messages, base.overlap),
        ChunkRecipe(base.quiet_gap, 25, base.overlap),
        ChunkRecipe(base.quiet_gap, base.max_messages, 1),
    ]

    versions = {register_recipe(conn, variant, AT) for variant in variants}

    assert len(versions) == 3


def test_an_unknown_version_has_no_recipe(conn: sqlite3.Connection) -> None:
    assert recipe_for_version(conn, 999) is None


def test_existing_exchanges_carry_the_backfilled_recipe(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (1, 1, 1, 1, ?, ?, 1, 'quiet_gap', 'h1')",
        (AT.isoformat(), AT.isoformat()),
    )

    row = conn.execute("SELECT chunk_recipe FROM exchanges WHERE id = 1").fetchone()

    assert row["chunk_recipe"] is None, "a new row must name its own recipe, not inherit one"


def test_a_new_exchange_records_the_recipe_that_produced_it(conn: sqlite3.Connection) -> None:
    """The whole point of the change: an exchange written today must say which
    parameters grouped it, so a later re-chunk is comparable instead of
    silently replacing an unlabelled one."""
    from infovore.db.exchanges import insert_exchange
    from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule

    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (7, 1, 1, 1, 'a', ?, 'hi', ?, '{}')",
        (AT.isoformat(), AT.isoformat()),
    )
    exchange = ExchangeRow(
        id=None,
        channel_id=1,
        thread_id=None,
        first_message_id=7,
        last_message_id=7,
        started_at=AT,
        ended_at=AT,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash="fresh",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
        chunk_recipe=register_recipe(conn, current_recipe(), AT),
    )

    exchange_id = insert_exchange(conn, exchange, [7])

    row = conn.execute("SELECT chunk_recipe FROM exchanges WHERE id = ?", (exchange_id,)).fetchone()
    assert row["chunk_recipe"] == register_recipe(conn, current_recipe(), AT)
