import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from infovore.db.archived import SCORER_PREFIX, STAGES, archived_clause
from infovore.db.connection import migrate, open_database
from infovore.triage.cascade import SCORERS
from tests.cascade_marks import ARCHIVED_KINDS, AT, RULED_OUT_KINDS, mark


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'c', 'text')"
    )
    for exchange_id in range(1, 9):
        connection.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
            " ended_at, message_count, grouping_rule, content_hash)"
            " VALUES (?, 1, 1, 1, 't', 't', 1, 'quiet_gap', ?)",
            (exchange_id, f"h{exchange_id}"),
        )
    return connection


def archived(conn: sqlite3.Connection) -> set[int]:
    rows = conn.execute(f"SELECT id FROM exchanges WHERE {archived_clause('exchanges.id')}")
    return {row["id"] for row in rows}


def test_scorer_names_track_the_cascade() -> None:
    assert {SCORERS[stage] for stage in STAGES} == {f"{SCORER_PREFIX}{stage}" for stage in STAGES}


@pytest.mark.parametrize("kind", ARCHIVED_KINDS)
def test_relevant_and_residue_exchanges_are_archived(conn: sqlite3.Connection, kind: str) -> None:
    mark(conn, 1, kind)
    assert archived(conn) == {1}


@pytest.mark.parametrize("kind", RULED_OUT_KINDS)
def test_ruled_out_exchanges_are_not_archived(conn: sqlite3.Connection, kind: str) -> None:
    mark(conn, 1, kind)
    assert archived(conn) == set()


def test_an_exchange_the_cascade_never_saw_is_not_archived(conn: sqlite3.Connection) -> None:
    assert archived(conn) == set()


def test_each_exchange_follows_its_own_latest_run(conn: sqlite3.Connection) -> None:
    mark(conn, 1, "residue")
    mark(conn, 2, "denylist")
    mark(conn, 3, "lexicon")
    later = AT + timedelta(days=1)
    mark(conn, 1, "embed_irrelevant", later)
    mark(conn, 2, "lexicon", later)
    assert archived(conn) == {2, 3}


def test_an_older_ruling_does_not_outvote_the_latest_run(conn: sqlite3.Connection) -> None:
    mark(conn, 1, "no_text")
    mark(conn, 1, "residue", AT + timedelta(days=1))
    assert archived(conn) == {1}


def test_other_scorers_do_not_decide_archival(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO annotations (subject_kind, subject_id, scorer, scorer_version,"
        " reproducibility, score, label, recipe_json, created_at)"
        " VALUES ('exchange', 1, 'p_lore', 1, 'derived', 0.99, 'relevant', '{}', 't')"
    )
    assert archived(conn) == set()
