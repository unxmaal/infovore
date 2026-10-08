import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.speaker_drops import DECISIONS, decisions, dropped_authors, record_decision
from tests.claims.seed import db

AT = datetime(2026, 2, 1, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return db(tmp_path)


def test_the_latest_decision_per_author_wins(conn: sqlite3.Connection) -> None:
    record_decision(conn, 7, "drop", AT)
    record_decision(conn, 8, "drop", AT)
    record_decision(conn, 7, "keep", AT)
    record_decision(conn, 9, "keep", AT)

    assert DECISIONS == ("drop", "keep")
    assert decisions(conn) == {7: "keep", 8: "drop", 9: "keep"}
    assert dropped_authors(conn) == frozenset({8})
    assert conn.execute("SELECT COUNT(*) FROM speaker_drops").fetchone()[0] == 4


def test_nothing_is_dropped_by_default(conn: sqlite3.Connection) -> None:
    assert dropped_authors(conn) == frozenset()
    assert decisions(conn) == {}


def test_the_log_is_append_only_and_checks_the_decision(conn: sqlite3.Connection) -> None:
    record_decision(conn, 7, "drop", AT)

    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE speaker_drops SET decision = 'keep'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM speaker_drops")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO speaker_drops (author_id, decision, decided_at) VALUES (1, 'maybe', 'x')"
        )
