import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.reviewed_words import (
    approved_words,
    decided_words,
    decision_counts,
    record_decisions,
)
from tests.triage.test_human import db

AT = datetime(2026, 1, 1, tzinfo=UTC)


def test_both_outcomes_are_recorded_and_count(tmp_path: Path) -> None:
    conn = db(tmp_path)
    record_decisions(conn, {"ubr": True, "lunch": False}, AT)

    assert decided_words(conn) == {"ubr", "lunch"}
    assert approved_words(conn) == {"ubr"}
    assert decision_counts(conn) == (1, 1)


def test_the_latest_decision_for_a_word_wins(tmp_path: Path) -> None:
    conn = db(tmp_path)
    record_decisions(conn, {"g5": False}, AT)
    record_decisions(conn, {"g5": True}, AT)

    assert approved_words(conn) == {"g5"}
    assert decision_counts(conn) == (1, 0)


def test_decisions_are_append_only(tmp_path: Path) -> None:
    conn = db(tmp_path)
    record_decisions(conn, {"ubr": True}, AT)

    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE reviewed_words SET tech = 0")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM reviewed_words")


def test_nothing_recorded_is_empty(tmp_path: Path) -> None:
    conn = db(tmp_path)
    record_decisions(conn, {}, AT)

    assert decided_words(conn) == frozenset()
    assert decision_counts(conn) == (0, 0)
