import sqlite3
from pathlib import Path

import pytest

from infovore.db.wiki_articles import (
    create_article_run,
    latest_article_run,
    record_section,
    run_sections,
    sections_for,
)
from infovore.db.wiki_tags import create_tag_run
from tests.claims.seed import NOW
from tests.wiki.seed import wiki_db


def tag_run(conn: sqlite3.Connection) -> int:
    return create_tag_run(
        conn, endpoint="e", model_alias="a", model_id="m", prompt_hash="p", now=NOW
    )


def make_run(
    conn: sqlite3.Connection, tag_run_id: int, model_id: str = "m1", prompt_hash: str = "h1"
) -> int:
    return create_article_run(
        conn,
        endpoint="http://x",
        model_alias="alias",
        model_id=model_id,
        prompt_hash=prompt_hash,
        tag_run_id=tag_run_id,
        now=NOW,
    )


def test_record_then_read_back(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    run = make_run(conn, tag_run(conn))
    record_section(conn, run, "O2", "General", [("a", [1, 2]), ("b", [3])], 2)
    record_section(conn, run, "O2", "With Indy", [], 0)
    assert sections_for(conn, [run]) == {
        "O2": {"General": [("a", [1, 2]), ("b", [3])], "With Indy": []}
    }
    dropped = conn.execute("SELECT dropped FROM article_sections WHERE section = 'General'")
    assert dropped.fetchone()[0] == 2
    started = conn.execute("SELECT started_at FROM article_runs WHERE id = ?", (run,))
    assert started.fetchone()[0] == NOW.isoformat()


def test_latest_article_run_filters_on_model_hash_and_tag_run(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    first, second = tag_run(conn), tag_run(conn)
    assert latest_article_run(conn, "m1", "h1", first) is None
    a = make_run(conn, first)
    b = make_run(conn, first, model_id="m2")
    c = make_run(conn, first, prompt_hash="h2")
    d = make_run(conn, second)
    e = make_run(conn, first)
    assert latest_article_run(conn, "m1", "h1", first) == e != a
    assert latest_article_run(conn, "m2", "h1", first) == b
    assert latest_article_run(conn, "m1", "h2", first) == c
    assert latest_article_run(conn, "m1", "h1", second) == d


def test_run_sections_reads_one_run_only(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    first = tag_run(conn)
    a, b = make_run(conn, first), make_run(conn, first)
    record_section(conn, a, "O2", "General", [], 0)
    record_section(conn, b, "O2", "General#2", [], 0)
    record_section(conn, b, "Indy", "General", [], 0)
    assert run_sections(conn, a) == {("O2", "General")}
    assert run_sections(conn, b) == {("O2", "General#2"), ("Indy", "General")}


def test_later_run_wins_per_topic_and_section(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    tr = tag_run(conn)
    r1, r2 = make_run(conn, tr), make_run(conn, tr)
    record_section(conn, r1, "O2", "General", [("old", [1])], 0)
    record_section(conn, r1, "O2", "With Indy", [("kept", [2])], 0)
    record_section(conn, r2, "O2", "General", [("new", [3])], 0)
    assert sections_for(conn, [r1, r2]) == {
        "O2": {"General": [("new", [3])], "With Indy": [("kept", [2])]}
    }
    assert sections_for(conn, [r1])["O2"]["General"] == [("old", [1])]


def test_sections_for_without_runs_is_empty(tmp_path: Path) -> None:
    assert sections_for(wiki_db(tmp_path), []) == {}


def test_updates_and_deletes_are_append_only(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    run = make_run(conn, tag_run(conn))
    record_section(conn, run, "O2", "General", [("a", [1])], 0)
    with pytest.raises(sqlite3.DatabaseError, match="article runs are append-only"):
        conn.execute("UPDATE article_runs SET endpoint = 'z'")
    with pytest.raises(sqlite3.DatabaseError, match="article runs are append-only"):
        conn.execute("DELETE FROM article_runs")
    with pytest.raises(sqlite3.DatabaseError, match="article sections are append-only"):
        conn.execute("UPDATE article_sections SET dropped = 1")
    with pytest.raises(sqlite3.DatabaseError, match="article sections are append-only"):
        conn.execute("DELETE FROM article_sections")
