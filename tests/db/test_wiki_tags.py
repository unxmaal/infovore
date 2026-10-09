import sqlite3
from pathlib import Path

import pytest

from infovore.db.wiki_tags import create_tag_run, record_tags, tagged_claim_ids, tags_for
from tests.claims.seed import NOW
from tests.wiki.seed import add_claim, add_exchange, wiki_db


def make_run(conn: sqlite3.Connection, model_id: str = "m7", prompt_hash: str = "h1") -> int:
    return create_tag_run(
        conn,
        endpoint="http://x",
        model_alias="alias",
        model_id=model_id,
        prompt_hash=prompt_hash,
        now=NOW,
    )


def one_claim(tmp_path: Path) -> tuple[sqlite3.Connection, int]:
    conn = wiki_db(tmp_path)
    exchange = add_exchange(conn, 1, "2026-02-01")
    return conn, add_claim(conn, exchange, "user-aaaa", "The O2 is quiet.")


def test_record_then_read_back_keeps_order_and_empty(tmp_path: Path) -> None:
    conn, c1 = one_claim(tmp_path)
    c2 = add_claim(conn, 1, "user-aaaa", "Another.")
    run_id = make_run(conn)
    record_tags(conn, run_id, {c1: ["zeta", "alpha", "mid"], c2: []})
    assert tags_for(conn, [run_id]) == {c1: ["zeta", "alpha", "mid"], c2: []}
    started = conn.execute("SELECT started_at FROM tag_runs WHERE id = ?", (run_id,)).fetchone()
    assert started[0] == NOW.isoformat()


def test_tagged_claim_ids_matches_only_same_model_and_hash(tmp_path: Path) -> None:
    conn, claim = one_claim(tmp_path)
    record_tags(conn, make_run(conn), {claim: ["a"]})
    assert tagged_claim_ids(conn, "m7", "h1") == {claim}
    assert tagged_claim_ids(conn, "m7", "other") == set()
    assert tagged_claim_ids(conn, "m99", "h1") == set()


def test_updates_and_deletes_are_append_only(tmp_path: Path) -> None:
    conn, claim = one_claim(tmp_path)
    run_id = make_run(conn)
    record_tags(conn, run_id, {claim: ["a"]})
    with pytest.raises(sqlite3.DatabaseError, match="tag runs are append-only"):
        conn.execute("UPDATE tag_runs SET model_alias = 'x'")
    with pytest.raises(sqlite3.DatabaseError, match="tag runs are append-only"):
        conn.execute("DELETE FROM tag_runs")
    with pytest.raises(sqlite3.DatabaseError, match="claim tags are append-only"):
        conn.execute("UPDATE claim_tags SET tags_json = '[]'")
    with pytest.raises(sqlite3.DatabaseError, match="claim tags are append-only"):
        conn.execute("DELETE FROM claim_tags")


def test_highest_run_wins_and_other_runs_are_ignored(tmp_path: Path) -> None:
    conn, claim = one_claim(tmp_path)
    first, second, third = make_run(conn), make_run(conn), make_run(conn)
    record_tags(conn, first, {claim: ["first"]})
    record_tags(conn, second, {claim: ["second"]})
    record_tags(conn, third, {claim: ["third"]})
    assert tags_for(conn, [first, second]) == {claim: ["second"]}
    assert tags_for(conn, [first]) == {claim: ["first"]}
