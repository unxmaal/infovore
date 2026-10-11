from pathlib import Path

from infovore.db.novelty import (
    Novelty,
    Reference,
    descriptions,
    novelty_for,
    references,
    set_descriptions,
    store_novelty,
    store_reference,
)
from tests.claims.seed import NOW
from tests.wiki.seed import add_claim, add_exchange, wiki_db


def test_references_replace_per_topic(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    store_reference(conn, Reference("Octane", "SGI Octane", "u", "text", 0.4), NOW)
    store_reference(conn, Reference("PSU", "", "", "", 0.0), NOW)
    store_reference(conn, Reference("Octane", "Octane", "u2", "newer", 0.6, "Fuel"), NOW)
    refs = references(conn)
    assert refs["Octane"] == Reference("Octane", "Octane", "u2", "newer", 0.6, "Fuel")
    assert refs["Octane"].found and not refs["PSU"].found
    set_descriptions(conn, {"Octane": "Computer series", "Nope": "x"})
    assert references(conn)["Octane"].description == "Computer series"
    assert descriptions(conn) == {"Octane": "Computer series"}


def test_novelty_rows_replace_per_claim_and_read_in_chunks(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    exchange = add_exchange(conn, 1, "2026-02-01")
    ids = [add_claim(conn, exchange, "user-aaaa", f"claim {i}") for i in range(600)]
    store_novelty(conn, [Novelty(i, "T", "Title", 0.5, "near", 0.25) for i in ids], NOW)
    store_novelty(conn, [Novelty(ids[0], None, None, None, None, 1.0)], NOW)
    rows = novelty_for(conn, ids)
    assert len(rows) == 600
    assert rows[ids[0]] == Novelty(ids[0], None, None, None, None, 1.0)
    assert rows[ids[1]] == Novelty(ids[1], "T", "Title", 0.5, "near", 0.25)
    assert novelty_for(conn, []) == {}
