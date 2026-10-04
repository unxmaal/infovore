import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.reviewed_words import record_decisions
from infovore.words.review import (
    ReviewQueue,
    UnknownWordError,
    build_pile,
    candidates,
    common_words,
    gain,
    undecided_ids,
)
from tests.cascade_marks import mark
from tests.triage.test_human import db, seed

AT = datetime(2026, 1, 1, tzinfo=UTC)


def world(tmp_path: Path) -> sqlite3.Connection:
    conn = db(tmp_path)
    residue = seed(conn, 1, "the ubr ubr hates my g5 see https://ebay.com/x")
    other = seed(conn, 2, "ubr and the 7200 modem")
    decided = seed(conn, 3, "relevant zzzz")
    seed(conn, 4, "unscored qqqq")
    mark(conn, residue, "residue")
    mark(conn, other, "residue")
    mark(conn, decided, "lexicon")
    return conn


def test_only_residue_conversations_are_undecided(tmp_path: Path) -> None:
    conn = world(tmp_path)

    assert len(undecided_ids(conn)) == 2


def test_a_later_cascade_run_replaces_the_earlier_one(tmp_path: Path) -> None:
    conn = world(tmp_path)
    first = undecided_ids(conn)[0]
    mark(conn, first, "lexicon", datetime(2026, 2, 1, tzinfo=UTC))

    assert first not in undecided_ids(conn)


def test_the_pile_counts_every_occurrence(tmp_path: Path) -> None:
    conn = world(tmp_path)
    pile = build_pile(conn)

    assert pile["ubr"] == 3
    assert pile["g5"] == 1
    assert pile["7200"] == 1
    assert pile["ebay"] == 1
    assert "zzzz" not in pile


def test_common_english_is_subtracted_with_a_configurable_size() -> None:
    assert "the" in common_words(100)
    assert len(common_words(10)) <= 10
    assert len(common_words(10)) < len(common_words(1000))


def test_candidates_rank_by_count_minus_common_known_and_reviewed(tmp_path: Path) -> None:
    conn = world(tmp_path)
    found = candidates(conn, 10000)
    names = [w for w, _ in found]

    assert found[0] == ("ubr", 3)
    assert "the" not in names
    assert "g5" in names
    assert "7200" not in names
    assert "ebay" not in names

    record_decisions(conn, {"ubr": False, "g5": True}, AT)
    again = [w for w, _ in candidates(conn, 10000)]
    assert "ubr" not in again
    assert "g5" not in again


def test_words_already_in_the_lexicon_are_dropped(tmp_path: Path) -> None:
    conn = db(tmp_path)
    mark(conn, seed(conn, 1, "my disk zorpish zorpish"), "residue")

    assert [w for w, _ in candidates(conn, 10000)] == ["zorpish"]


def test_the_queue_serves_highest_frequency_first_and_never_repeats() -> None:
    queue = ReviewQueue([("a", 5), ("b", 4), ("c", 3), ("d", 2)])

    assert queue.page(2) == [("a", 5), ("b", 4)]
    assert queue.page(2) == [("a", 5), ("b", 4)]
    queue.mark_decided(["a", "b"])
    assert queue.page(5) == [("c", 3), ("d", 2)]
    assert queue.remaining == 2
    queue.mark_decided(["d"])
    assert queue.page(5) == [("c", 3)]
    queue.mark_decided(["c"])
    assert queue.page(5) == []


def test_the_queue_refuses_unknown_words() -> None:
    queue = ReviewQueue([("a", 5)])

    with pytest.raises(UnknownWordError, match="nope"):
        queue.mark_decided(["nope"])


def test_gain_counts_undecided_conversations_with_an_approved_word(tmp_path: Path) -> None:
    conn = world(tmp_path)

    assert gain(conn, frozenset()) == (2, 0)
    assert gain(conn, frozenset({"g5"})) == (2, 1)
    assert gain(conn, frozenset({"ubr"})) == (2, 2)
    assert gain(conn, frozenset({"modem"})) == (2, 1)
    assert gain(conn, frozenset({"modems"})) == (2, 0)


def test_the_undecided_query_never_searches_annotations_by_scorer(tmp_path: Path) -> None:
    from infovore.words.review import _LATEST_CASCADE

    plan = " ".join(
        str(tuple(row)) for row in db(tmp_path).execute("EXPLAIN QUERY PLAN " + _LATEST_CASCADE)
    )

    assert "annotations_scorer" not in plan


def test_ids_digits_contractions_and_gazetteer_words_are_not_candidates(tmp_path: Path) -> None:
    conn = db(tmp_path)
    text = "<@142833170897698816> 100 zq7200 didn\N{RIGHT SINGLE QUOTATION MARK}t zxqv"
    text += " octane o2 indy iris"
    mark(conn, seed(conn, 1, text), "residue")

    names = [w for w, _ in candidates(conn, 10000)]

    assert "zq7200" in names
    assert "zxqv" in names
    assert "142833170897698816" not in names
    assert "100" not in names
    assert "didn" not in names
    assert "octane" not in names
    assert "o2" not in names
    assert "indy" not in names
