import sqlite3
from pathlib import Path

from infovore.reputation.evaluate import (
    DENYLIST_WRONG,
    LEXICON_UNDECIDED,
    LEXICON_WRONG,
    MANY_MESSAGES,
    SUBSETS,
    evaluate_population,
    scored_pairs,
    stage_labels,
    strata,
)
from infovore.rows import Label
from tests.reputation.world import exchange, label_exchange, mark_stage, world

LORE, NOISE = Label.LORE, Label.NOISE


def staged(tmp_path: Path) -> tuple[sqlite3.Connection, dict[str, int], dict[int, Label]]:
    conn = world(tmp_path)
    ids: dict[str, int] = {}
    for name, lines in {
        "right": [(1, "a", "x")],
        "wrong": [(1, "a", "x")],
        "undecided": [(1, "a", "x")],
        "denied": [(1, "a", "x")],
        "long": [(1, "a", "x"), (2, "b", "y"), (3, "c", "z")],
    }.items():
        ids[name], _ = exchange(conn, lines, len(ids) + 1)
    mark_stage(conn, ids["right"], "lexicon", "relevant")
    mark_stage(conn, ids["wrong"], "lexicon", "relevant")
    mark_stage(conn, ids["undecided"], "lexicon", None)
    mark_stage(conn, ids["denied"], "denylist", "irrelevant")
    mark_stage(conn, ids["long"], "lexicon", "irrelevant")
    labels = {
        ids["right"]: LORE,
        ids["wrong"]: NOISE,
        ids["undecided"]: LORE,
        ids["denied"]: LORE,
        ids["long"]: NOISE,
    }
    return conn, ids, labels


def test_the_latest_stage_row_wins_and_a_score_only_row_has_no_label(tmp_path: Path) -> None:
    conn, ids, _ = staged(tmp_path)
    mark_stage(conn, ids["undecided"], "lexicon", "relevant", version=2)
    mark_stage(conn, ids["right"], "lexicon", None, version=2)

    found = stage_labels(conn, "lexicon", sorted(ids.values()))

    assert found[ids["undecided"]] == "relevant"
    assert found[ids["right"]] is None
    assert ids["denied"] not in found
    assert stage_labels(conn, "lexicon", []) == {}


def test_strata_split_by_what_the_cascade_decided(tmp_path: Path) -> None:
    conn, ids, labels = staged(tmp_path)

    found = strata(conn, labels)

    assert set(found) == set(SUBSETS)
    assert found[LEXICON_UNDECIDED] == sorted([ids["undecided"], ids["denied"]])
    assert found[LEXICON_WRONG] == sorted([ids["wrong"], ids["undecided"], ids["denied"]])
    assert found[DENYLIST_WRONG] == [ids["denied"]]
    assert found[MANY_MESSAGES] == [ids["long"]]


def test_a_denylist_drop_of_an_irrelevant_exchange_is_not_wrong(tmp_path: Path) -> None:
    conn, ids, labels = staged(tmp_path)
    labels[ids["denied"]] = NOISE

    assert strata(conn, labels)[DENYLIST_WRONG] == []


def test_scored_pairs_keep_only_scored_labelled_exchanges() -> None:
    pairs = scored_pairs({1: 0.5, 2: 0.7}, {1: LORE, 3: NOISE}, [1, 2, 3])

    assert pairs == [(0.5, LORE)]


def test_a_population_report_has_an_overall_interval_and_one_per_subset(tmp_path: Path) -> None:
    conn, _, labels = staged(tmp_path)
    for eid, label in labels.items():
        label_exchange(conn, eid, label is LORE)
    scores = {eid: (1.0 if label is LORE else 0.0) for eid, label in labels.items()}

    report = evaluate_population(conn, scores, labels, seed=1)

    assert report.overall.auc == 1.0
    assert (report.overall.relevant, report.overall.irrelevant) == (3, 2)
    assert set(report.subsets) == set(SUBSETS)
    assert report.subsets[LEXICON_UNDECIDED].auc is None
    assert report.subsets[LEXICON_WRONG].auc == 1.0
