from pathlib import Path

import pytest

from infovore.reputation.evidence import build_evidence
from infovore.reputation.exchange import exchange_score, exchange_shares, score_exchanges
from infovore.reputation.people import People
from infovore.reputation.score import build_reputation
from infovore.rows import Label
from tests.reputation.world import SALT, exchange, react, world

PLAIN = People({}, frozenset(), {})


def test_shares_are_authorship_fractions_merged_over_a_persons_accounts(tmp_path: Path) -> None:
    conn = world(tmp_path)
    first, (_, bot, _, deleted) = exchange(
        conn, [(1, "a", "x"), (9, "b", "y"), (2, "c", "z"), (3, "d", "w")], 1
    )
    conn.execute("UPDATE messages SET author_is_bot = 1 WHERE id = ?", (bot,))
    conn.execute("UPDATE messages SET deleted_at = 't' WHERE id = ?", (deleted,))
    empty, (lone,) = exchange(conn, [(5, "e", "v")], 2)
    conn.execute("UPDATE messages SET deleted_at = 't' WHERE id = ?", (lone,))
    people = People({1: "p", 2: "p"}, frozenset(), {"p": (1, 2)})

    shares = exchange_shares(conn, people, [first, empty])

    assert shares == {first: {"p": 1.0}, empty: {}}


def test_duplicate_ids_collapse(tmp_path: Path) -> None:
    conn = world(tmp_path)
    ids = [exchange(conn, [(1, "a", "x")], base)[0] for base in range(1, 4)]

    assert set(exchange_shares(conn, PLAIN, ids * 400)) == set(ids)


def test_an_exchange_scores_the_share_weighted_reputation_of_its_authors(tmp_path: Path) -> None:
    conn = world(tmp_path)
    exchange(conn, [(1, "a", "x"), (1, "a", "y"), (2, "b", "z")], 1)
    _, (m2,) = exchange(conn, [(1, "a", "w")], 2)
    react(conn, m2, 4)
    rep = build_reputation(
        build_evidence(conn, PLAIN, SALT, {}, frozenset(), frozenset(), []), PLAIN
    )

    score = exchange_score(rep, {"1": 2 / 3, "2": 1 / 3})

    assert score == pytest.approx(2 / 3 * rep.of("1") + 1 / 3 * rep.of("2"))
    assert exchange_score(rep, {}) == 0.0


def test_leave_one_out_removes_the_exchanges_own_contribution(tmp_path: Path) -> None:
    conn = world(tmp_path)
    eid, _ = exchange(conn, [(1, "a", "x")], 1)
    others = [exchange(conn, [(2, "b", "y")], base)[0] for base in range(2, 6)]
    labels = {eid: Label.LORE, **dict.fromkeys(others, Label.NOISE)}
    evidence = build_evidence(conn, PLAIN, SALT, labels, frozenset(), frozenset(), [eid])
    rep = build_reputation(evidence, PLAIN)

    scored = score_exchanges(conn, rep, [eid, others[0]])

    assert scored[eid] == pytest.approx(rep.of("1", evidence.contribution(eid)))
    assert scored[eid] < rep.of("1")
    assert scored[others[0]] == pytest.approx(rep.of("2"))


def test_all_of_a_persons_accounts_are_removed_together(tmp_path: Path) -> None:
    conn = world(tmp_path)
    eid, _ = exchange(conn, [(1, "a", "x"), (2, "b", "y")], 1)
    other, _ = exchange(conn, [(3, "c", "z")], 2)
    people = People({1: "p", 2: "p"}, frozenset(), {"p": (1, 2)})
    labels = {eid: Label.LORE, other: Label.NOISE}
    evidence = build_evidence(conn, people, SALT, labels, frozenset(), frozenset(), [eid])
    rep = build_reputation(evidence, people)

    scored = score_exchanges(conn, rep, [eid])

    assert scored[eid] == pytest.approx(rep.of("p", evidence.contribution(eid)))
    assert evidence.contribution(eid)["p"]["exchanges"] == (1.0, 1.0)
