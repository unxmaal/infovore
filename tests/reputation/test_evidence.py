import sqlite3
from pathlib import Path

import pytest

from infovore.db.claims_v2 import record_review
from infovore.reputation.evidence import SIGNALS, Evidence, add_cell, build_evidence
from infovore.reputation.people import People
from infovore.rows import Label
from tests.claims.seed import NOW
from tests.reputation.world import (
    SALT,
    exchange,
    label_exchange,
    label_message,
    react,
    reply,
    world,
)
from tests.wiki.seed import add_claim, add_run

PLAIN = People({}, frozenset(), {})
LONG_QUESTION = "how do I flash the prom on this?"


def build(
    conn: sqlite3.Connection,
    people: People = PLAIN,
    labels: dict[int, Label] | None = None,
    exclude: frozenset[str] = frozenset(),
    held: frozenset[int] = frozenset(),
    keep: tuple[int, ...] = (),
) -> Evidence:
    return build_evidence(conn, people, SALT, labels or {}, exclude, held, keep)


def test_replies_count_distinct_other_persons_and_skip_self_alts_bots_and_deleted(
    tmp_path: Path,
) -> None:
    conn = world(tmp_path)
    _, (first, second) = exchange(conn, [(1, "a", "hello"), (1, "a", "again")], 1)
    reply(conn, first, 2)
    reply(conn, first, 3)
    reply(conn, first, 3)
    reply(conn, first, 1)
    reply(conn, first, 4, bot=True)
    reply(conn, first, 5, deleted=True)
    reply(conn, second, 11)

    ev = build(conn, People({11: "1"}, frozenset(), {"1": (1, 11)}))

    assert ev.totals["1"]["replies"] == (2.0, 2.0)


def test_reactions_are_summed_over_emoji(tmp_path: Path) -> None:
    conn = world(tmp_path)
    _, (message,) = exchange(conn, [(1, "a", "hello")], 1)
    react(conn, message, 3)
    conn.execute("INSERT INTO reactions (message_id, emoji, count) VALUES (?, 'y', 2)", (message,))

    assert build(conn).totals["1"]["reactions"] == (5.0, 1.0)


def test_mentions_credit_others_once_and_never_the_author(tmp_path: Path) -> None:
    conn = world(tmp_path)
    exchange(conn, [(1, "a", "hi <@2> and <@!2> and <@1>")], 1)

    ev = build(conn)

    assert ev.totals["1"]["mentions"] == (0.0, 1.0)
    assert ev.totals["2"]["mentions"] == (1.0, 0.0)


def test_a_question_is_answered_by_a_native_reply_from_someone_else(tmp_path: Path) -> None:
    conn = world(tmp_path)
    _, (answered,) = exchange(conn, [(1, "a", LONG_QUESTION)], 1)
    exchange(conn, [(1, "a", LONG_QUESTION + " 2")], 2)
    exchange(conn, [(1, "a", "why?")], 3)
    exchange(conn, [(1, "a", "a statement of some length")], 4)
    _, (selfie,) = exchange(conn, [(1, "a", LONG_QUESTION + " 3")], 5)
    reply(conn, answered, 2)
    reply(conn, selfie, 1)

    assert build(conn).totals["1"]["answered"] == (1.0, 3.0)


def test_only_human_context_and_value_message_labels_count(tmp_path: Path) -> None:
    conn = world(tmp_path)
    for base, keep, regime, source in (
        (1, True, "value", "human"),
        (2, False, "context", "human"),
        (3, True, "isolated", "human"),
        (4, True, "context", "citation"),
    ):
        _, (message,) = exchange(conn, [(1, "a", f"text {base}")], base)
        label_message(conn, message, keep, regime, source)
    exchange(conn, [(1, "a", "unlabelled")], 5)

    assert build(conn).totals["1"]["messages"] == (1.0, 2.0)


def test_exchange_labels_are_weighted_by_authorship_share(tmp_path: Path) -> None:
    conn = world(tmp_path)
    eid, _ = exchange(conn, [(1, "a", "x"), (1, "a", "y"), (2, "b", "z")], 1)
    other, _ = exchange(conn, [(2, "b", "w")], 2)

    ev = build(conn, labels={eid: Label.LORE, other: Label.NOISE})

    assert ev.totals["1"]["exchanges"] == pytest.approx((2 / 3, 2 / 3))
    assert ev.totals["2"]["exchanges"] == pytest.approx((1 / 3, 1 / 3 + 1))


def test_held_out_and_excluded_exchanges_contribute_nothing(tmp_path: Path) -> None:
    conn = world(tmp_path)
    held, (message,) = exchange(conn, [(1, "a", "held")], 1, held_out=True)
    react(conn, message, 9)
    label_exchange(conn, held, True)
    skipped, _ = exchange(conn, [(2, "b", "skipped")], 2)
    exchange(conn, [(3, "c", "kept")], 3)

    ev = build(conn, held=frozenset({held}), exclude=frozenset({"c"}))

    assert ev.totals == {}
    assert ev.contributions == {}
    again = build(conn, held=frozenset({held, skipped}), keep=(held, skipped))
    assert set(again.totals) == {"3"}
    assert again.contributions == {}


def test_bot_and_deleted_authors_are_not_counted(tmp_path: Path) -> None:
    conn = world(tmp_path)
    _, (bot, gone, _) = exchange(conn, [(1, "a", "bot"), (2, "b", "gone"), (3, "c", "kept")], 1)
    conn.execute("UPDATE messages SET author_is_bot = 1 WHERE id = ?", (bot,))
    conn.execute("UPDATE messages SET deleted_at = 't' WHERE id = ?", (gone,))

    assert set(build(conn).totals) == {"3"}


def test_claims_count_good_verdicts_by_resolved_speaker(tmp_path: Path) -> None:
    from infovore.claims.redact import pseudonym

    conn = world(tmp_path)
    eid, _ = exchange(conn, [(1, "a", "hello"), (2, "b", "hi")], 1)
    add_run(conn)
    speaker = pseudonym(1, SALT)
    add_claim(conn, eid, speaker, "one", review="good")
    add_claim(conn, eid, speaker, "two", review="wrong")
    cited = add_claim(conn, eid, speaker, "three", review=None)
    record_review(conn, cited, "good", NOW, "cited-only")
    add_claim(conn, eid, "user-unknown", "four", review="good")
    add_claim(conn, eid, speaker, "five", review=None)

    ev = build(conn)

    assert ev.totals["1"]["claims"] == (1.0, 2.0)
    assert "claims" not in ev.totals["2"]


def test_claims_on_skipped_exchanges_are_ignored(tmp_path: Path) -> None:
    from infovore.claims.redact import pseudonym

    conn = world(tmp_path)
    eid, _ = exchange(conn, [(1, "a", "hello")], 1, held_out=True)
    add_run(conn)
    add_claim(conn, eid, pseudonym(1, SALT), "one", review="good")

    assert build(conn, held=frozenset({eid})).totals == {}


def test_accounts_of_one_person_share_a_ledger(tmp_path: Path) -> None:
    conn = world(tmp_path)
    exchange(conn, [(1, "a", "x"), (2, "b", "y")], 1)

    ev = build(conn, People({1: "p", 2: "p"}, frozenset(), {"p": (1, 2)}))

    assert list(ev.totals) == ["p"]
    assert ev.totals["p"]["mentions"] == (0.0, 2.0)
    assert ev.volume("p") == 2.0
    assert ev.volume("nobody") == 0.0


def test_contributions_are_kept_for_requested_exchanges_and_sum_to_the_totals(
    tmp_path: Path,
) -> None:
    conn = world(tmp_path)
    first, (message,) = exchange(conn, [(1, "a", "x")], 1)
    second, _ = exchange(conn, [(1, "a", "y"), (2, "b", "z")], 2)
    reply(conn, message, 2)

    ev = build(conn, labels={second: Label.LORE}, keep=(first,))

    assert ev.contribution(first)["1"]["replies"] == (1.0, 1.0)
    assert ev.contribution(second) == {}
    assert ev.totals["1"]["replies"] == (1.0, 2.0)
    assert ev.totals["1"]["exchanges"] == (0.5, 0.5)


def test_priors_are_pooled_rates_and_zero_without_data(tmp_path: Path) -> None:
    conn = world(tmp_path)
    _, (message, _other) = exchange(conn, [(1, "a", "x"), (2, "b", "y")], 1)
    reply(conn, message, 3)

    ev = build(conn)

    assert set(ev.priors) == set(SIGNALS)
    assert ev.priors["replies"] == 0.5
    assert ev.priors["claims"] == 0.0


def test_add_cell_accumulates() -> None:
    ledger: dict[str, dict[str, tuple[float, float]]] = {}
    add_cell(ledger, "p", "s", 1.0, 2.0)
    add_cell(ledger, "p", "s", 0.5, 1.0)
    assert ledger == {"p": {"s": (1.5, 3.0)}}
