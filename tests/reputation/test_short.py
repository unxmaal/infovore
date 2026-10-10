from pathlib import Path

import pytest

from infovore.reputation.evidence import Evidence, Ledger
from infovore.reputation.people import People
from infovore.reputation.score import Reputation, build_reputation
from infovore.reputation.short import ShortMessage, short_messages, short_test
from infovore.triage.lexicon import load_lexicon, message_hits
from tests.reputation.world import exchange, label_message, world

HIT = "my Indy runs IRIX 6.5.22 with a PROM"
PLAIN = "lol ok fine then"
PRIORS = dict.fromkeys(
    ("replies", "reactions", "mentions", "answered", "messages", "exchanges", "claims"), 0.5
)
NOBODY = People({}, frozenset(), {})


def test_the_shipped_lexicon_separates_the_fixture_lines() -> None:
    lexicon = load_lexicon()

    assert message_hits(lexicon, HIT)
    assert not message_hits(lexicon, PLAIN)


def test_only_short_human_labelled_messages_without_lexicon_hits_qualify(tmp_path: Path) -> None:
    conn = world(tmp_path)
    cases = [
        (PLAIN, True, "context", "human"),
        (PLAIN + " again", False, "value", "human"),
        (HIT, True, "context", "human"),
        ("x" * 200, True, "context", "human"),
        (PLAIN + " isolated", True, "isolated", "human"),
        (PLAIN + " cited", True, "context", "citation"),
    ]
    wanted = []
    for base, (text, keep, regime, source) in enumerate(cases, start=1):
        _, (message,) = exchange(conn, [(base, "a", text)], base)
        label_message(conn, message, keep, regime, source)
        wanted.append(message)
    exchange(conn, [(9, "a", PLAIN + " unlabelled")], 9)

    found = short_messages(conn, load_lexicon(), frozenset())

    assert [(m.message_id, m.author_id, m.keep) for m in found] == [
        (wanted[0], 1, True),
        (wanted[1], 2, False),
    ]
    assert short_messages(conn, load_lexicon(), frozenset({"c"})) == []


def reputation_of(totals: Ledger, contributions: dict[int, Ledger] | None = None) -> Reputation:
    return build_reputation(Evidence(totals, PRIORS, contributions or {}), NOBODY)


def test_the_top_decile_keeps_more_than_the_base_rate_when_reputation_predicts() -> None:
    totals = {str(i): {"replies": (float(i), 10.0)} for i in range(1, 31)}
    messages = [ShortMessage(i, i, i, keep=i > 25) for i in range(1, 31)]

    result = short_test(reputation_of(totals), messages, seed=0, permutations=2000)

    assert result is not None
    assert (result.n, result.top_n) == (30, 3)
    assert result.base_rate == pytest.approx(5 / 30)
    assert result.top_rate == 1.0
    assert 0.0 < result.low < result.high <= 1.0
    assert result.p < 0.01


def test_the_authors_reputation_is_computed_without_the_messages_own_exchange() -> None:
    totals = {"1": {"replies": (100.0, 100.0)}, "2": {"replies": (80.0, 100.0)}}
    contributions = {7: {"1": {"replies": (100.0, 100.0)}}}
    messages = [ShortMessage(1, 7, 1, keep=False), ShortMessage(2, 8, 2, keep=True)]

    result = short_test(reputation_of(totals, contributions), messages, seed=0, permutations=10)

    assert result is not None
    assert result.top_rate == 1.0


def test_no_messages_give_no_test() -> None:
    assert short_test(reputation_of({}), [], seed=0) is None
