import math
from datetime import UTC, datetime, timedelta

import pytest

from infovore.rows import MessageRow
from infovore.triage.bayes import (
    Label,
    Metrics,
    chi2q,
    evaluate,
    features,
    in_holdout,
    p_lore,
    recommend_threshold,
    token_probability,
    train,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def msg(message_id: int, content: str) -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=10,
        guild_id=9,
        author_id=message_id,
        author_name_at_time="u",
        author_is_bot=False,
        created_at=START + timedelta(minutes=message_id),
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        deleted_at=None,
        ingested_at=START,
        raw_json="{}",
    )


def test_features_are_lowercased_words_plus_virtual_tokens() -> None:
    found = features([msg(1, "The Octane PSU is 060-0035-003, see /usr/sbin/inst!")], channel_id=42)
    assert {"the", "octane", "psu", "060-0035-003", "/usr/sbin/inst"} <= found
    assert {"SIG_part_number", "SIG_unix_path", "SIG_domain_terms"} <= found
    assert "CHAN_42" in found
    assert "LEN_1" in found
    assert "inst!" not in found


def test_length_buckets() -> None:
    def bucket(count: int) -> str:
        found = features([msg(i, "x") for i in range(count)], channel_id=1)
        return next(token for token in found if token.startswith("LEN_"))

    assert [bucket(1), bucket(3), bucket(10), bucket(30)] == [
        "LEN_1",
        "LEN_2-5",
        "LEN_6-20",
        "LEN_21+",
    ]


def test_chi2q_matches_closed_forms() -> None:
    assert chi2q(0.0, 2) == 1.0
    assert chi2q(3.0, 2) == pytest.approx(math.exp(-1.5))
    assert chi2q(8.3178, 4) == pytest.approx(math.exp(-4.1589) * (1 + 4.1589), rel=1e-4)
    assert chi2q(1e6, 4) == pytest.approx(0.0)


def lore_noise_model():  # type: ignore[no-untyped-def]
    return train(
        [
            (frozenset({"prom", "octane"}), Label.LORE),
            (frozenset({"prom"}), Label.LORE),
            (frozenset({"prom", "lol"}), Label.LORE),
            (frozenset({"lol"}), Label.NOISE),
            (frozenset({"lol", "gm"}), Label.NOISE),
            (frozenset({"gm"}), Label.NOISE),
        ]
    )


def test_token_probability_uses_robinson_smoothing() -> None:
    model = lore_noise_model()
    assert token_probability(model, "prom") == pytest.approx(0.875)
    assert token_probability(model, "gm") == pytest.approx(0.1666667, rel=1e-5)
    assert token_probability(model, "never-seen") == 0.5


def test_single_clue_scores_its_own_probability() -> None:
    assert p_lore(lore_noise_model(), frozenset({"prom"})) == pytest.approx(0.875)


def test_fisher_combining_of_two_strong_clues() -> None:
    model = train(
        [
            (frozenset({"a", "b"}), Label.LORE),
            (frozenset({"a", "b"}), Label.LORE),
            (frozenset({"a", "b"}), Label.LORE),
            (frozenset({"z"}), Label.NOISE),
            (frozenset({"z"}), Label.NOISE),
            (frozenset({"z"}), Label.NOISE),
        ]
    )
    assert p_lore(model, frozenset({"a", "b"})) == pytest.approx(0.94474, rel=1e-4)


def test_no_informative_tokens_scores_half() -> None:
    assert p_lore(lore_noise_model(), frozenset({"unseen", "also-unseen"})) == 0.5


def test_untrained_class_scores_half() -> None:
    model = train([(frozenset({"prom"}), Label.LORE)])
    assert token_probability(model, "prom") == 0.5
    assert p_lore(model, frozenset({"prom"})) == 0.5


def test_only_the_most_interesting_tokens_are_used() -> None:
    model = lore_noise_model()
    few = p_lore(model, frozenset({"prom", "gm"}), max_clues=1)
    assert few == pytest.approx(0.875)


def test_holdout_split_is_stable_and_about_a_fifth() -> None:
    chosen = [exchange_id for exchange_id in range(1000) if in_holdout(exchange_id)]
    assert chosen == [exchange_id for exchange_id in range(1000) if in_holdout(exchange_id)]
    assert 150 < len(chosen) < 250


def test_evaluate_and_recommend_threshold() -> None:
    scored = [
        (0.95, Label.LORE),
        (0.85, Label.LORE),
        (0.6, Label.NOISE),
        (0.4, Label.LORE),
        (0.2, Label.NOISE),
        (0.1, Label.NOISE),
    ]
    table = evaluate(scored, [0.3, 0.5, 0.9])
    assert table[0] == Metrics(0.3, tp=3, fp=1, fn=0, tn=2)
    assert table[1] == Metrics(0.5, tp=2, fp=1, fn=1, tn=2)
    assert table[2].recall == pytest.approx(1 / 3)
    assert table[1].precision == pytest.approx(2 / 3)
    assert table[1].f1 == pytest.approx(2 / 3)
    assert recommend_threshold(table, min_recall=0.6) == table[1]
    assert recommend_threshold(table, min_recall=1.0) == table[0]
    assert recommend_threshold(evaluate([(0.1, Label.LORE)], [0.5]), min_recall=0.5) is None


def test_metrics_with_no_positives_are_zero_not_errors() -> None:
    empty = Metrics(0.5, tp=0, fp=0, fn=0, tn=3)
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)
