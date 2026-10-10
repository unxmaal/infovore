import math

import pytest

from infovore.reputation.evidence import SIGNALS, Evidence
from infovore.reputation.people import People
from infovore.reputation.score import (
    BAN_MARGIN,
    CLAIM_WEIGHT,
    K_LABEL,
    K_RESPONSE,
    Smoothing,
    breakdown,
    build_reputation,
    cells_after,
    earned,
    lift,
    scores,
    weight,
)

PRIORS = dict.fromkeys(SIGNALS, 0.5)


def people(*banned: str) -> People:
    return People({}, frozenset(banned), {})


def evidence(totals: dict[str, dict[str, tuple[float, float]]]) -> Evidence:
    return Evidence(totals, PRIORS, {})


def test_lift_is_zero_without_a_prior_or_without_data() -> None:
    assert lift(3.0, 4.0, 0.0, K_RESPONSE) == 0.0
    assert lift(0.0, 0.0, 0.5, K_RESPONSE) == 0.0


def test_lift_follows_the_smoothed_rate_against_the_prior() -> None:
    assert lift(2.0, 2.0, 0.5, 20.0) == pytest.approx(math.log((2 + 10) / 22 / 0.5))
    assert lift(2.0, 2.0, 0.5, 20.0) > 0
    assert lift(0.0, 4.0, 0.5, 20.0) < 0


def test_a_larger_k_pulls_the_lift_toward_zero() -> None:
    assert abs(lift(4.0, 4.0, 0.5, 100.0)) < abs(lift(4.0, 4.0, 0.5, 1.0))


def test_k_is_per_signal_family() -> None:
    smoothing = Smoothing(k_response=3.0, k_label=7.0)
    assert smoothing.k("replies") == 3.0
    assert smoothing.k("messages") == 7.0
    assert (Smoothing().k("answered"), Smoothing().k("claims")) == (K_RESPONSE, K_LABEL)


def test_claims_weigh_less_than_the_other_signals() -> None:
    assert weight("claims") == CLAIM_WEIGHT < weight("replies") == 1.0


def test_removal_subtracts_clamps_and_leaves_other_signals_alone() -> None:
    cells = {"replies": (1.0, 5.0), "messages": (3.0, 2.0)}

    after = cells_after(cells, {"replies": (2.0, 1.0)})

    assert after["replies"] == (0.0, 4.0)
    assert after["messages"] == (3.0, 2.0)
    assert after["claims"] == (0.0, 0.0)
    assert set(after) == set(SIGNALS)


def test_reputation_is_the_weighted_sum_of_lifts() -> None:
    cells = {"replies": (2.0, 2.0), "claims": (1.0, 1.0)}
    smoothing = Smoothing()

    parts = breakdown(cells, PRIORS, smoothing)

    assert parts["replies"] == pytest.approx(lift(2.0, 2.0, 0.5, K_RESPONSE))
    assert parts["claims"] == pytest.approx(CLAIM_WEIGHT * lift(1.0, 1.0, 0.5, K_LABEL))
    assert earned(cells, PRIORS, smoothing) == pytest.approx(sum(parts.values()))
    assert earned({}, PRIORS, smoothing) == 0.0


def test_a_person_without_evidence_scores_zero() -> None:
    rep = build_reputation(evidence({"a": {"replies": (1.0, 1.0)}}), people())

    assert rep.of("nobody") == 0.0


def test_a_banned_person_sits_below_every_earned_score() -> None:
    totals = {
        "good": {"replies": (9.0, 9.0)},
        "bad": {"replies": (0.0, 99.0)},
        "ban": {"replies": (99.0, 99.0)},
    }
    rep = build_reputation(evidence(totals), people("ban"))

    earned_scores = [score for person, score in scores(rep).items() if person != "ban"]
    assert rep.of("ban") == rep.floor == min(earned_scores) - BAN_MARGIN
    assert rep.of("ban", {"ban": {"replies": (99.0, 99.0)}}) == rep.floor
    assert scores(rep)["ban"] == rep.floor


def test_the_floor_without_anyone_earning_is_the_margin() -> None:
    rep = build_reputation(evidence({"ban": {"replies": (1.0, 1.0)}}), people("ban"))

    assert rep.floor == -BAN_MARGIN


def test_removing_a_contribution_lowers_the_score_of_a_high_scorer() -> None:
    rep = build_reputation(evidence({"a": {"replies": (10.0, 10.0)}}), people())

    assert rep.of("a", {"a": {"replies": (5.0, 5.0)}}) < rep.of("a")
    assert rep.of("a", {"b": {"replies": (5.0, 5.0)}}) == rep.of("a")


def test_smoothing_is_a_parameter_of_the_build() -> None:
    totals = {"a": {"replies": (4.0, 4.0)}}

    tight = build_reputation(evidence(totals), people(), Smoothing(k_response=200.0))
    loose = build_reputation(evidence(totals), people(), Smoothing(k_response=1.0))

    assert tight.of("a") < loose.of("a")
