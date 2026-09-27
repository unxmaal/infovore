import math

import pytest

from infovore.triage.logistic import LogisticModel, predict_proba, sigmoid, train_logistic


def test_sigmoid_of_zero_is_one_half() -> None:
    assert sigmoid(0.0) == 0.5


def test_sigmoid_saturates_without_overflowing() -> None:
    assert sigmoid(1000.0) == pytest.approx(1.0)
    assert sigmoid(-1000.0) == pytest.approx(0.0)


def test_sigmoid_is_symmetric() -> None:
    assert sigmoid(-2.0) == pytest.approx(1.0 - sigmoid(2.0))


def test_predict_proba_matches_manual_sigmoid() -> None:
    model = LogisticModel(intercept=1.0, weights={"x": 2.0, "y": -0.5})
    assert predict_proba(model, frozenset({"x"})) == pytest.approx(sigmoid(3.0))
    assert predict_proba(model, frozenset({"x", "y"})) == pytest.approx(sigmoid(2.5))
    assert predict_proba(model, frozenset()) == pytest.approx(sigmoid(1.0))


def test_predict_proba_ignores_unknown_features() -> None:
    model = LogisticModel(intercept=0.0, weights={"x": 5.0})
    assert predict_proba(model, frozenset({"never-seen"})) == pytest.approx(sigmoid(0.0))


def test_train_logistic_on_no_examples_returns_a_neutral_model() -> None:
    model = train_logistic([])
    assert model.intercept == 0.0
    assert dict(model.weights) == {}
    assert predict_proba(model, frozenset({"anything"})) == 0.5


SEPARABLE_EXAMPLES: list[tuple[frozenset[str], bool]] = [
    (frozenset({"a"}), True),
    (frozenset({"a"}), True),
    (frozenset({"a"}), True),
    (frozenset({"b"}), False),
    (frozenset({"b"}), False),
    (frozenset({"b"}), False),
]


def test_train_logistic_separates_linearly_separable_toy_data() -> None:
    model = train_logistic(SEPARABLE_EXAMPLES, l2=0.01)

    assert model.weights["a"] > 0
    assert model.weights["b"] < 0
    assert predict_proba(model, frozenset({"a"})) > 0.5
    assert predict_proba(model, frozenset({"b"})) < 0.5


def test_train_logistic_is_deterministic_across_runs() -> None:
    first = train_logistic(SEPARABLE_EXAMPLES)
    second = train_logistic(SEPARABLE_EXAMPLES)

    assert first.intercept == second.intercept
    assert dict(first.weights) == dict(second.weights)


def test_regularization_shrinks_weights_toward_zero() -> None:
    light = train_logistic(SEPARABLE_EXAMPLES, l2=0.001)
    heavy = train_logistic(SEPARABLE_EXAMPLES, l2=5.0)

    assert abs(heavy.weights["a"]) < abs(light.weights["a"])
    assert abs(heavy.weights["b"]) < abs(light.weights["b"])


def test_intercept_reflects_class_balance_when_there_are_no_features() -> None:
    empty: frozenset[str] = frozenset()
    imbalanced: list[tuple[frozenset[str], bool]] = [(empty, True) for _ in range(8)] + [
        (empty, False) for _ in range(2)
    ]

    model = train_logistic(imbalanced, l2=0.0)

    assert model.intercept > 0
    assert predict_proba(model, frozenset()) > 0.5
    # With no features at all, the fitted intercept should recover close to
    # the empirical log-odds of the class balance: logit(0.8) = ln(0.8/0.2).
    assert model.intercept == pytest.approx(math.log(0.8 / 0.2), abs=0.05)
