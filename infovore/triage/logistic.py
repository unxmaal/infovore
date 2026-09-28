"""Pure-Python L2-regularized logistic regression (issue #95, deliverable 2).

Standard library only — no numpy/sklearn (per the issue's non-goals: no new
heavy deps). Deterministic: a fixed number of full-batch gradient-descent
iterations at a fixed learning rate, weights and intercept initialized to
zero, examples visited in the given order every iteration — no randomness
anywhere, so the same examples in the same order always produce the exact
same model.

Examples are sparse feature vectors, in either of two equivalent shapes:

- a `frozenset[str]`/other `Iterable[str]` of *active* feature names, every
  other known feature implicitly 0 and every active one implicitly 1 -- the
  representation `infovore.triage.bayes.features` and `infovore.triage.tuning`
  already use for the exchange-level Bayes/rule classifiers; or
- a `Mapping[str, float]` of feature name to a continuous value (issue #135:
  the message classifier's combiner fits real-valued inputs like
  `logit(p_citation)`, not indicator features), which the first form is a
  special case of (`{name: 1.0 for name in features}`) -- a caller passing
  binary features gets exactly the same fit either way, since a weight of
  1.0 multiplies a coefficient unchanged.

A training pass is therefore O(iterations * total active feature-value pairs
across all examples), not O(iterations * examples * vocabulary size).
"""

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

DEFAULT_ITERATIONS = 200
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_L2 = 1.0

__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_L2",
    "DEFAULT_LEARNING_RATE",
    "FeatureValues",
    "LogisticModel",
    "predict_proba",
    "sigmoid",
    "train_logistic",
]

# Either a set of active binary feature names, or a mapping of feature name
# to its continuous value -- see the module docstring.
FeatureValues = Mapping[str, float] | Iterable[str]


@dataclass(frozen=True)
class LogisticModel:
    """A fitted (or zero) logistic model: `intercept` plus one weight per
    feature name seen during training. A feature not in `weights` is treated
    as weight 0.0 (e.g. a feature never seen in training data)."""

    intercept: float
    weights: Mapping[str, float] = field(default_factory=dict)


def sigmoid(x: float) -> float:
    """The logistic function, computed so it never overflows `math.exp` for
    a large-magnitude `x` of either sign."""
    if x >= 0:
        denominator = 1.0 + math.exp(-x)
        return 1.0 / denominator
    numerator = math.exp(x)
    return numerator / (1.0 + numerator)


def _feature_items(features: FeatureValues) -> Iterator[tuple[str, float]]:
    """Normalize either shape `FeatureValues` can take to `(name, value)`
    pairs: a `Mapping` yields its own `(name, value)` items unchanged; any
    other iterable of names is treated as binary, each active name paired
    with an implicit value of `1.0`."""
    if isinstance(features, Mapping):
        yield from features.items()
        return
    for name in features:
        yield (name, 1.0)


def _logit(
    intercept: float, weights: Mapping[str, float], items: Iterable[tuple[str, float]]
) -> float:
    total = intercept
    for name, value in items:
        total += weights.get(name, 0.0) * value
    return total


def predict_proba(model: LogisticModel, features: FeatureValues) -> float:
    """P(positive class) for an example described by `features` -- either a
    binary active-feature set (any iterable of feature names; a
    `frozenset[str]` from `infovore.triage.bayes.features` works directly)
    or a `Mapping[str, float]` of feature name to continuous value (issue
    #135's combiner)."""
    return sigmoid(_logit(model.intercept, model.weights, _feature_items(features)))


def train_logistic(
    examples: Sequence[tuple[FeatureValues, bool]],
    iterations: int = DEFAULT_ITERATIONS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    l2: float = DEFAULT_L2,
) -> LogisticModel:
    """Fit an intercept and per-feature weight minimizing L2-regularized
    average log loss, `mean(log_loss) + (l2 / 2n) * sum(w**2)`, by
    full-batch gradient descent. Scaling the penalty by `1/n` (the same
    average the data term already uses) keeps `l2` a dataset-size-independent
    "how much to shrink" knob, and keeps gradient steps stable across widely
    different `l2` values at one fixed `learning_rate`.

    `examples` pairs a sparse feature set (binary or continuous -- see
    `FeatureValues`) with a bool label (`True` for the positive class, e.g.
    lore). The intercept is never regularized (only `weights` are), the
    standard convention: it exists to fit the overall class balance, not to
    be shrunk toward it.

    With no examples, returns a model with a zero intercept and no weights
    (`predict_proba` on it is always 0.5), so callers don't need a special
    case for "nothing to fit".
    """
    total = len(examples)
    if total == 0:
        return LogisticModel(intercept=0.0, weights={})

    materialized: list[tuple[list[tuple[str, float]], bool]] = [
        (list(_feature_items(features)), label) for features, label in examples
    ]

    weights: dict[str, float] = {}
    for items, _ in materialized:
        for name, _value in items:
            weights.setdefault(name, 0.0)

    intercept = 0.0
    for _ in range(iterations):
        errors = [
            sigmoid(_logit(intercept, weights, items)) - (1.0 if label else 0.0)
            for items, label in materialized
        ]
        gradient_weights = dict.fromkeys(weights, 0.0)
        for (items, _), error in zip(materialized, errors, strict=True):
            for name, value in items:
                gradient_weights[name] += error * value
        for name, summed_error in gradient_weights.items():
            gradient_weights[name] = summed_error / total + (l2 / total) * weights[name]
        gradient_intercept = sum(errors) / total

        intercept -= learning_rate * gradient_intercept
        for name in weights:
            weights[name] -= learning_rate * gradient_weights[name]

    return LogisticModel(intercept=intercept, weights=weights)
