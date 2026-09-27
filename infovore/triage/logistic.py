"""Pure-Python L2-regularized logistic regression (issue #95, deliverable 2).

Standard library only — no numpy/sklearn (per the issue's non-goals: no new
heavy deps). Deterministic: a fixed number of full-batch gradient-descent
iterations at a fixed learning rate, weights and intercept initialized to
zero, examples visited in the given order every iteration — no randomness
anywhere, so the same examples in the same order always produce the exact
same model.

Examples are sparse binary feature vectors: each is the frozenset of
*active* feature names for that example (every other known feature is
implicitly 0), the same representation `infovore.triage.bayes.features`
already uses for the Bayes classifier. A training pass is therefore
O(iterations * total active features across all examples), not
O(iterations * examples * vocabulary size).
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

DEFAULT_ITERATIONS = 200
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_L2 = 1.0

__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_L2",
    "DEFAULT_LEARNING_RATE",
    "LogisticModel",
    "predict_proba",
    "sigmoid",
    "train_logistic",
]


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


def _logit(intercept: float, weights: Mapping[str, float], features: Iterable[str]) -> float:
    total = intercept
    for name in features:
        total += weights.get(name, 0.0)
    return total


def predict_proba(model: LogisticModel, features: Iterable[str]) -> float:
    """P(positive class) for an example whose active features are `features`
    (any iterable of feature names; a `frozenset[str]` from
    `infovore.triage.bayes.features` works directly)."""
    return sigmoid(_logit(model.intercept, model.weights, features))


def train_logistic(
    examples: Sequence[tuple[frozenset[str], bool]],
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

    `examples` pairs a sparse active-feature set with a bool label (`True`
    for the positive class, e.g. lore). The intercept is never regularized
    (only `weights` are), the standard convention: it exists to fit the
    overall class balance, not to be shrunk toward it.

    With no examples, returns a model with a zero intercept and no weights
    (`predict_proba` on it is always 0.5), so callers don't need a special
    case for "nothing to fit".
    """
    total = len(examples)
    if total == 0:
        return LogisticModel(intercept=0.0, weights={})

    weights: dict[str, float] = {}
    for features, _ in examples:
        for name in features:
            weights.setdefault(name, 0.0)

    intercept = 0.0
    for _ in range(iterations):
        errors = [
            predict_proba(LogisticModel(intercept, weights), features) - (1.0 if label else 0.0)
            for features, label in examples
        ]
        gradient_weights = dict.fromkeys(weights, 0.0)
        for (features, _), error in zip(examples, errors, strict=True):
            for name in features:
                gradient_weights[name] += error
        for name, summed_error in gradient_weights.items():
            gradient_weights[name] = summed_error / total + (l2 / total) * weights[name]
        gradient_intercept = sum(errors) / total

        intercept -= learning_rate * gradient_intercept
        for name in weights:
            weights[name] -= learning_rate * gradient_weights[name]

    return LogisticModel(intercept=intercept, weights=weights)
