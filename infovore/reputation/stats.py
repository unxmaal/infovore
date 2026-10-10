import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from infovore.rows import Label
from infovore.triage.bayes import auc

Scored = Sequence[tuple[float, Label]]
LOW, HIGH = 2.5, 97.5


@dataclass(frozen=True)
class Interval:
    n: int
    relevant: int
    irrelevant: int
    auc: float | None
    low: float | None
    high: float | None


@dataclass(frozen=True)
class Accuracy:
    n: int
    threshold: float
    accuracy: float
    baseline: float


def _counts(scored: Scored) -> tuple[int, int]:
    relevant = sum(1 for _, label in scored if label is Label.LORE)
    return relevant, len(scored) - relevant


def percentile(ordered: Sequence[float], pct: float) -> float:
    rank = pct / 100 * (len(ordered) - 1)
    below = math.floor(rank)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (rank - below) * (ordered[above] - ordered[below])


def auc_interval(scored: Scored, seed: int, resamples: int = 1000) -> Interval:
    relevant, irrelevant = _counts(scored)
    point = auc(scored)
    found: list[float] = []
    if point is not None:
        rng = random.Random(seed)
        for _ in range(resamples):
            value = auc(rng.choices(list(scored), k=len(scored)))
            if value is not None:
                found.append(value)
    found.sort()
    low = percentile(found, LOW) if found else None
    high = percentile(found, HIGH) if found else None
    return Interval(len(scored), relevant, irrelevant, point, low, high)


def youden_threshold(scored: Scored) -> float | None:
    relevant, irrelevant = _counts(scored)
    if not relevant or not irrelevant:
        return None
    best, best_gain = None, -math.inf
    for threshold in sorted({score for score, _ in scored}):
        hits = sum(1 for score, label in scored if score >= threshold and label is Label.LORE)
        false = sum(1 for score, label in scored if score >= threshold and label is Label.NOISE)
        gain = hits / relevant - false / irrelevant
        if gain > best_gain:
            best, best_gain = threshold, gain
    return best


def accuracy_at(threshold: float | None, scored: Scored) -> Accuracy | None:
    if threshold is None or not scored:
        return None
    right = sum(1 for score, label in scored if (score >= threshold) is (label is Label.LORE))
    relevant, irrelevant = _counts(scored)
    return Accuracy(
        len(scored), threshold, right / len(scored), max(relevant, irrelevant) / len(scored)
    )


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def rank_average(first: Sequence[float], second: Sequence[float]) -> list[float]:
    if len(first) != len(second):
        raise ValueError("score lists differ in length")
    return [(a + b) / 2 for a, b in zip(average_ranks(first), average_ranks(second), strict=True)]


def spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second):
        raise ValueError("lists differ in length")
    if len(first) < 2:
        return None
    a, b = average_ranks(first), average_ranks(second)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    spread_a = sum((x - mean_a) ** 2 for x in a)
    spread_b = sum((y - mean_b) ** 2 for y in b)
    if not spread_a or not spread_b:
        return None
    cross = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    return cross / math.sqrt(spread_a * spread_b)
