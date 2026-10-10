import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from infovore.reputation.evidence import RESPONSE, SIGNALS, Cell, Evidence, Ledger
from infovore.reputation.people import People

K_RESPONSE: Final = 20.0
K_LABEL: Final = 10.0
CLAIM_WEIGHT: Final = 0.25
BAN_MARGIN: Final = 1.0
_EMPTY: Final[Cell] = (0.0, 0.0)


@dataclass(frozen=True)
class Smoothing:
    k_response: float = K_RESPONSE
    k_label: float = K_LABEL

    def k(self, signal: str) -> float:
        return self.k_response if signal in RESPONSE else self.k_label


DEFAULT_SMOOTHING: Final = Smoothing()


def weight(signal: str) -> float:
    return CLAIM_WEIGHT if signal == "claims" else 1.0


def lift(x: float, n: float, prior: float, k: float) -> float:
    if prior <= 0:
        return 0.0
    return math.log((x + k * prior) / (n + k) / prior)


def cells_after(cells: Mapping[str, Cell], removed: Mapping[str, Cell]) -> dict[str, Cell]:
    out: dict[str, Cell] = {}
    for signal in SIGNALS:
        x, n = cells.get(signal, _EMPTY)
        rx, rn = removed.get(signal, _EMPTY)
        out[signal] = (max(0.0, x - rx), max(0.0, n - rn))
    return out


def breakdown(
    cells: Mapping[str, Cell], priors: Mapping[str, float], smoothing: Smoothing
) -> dict[str, float]:
    return {
        signal: weight(signal)
        * lift(*cells.get(signal, _EMPTY), priors[signal], smoothing.k(signal))
        for signal in SIGNALS
    }


def earned(cells: Mapping[str, Cell], priors: Mapping[str, float], smoothing: Smoothing) -> float:
    return sum(breakdown(cells, priors, smoothing).values())


@dataclass(frozen=True)
class Reputation:
    evidence: Evidence
    people: People
    smoothing: Smoothing
    floor: float
    ratings: Mapping[str, int] | None = None
    unrated: float = 0.0

    def of(self, person: str, removed: Ledger | None = None) -> float:
        if self.people.is_banned(person):
            return self.floor
        if self.ratings is not None:
            return float(self.ratings.get(person, self.unrated))
        cells = cells_after(self.evidence.totals.get(person, {}), (removed or {}).get(person, {}))
        return earned(cells, self.evidence.priors, self.smoothing)


def build_reputation(
    evidence: Evidence,
    people: People,
    smoothing: Smoothing = DEFAULT_SMOOTHING,
    ratings: Mapping[str, int] | None = None,
) -> Reputation:
    if ratings is not None:
        unrated = sum(ratings.values()) / len(ratings) if ratings else 0.0
        floor = min(ratings.values(), default=0) - BAN_MARGIN
        return Reputation(evidence, people, smoothing, floor, ratings, unrated)
    lowest = min(
        (
            earned(cells, evidence.priors, smoothing)
            for person, cells in evidence.totals.items()
            if not people.is_banned(person)
        ),
        default=0.0,
    )
    return Reputation(evidence, people, smoothing, lowest - BAN_MARGIN)


def scores(reputation: Reputation) -> dict[str, float]:
    return {person: reputation.of(person) for person in reputation.evidence.totals}
