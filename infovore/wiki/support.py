from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from infovore.claims.check import check_claim
from infovore.claims.value import tokens
from infovore.triage.lexicon import Lexicon

FLOOR: Final = 0.25
MIN_TOKENS: Final = 3


@dataclass(frozen=True)
class Dropped:
    text: str
    cited: list[str]
    reason: str


def _reason(text: str, cited: list[str], lexicon: Lexicon, floor: float) -> str | None:
    if len(tokens(text)) < MIN_TOKENS:
        return "too_short"
    verdict = check_claim(text, cited, lexicon, floor).verdict
    return verdict if verdict in ("unsupported_fact", "low_overlap") else None


def keep_supported(
    sentences: Sequence[tuple[str, Sequence[int]]],
    statements: Sequence[str],
    lexicon: Lexicon,
    floor: float = FLOOR,
) -> tuple[list[tuple[str, list[int]]], list[Dropped]]:
    kept: list[tuple[str, list[int]]] = []
    dropped: list[Dropped] = []
    for text, citations in sentences:
        positions = list(dict.fromkeys(citations))
        valid = [p for p in positions if 0 <= p < len(statements)]
        cited = [statements[p] for p in valid]
        if not positions:
            reason: str | None = "no_citation"
        elif len(valid) < len(positions):
            reason = "bad_citation"
        else:
            reason = _reason(text, cited, lexicon, floor)
        if reason is None:
            kept.append((text, positions))
        else:
            dropped.append(Dropped(text, cited, reason))
    return kept, dropped
