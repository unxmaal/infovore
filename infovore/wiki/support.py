from collections.abc import Sequence
from typing import Final

from infovore.claims.value import tokens

SUPPORT: Final = 0.6
MIN_TOKENS: Final = 3


def support(sentence: str, cited: Sequence[str]) -> float:
    words = tokens(sentence)
    if not words:
        return 0.0
    known = frozenset().union(*(tokens(c) for c in cited))
    return len(words & known) / len(words)


def keep_supported(
    sentences: Sequence[tuple[str, Sequence[int]]],
    statements: Sequence[str],
    threshold: float = SUPPORT,
) -> tuple[list[tuple[str, list[int]]], int]:
    kept: list[tuple[str, list[int]]] = []
    dropped = 0
    for text, citations in sentences:
        positions = list(dict.fromkeys(citations))
        if (
            not positions
            or len(tokens(text)) < MIN_TOKENS
            or any(not 0 <= p < len(statements) for p in positions)
            or support(text, [statements[p] for p in positions]) < threshold
        ):
            dropped += 1
        else:
            kept.append((text, positions))
    return kept, dropped
