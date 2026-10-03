from dataclasses import dataclass, replace
from datetime import timedelta

from infovore.chunk.rules import (
    DEFAULT_FOLD_FACTOR,
    DEFAULT_FOLD_SIZE,
    DEFAULT_GAP_CEILING,
    DEFAULT_GAP_PERCENTILE,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_OVERLAP,
    DEFAULT_QUIET_GAP,
    ZERO,
)


@dataclass(frozen=True)
class ChunkRecipe:
    """Every parameter that decides how messages become exchanges. `exchanges`
    records `grouping_rule` (which rule fired) but recorded none of these, so
    an exchange meant "whatever the chunker produced the day it ran" and
    re-chunking could not be compared against what came before (issue #176)."""

    quiet_gap: timedelta
    max_messages: int
    overlap: int
    gap_percentile: float = 0.0
    gap_floor: timedelta = ZERO
    gap_ceiling: timedelta = ZERO
    fold_factor: float = 0.0
    fold_size: int = 0

    @property
    def adaptive(self) -> bool:
        return self.gap_percentile > 0


def current_recipe() -> ChunkRecipe:
    return ChunkRecipe(
        quiet_gap=DEFAULT_QUIET_GAP,
        max_messages=DEFAULT_MAX_MESSAGES,
        overlap=DEFAULT_OVERLAP,
        gap_percentile=DEFAULT_GAP_PERCENTILE,
        gap_floor=DEFAULT_QUIET_GAP,
        gap_ceiling=DEFAULT_GAP_CEILING,
        fold_factor=DEFAULT_FOLD_FACTOR,
        fold_size=DEFAULT_FOLD_SIZE,
    )


def settings_recipe(quiet_gap: timedelta, max_messages: int) -> ChunkRecipe:
    return replace(
        current_recipe(), quiet_gap=quiet_gap, gap_floor=quiet_gap, max_messages=max_messages
    )
