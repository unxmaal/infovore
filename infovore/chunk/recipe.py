from dataclasses import dataclass
from datetime import timedelta

from infovore.chunk.rules import DEFAULT_MAX_MESSAGES, DEFAULT_OVERLAP, DEFAULT_QUIET_GAP


@dataclass(frozen=True)
class ChunkRecipe:
    """Every parameter that decides how messages become exchanges. `exchanges`
    records `grouping_rule` (which rule fired) but recorded none of these, so
    an exchange meant "whatever the chunker produced the day it ran" and
    re-chunking could not be compared against what came before (issue #176)."""

    quiet_gap: timedelta
    max_messages: int
    overlap: int


def current_recipe() -> ChunkRecipe:
    return ChunkRecipe(
        quiet_gap=DEFAULT_QUIET_GAP,
        max_messages=DEFAULT_MAX_MESSAGES,
        overlap=DEFAULT_OVERLAP,
    )
