from typing import Final

REJECTED_REVIEWS: Final = frozenset({"wrong", "made_up", "not_useful"})
PUBLISHABLE_CHECKS: Final = frozenset({"supported", "uncheckable"})


def is_publishable(review: str | None, check: str | None) -> bool:
    return review not in REJECTED_REVIEWS and check in PUBLISHABLE_CHECKS
