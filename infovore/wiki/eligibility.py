from typing import Final

REJECTED_REVIEWS: Final = frozenset({"wrong", "made_up", "not_useful"})


def is_publishable(review: str | None, check: str | None) -> bool:
    return review not in REJECTED_REVIEWS
