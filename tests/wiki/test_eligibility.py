import pytest

from infovore.wiki.eligibility import is_publishable


@pytest.mark.parametrize(
    "check", ["supported", "uncheckable", "unsupported_fact", "low_overlap", None]
)
@pytest.mark.parametrize("review", [None, "good"])
def test_check_verdict_never_blocks_publishing(check: str | None, review: str | None) -> None:
    assert is_publishable(review, check)


@pytest.mark.parametrize("review", ["wrong", "made_up", "not_useful"])
def test_rejected_by_eric_never_publishes(review: str) -> None:
    assert not is_publishable(review, "supported")
