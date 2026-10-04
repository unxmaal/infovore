import pytest

from infovore.wiki.eligibility import is_publishable


@pytest.mark.parametrize("check", ["supported", "uncheckable"])
@pytest.mark.parametrize("review", [None, "good"])
def test_unflagged_checked_claims_publish(check: str, review: str | None) -> None:
    assert is_publishable(review, check)


@pytest.mark.parametrize("review", ["wrong", "made_up", "not_useful"])
def test_rejected_by_eric_never_publishes(review: str) -> None:
    assert not is_publishable(review, "supported")


@pytest.mark.parametrize("check", ["unsupported_fact", "low_overlap", None])
def test_flagged_or_unchecked_claims_wait(check: str | None) -> None:
    assert not is_publishable(None, check)


def test_a_good_review_does_not_override_a_flag() -> None:
    assert not is_publishable("good", "unsupported_fact")
