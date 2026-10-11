import pytest

from infovore.config import ConfigError
from infovore.wiki.eligibility import SUPPORTED, is_publishable, parse_verdicts

CHECKS = ["supported", "uncheckable", "unsupported_fact", "low_overlap", None]


@pytest.mark.parametrize("check", CHECKS)
@pytest.mark.parametrize("review", [None, "good"])
def test_without_a_policy_the_check_verdict_never_blocks_publishing(
    check: str | None, review: str | None
) -> None:
    assert is_publishable(review, check)


@pytest.mark.parametrize("review", ["wrong", "made_up", "not_useful"])
def test_rejected_by_eric_never_publishes(review: str) -> None:
    assert not is_publishable(review, "supported")
    assert not is_publishable(review, "supported", SUPPORTED)


@pytest.mark.parametrize("check", CHECKS)
def test_the_policy_admits_only_listed_verdicts_unless_eric_said_good(check: str | None) -> None:
    assert is_publishable(None, check, SUPPORTED) is (check == "supported")
    assert is_publishable("good", check, SUPPORTED)
    assert is_publishable(None, check, frozenset({"supported", "unchecked"})) is (
        check in ("supported", None)
    )


def test_parse_verdicts_accepts_any_or_a_known_subset() -> None:
    assert parse_verdicts("any") is None
    assert parse_verdicts("supported") == SUPPORTED
    assert parse_verdicts(" supported, uncheckable ,unchecked") == frozenset(
        {"supported", "uncheckable", "unchecked"}
    )


@pytest.mark.parametrize("text", ["", "bogus", "supported,bogus", ","])
def test_parse_verdicts_refuses_unknown_names(text: str) -> None:
    with pytest.raises(ConfigError, match="--verdicts"):
        parse_verdicts(text)
