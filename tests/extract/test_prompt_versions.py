import pytest

from infovore.extract.prompt import (
    LIVE_PROMPT_VERSION,
    PROMPT_SHA256,
    PROMPT_VERSION,
    PROMPTS,
    UnknownPromptVersionError,
    system_prompt,
)

# The digest promoted in the live database on 2026-09-28.
PROMOTED_V5_SHA256 = "89d0fe56b602aa96adc6124ce3e37219123c9a75080b1e1ad228cdf98f475a4a"


def test_every_version_is_available() -> None:
    assert set(PROMPTS) == {"v5", "v6", "v7"}


def test_the_live_version_is_still_v5() -> None:
    """Bumping this constant halts live extraction until the new version is
    promoted (`runner.py` raises `PromptNotPromotedError`). v6 ships
    selectable and unpromoted so it can be measured without stopping
    anything."""
    assert PROMPT_VERSION == "v5"
    assert LIVE_PROMPT_VERSION == "v5"


def test_the_live_prompt_text_is_unchanged() -> None:
    """v5 is promoted in the live database against this digest. Editing it
    silently would make every historical run's `prompt_version` a lie."""
    assert PROMPT_SHA256 == PROMOTED_V5_SHA256


def test_system_prompt_defaults_to_the_live_version() -> None:
    assert system_prompt() == PROMPTS["v5"]


def test_system_prompt_selects_by_version() -> None:
    assert system_prompt("v6") == PROMPTS["v6"]
    assert PROMPTS["v6"] != PROMPTS["v5"]


def test_an_unknown_version_is_refused() -> None:
    with pytest.raises(UnknownPromptVersionError):
        system_prompt("v99")


def test_v6_does_not_phrase_the_privacy_rule_as_a_substitution() -> None:
    """v5 said "never name people; say a community member instead" and the
    model used the substitute as a sentence template: 48.7% of statements
    began with it (issue #165, RULE #304). v6 must state the constraint as a
    property of the output instead."""
    v6 = PROMPTS["v6"]

    assert "a community member instead" not in v6
    assert "community member" not in v6.lower()


def test_v6_does_not_delegate_filtering_to_the_probe() -> None:
    """ "Extract generously, a later probe is the filter" made the producer
    indiscriminate and pushed the cost onto the stage that cost 1.66B tokens
    to suppress 7.8% of claims."""
    v6 = PROMPTS["v6"]

    assert "generously" not in v6.lower()
    assert "the filter, not you" not in v6


def test_v6_keeps_the_structural_rules_the_schema_depends_on() -> None:
    """The ref/citation rules are not style: `ClaimOut` validates sources
    against the refs the prompt handed out, and context refs are rejected."""
    v6 = PROMPTS["v6"]

    assert "sources" in v6
    assert "CONTEXT" in v6
    assert "supersedes" in v6
    assert "JSON" in v6


def test_v6_does_not_ask_for_a_probe_question() -> None:
    """The field fed the closed-book probe only. With novelty redefined as
    corpus novelty, asking for it buys 34.8% of the claim payload for
    nothing (issue #165). v5 still asks, because v5 is promoted and its
    recorded runs must stay reproducible."""
    assert "probe_question" not in PROMPTS["v6"]
    assert "probe_question" in PROMPTS["v5"]


def test_v7_is_v6_with_only_the_hedging_rule_changed() -> None:
    """v6 measured 26.5% recovery of v5's cited sources against v5's own 83%
    ceiling, and produced nothing on 20 of 40 exchanges, because refusing
    anything hedged also refuses facts: in this archive almost every real
    fact arrives attributed and tentative. v7 changes that paragraph and
    nothing else, so the next comparison attributes any difference to it."""
    v6, v7 = PROMPTS["v6"], PROMPTS["v7"]
    unchanged = [
        "Every claim is about a THING",
        "An occasion is not a fact",
        "Never return more than five claims",
        "sources",
        "CONTEXT",
        "supersedes",
    ]

    for fragment in unchanged:
        assert fragment in v6
        assert fragment in v7
    assert "leave it out rather than hedging it in words" in v6
    assert "leave it out rather than hedging it in words" not in v7


def test_v7_keeps_a_tentatively_stated_fact_and_refuses_a_guess() -> None:
    v7 = PROMPTS["v7"]

    assert "A fact stated tentatively is still a fact" in v7
    assert "GUESS about what might be true" in v7
    assert "confidence rather than in the words" in v7


def test_v6_is_unchanged_because_its_results_are_published() -> None:
    """Editing a measured prompt in place makes the published figure describe
    a text that no longer exists, which is why v5 carries a pinned digest."""
    assert "probe_question" not in PROMPTS["v6"]
    assert len(PROMPTS["v6"]) == 2372


def test_v7_does_not_ask_for_a_probe_question_either() -> None:
    assert "probe_question" not in PROMPTS["v7"]
