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


def test_both_versions_are_available() -> None:
    assert set(PROMPTS) == {"v5", "v6"}


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
    assert "probe_question" in v6
    assert "supersedes" in v6
    assert "JSON" in v6
