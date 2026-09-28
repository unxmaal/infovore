import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.rows import MessageLabel
from infovore.sift.export import SiftBatchMessage
from infovore.sift.importer import TRASH_RULES_DIR_NAME
from infovore.sift.rules import (
    BulkRule,
    InvalidRuleError,
    RuleType,
    parse_bulk_rule,
    preview_rule,
    rule_matches,
    save_bulk_rule,
)


def _message(
    id: int = 1,
    exchange_id: int = 1,
    channel_name: str = "general",
    author_name: str = "alice",
    content: str = "hello world",
) -> SiftBatchMessage:
    return SiftBatchMessage(
        id=id,
        exchange_id=exchange_id,
        channel_name=channel_name,
        author_name=author_name,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        content=content,
        p_trash=None,
    )


# --- parse_bulk_rule -------------------------------------------------------


def test_parse_bulk_rule_contains() -> None:
    rule = parse_bulk_rule({"type": "contains", "value": "lol"})
    assert rule == BulkRule(type=RuleType.CONTAINS, value="lol")


def test_parse_bulk_rule_exact() -> None:
    rule = parse_bulk_rule({"type": "exact", "value": "gg"})
    assert rule.type is RuleType.EXACT
    assert rule.value == "gg"


def test_parse_bulk_rule_author() -> None:
    rule = parse_bulk_rule({"type": "author", "value": "bob"})
    assert rule.type is RuleType.AUTHOR


def test_parse_bulk_rule_channel() -> None:
    rule = parse_bulk_rule({"type": "channel", "value": "off-topic"})
    assert rule.type is RuleType.CHANNEL


def test_parse_bulk_rule_shorter_than() -> None:
    rule = parse_bulk_rule({"type": "shorter_than", "max_length": 5})
    assert rule.type is RuleType.SHORTER_THAN
    assert rule.max_length == 5


def test_parse_bulk_rule_regex() -> None:
    rule = parse_bulk_rule({"type": "regex", "value": r"^lol+$"})
    assert rule.type is RuleType.REGEX
    assert rule.value == r"^lol+$"


def test_parse_bulk_rule_rejects_unknown_type() -> None:
    with pytest.raises(InvalidRuleError, match="unknown rule type"):
        parse_bulk_rule({"type": "bogus", "value": "x"})


def test_parse_bulk_rule_rejects_missing_value() -> None:
    with pytest.raises(InvalidRuleError, match="non-empty value"):
        parse_bulk_rule({"type": "contains"})


def test_parse_bulk_rule_rejects_blank_value() -> None:
    with pytest.raises(InvalidRuleError, match="non-empty value"):
        parse_bulk_rule({"type": "contains", "value": "   "})


def test_parse_bulk_rule_rejects_non_string_value() -> None:
    with pytest.raises(InvalidRuleError, match="non-empty value"):
        parse_bulk_rule({"type": "contains", "value": 5})


def test_parse_bulk_rule_rejects_missing_max_length() -> None:
    with pytest.raises(InvalidRuleError, match="positive integer"):
        parse_bulk_rule({"type": "shorter_than"})


def test_parse_bulk_rule_rejects_non_positive_max_length() -> None:
    with pytest.raises(InvalidRuleError, match="positive integer"):
        parse_bulk_rule({"type": "shorter_than", "max_length": 0})


def test_parse_bulk_rule_rejects_bool_max_length() -> None:
    with pytest.raises(InvalidRuleError, match="positive integer"):
        parse_bulk_rule({"type": "shorter_than", "max_length": True})


def test_parse_bulk_rule_rejects_non_int_max_length() -> None:
    with pytest.raises(InvalidRuleError, match="positive integer"):
        parse_bulk_rule({"type": "shorter_than", "max_length": "5"})


def test_parse_bulk_rule_rejects_invalid_regex() -> None:
    with pytest.raises(InvalidRuleError, match="invalid regex"):
        parse_bulk_rule({"type": "regex", "value": "("})


# --- rule_matches -----------------------------------------------------------


def test_rule_matches_contains_is_case_insensitive() -> None:
    rule = BulkRule(type=RuleType.CONTAINS, value="WORLD")
    assert rule_matches(rule, _message(content="hello world")) is True
    assert rule_matches(rule, _message(content="hello there")) is False


def test_rule_matches_exact_is_case_sensitive() -> None:
    rule = BulkRule(type=RuleType.EXACT, value="lol")
    assert rule_matches(rule, _message(content="lol")) is True
    assert rule_matches(rule, _message(content="LOL")) is False
    assert rule_matches(rule, _message(content="lol!")) is False


def test_rule_matches_author_is_case_insensitive() -> None:
    rule = BulkRule(type=RuleType.AUTHOR, value="Alice")
    assert rule_matches(rule, _message(author_name="alice")) is True
    assert rule_matches(rule, _message(author_name="bob")) is False


def test_rule_matches_channel_is_case_insensitive() -> None:
    rule = BulkRule(type=RuleType.CHANNEL, value="General")
    assert rule_matches(rule, _message(channel_name="general")) is True
    assert rule_matches(rule, _message(channel_name="food")) is False


def test_rule_matches_shorter_than_is_exclusive_of_the_bound() -> None:
    rule = BulkRule(type=RuleType.SHORTER_THAN, max_length=4)
    assert rule_matches(rule, _message(content="hi")) is True
    assert rule_matches(rule, _message(content="hiya")) is False
    assert rule_matches(rule, _message(content="hiyaa")) is False


def test_rule_matches_regex() -> None:
    rule = BulkRule(type=RuleType.REGEX, value=r"^lol+$")
    assert rule_matches(rule, _message(content="lolll")) is True
    assert rule_matches(rule, _message(content="not lol")) is False


# --- preview_rule -------------------------------------------------------


def test_preview_rule_counts_matches_and_conflicts() -> None:
    messages = [
        _message(id=1, content="lol nice"),
        _message(id=2, content="lol wat"),
        _message(id=3, content="serious business"),
    ]
    current_labels = {2: MessageLabel.KEEP}
    rule = BulkRule(type=RuleType.CONTAINS, value="lol")

    preview = preview_rule(messages, current_labels, rule)

    assert preview.matched == (1, 2)
    assert preview.matched_count == 2
    assert preview.conflicts == (2,)
    assert preview.conflict_count == 1


def test_preview_rule_no_matches() -> None:
    messages = [_message(id=1, content="serious business")]
    rule = BulkRule(type=RuleType.CONTAINS, value="lol")

    preview = preview_rule(messages, {}, rule)

    assert preview.matched == ()
    assert preview.conflicts == ()


# --- save_bulk_rule -------------------------------------------------------


def test_save_bulk_rule_writes_the_shared_trash_rules_store(tmp_path: Path) -> None:
    rule = BulkRule(type=RuleType.CONTAINS, value="lol")

    path = save_bulk_rule(tmp_path, "lol-chatter", rule, datetime(2026, 1, 2, tzinfo=UTC))

    assert path == tmp_path / TRASH_RULES_DIR_NAME / "lol-chatter.json"
    saved = json.loads(path.read_text())
    assert saved == {
        "name": "lol-chatter",
        "type": "contains",
        "value": "lol",
        "max_length": None,
        "saved_at": "2026-01-02T00:00:00+00:00",
    }


def test_save_bulk_rule_shorter_than(tmp_path: Path) -> None:
    rule = BulkRule(type=RuleType.SHORTER_THAN, max_length=3)

    path = save_bulk_rule(tmp_path, "tiny", rule, datetime(2026, 1, 2, tzinfo=UTC))

    saved = json.loads(path.read_text())
    assert saved["type"] == "shorter_than"
    assert saved["max_length"] == 3
