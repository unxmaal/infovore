"""Bulk trash rules for `sift serve`'s "hide everything like this" panel
(issue #131). The server is the single source of truth for what a rule
matches: the page only calls `/api/rules/preview` and `/api/rules/apply`
(`infovore.sift.httpd`), and both endpoints call `preview_rule` /
`rule_matches` below, so there is never a separate client-side matcher to
drift out of sync with the server's.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from infovore.db.codec import to_db_time
from infovore.rows import MessageLabel
from infovore.sift.export import SiftBatchMessage
from infovore.sift.importer import TRASH_RULES_DIR_NAME


class RuleType(StrEnum):
    CONTAINS = "contains"
    EXACT = "exact"
    AUTHOR = "author"
    CHANNEL = "channel"
    SHORTER_THAN = "shorter_than"
    REGEX = "regex"


class InvalidRuleError(Exception):
    pass


@dataclass(frozen=True)
class BulkRule:
    type: RuleType
    value: str = ""
    max_length: int | None = None


@dataclass(frozen=True)
class RulePreview:
    matched: tuple[int, ...]
    conflicts: tuple[int, ...]

    @property
    def matched_count(self) -> int:
        return len(self.matched)

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)


def parse_bulk_rule(payload: Mapping[str, object]) -> BulkRule:
    """Build a `BulkRule` from a bulk-hide panel request body, raising
    `InvalidRuleError` (rendered as HTTP 400 by `infovore.sift.httpd`) for
    anything the panel could not have produced: an unknown `type`, a blank
    `value` where one is required, a non-positive/non-integer `max_length`
    for `shorter_than`, or a `regex` that doesn't compile."""
    raw_type = payload.get("type")
    if not isinstance(raw_type, str):
        raise InvalidRuleError(f"unknown rule type {raw_type!r}")
    try:
        rule_type = RuleType(raw_type)
    except ValueError as error:
        raise InvalidRuleError(f"unknown rule type {raw_type!r}") from error

    if rule_type is RuleType.SHORTER_THAN:
        max_length = payload.get("max_length")
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise InvalidRuleError("shorter_than requires a positive integer max_length")
        return BulkRule(type=rule_type, max_length=max_length)

    value = payload.get("value")
    if not isinstance(value, str) or not value.strip():
        raise InvalidRuleError(f"{rule_type.value} requires a non-empty value")
    if rule_type is RuleType.REGEX:
        try:
            re.compile(value)
        except re.error as error:
            raise InvalidRuleError(f"invalid regex: {error}") from error
    return BulkRule(type=rule_type, value=value)


def rule_matches(rule: BulkRule, message: SiftBatchMessage) -> bool:
    """Whether one bulk rule matches one batch message — the panel's five
    kinds (plus regex), each a direct, literal test against the message's
    own fields rather than a reconstructed log line."""
    if rule.type is RuleType.CONTAINS:
        return rule.value.lower() in message.content.lower()
    if rule.type is RuleType.EXACT:
        return message.content == rule.value
    if rule.type is RuleType.AUTHOR:
        return message.author_name.lower() == rule.value.lower()
    if rule.type is RuleType.CHANNEL:
        return message.channel_name.lower() == rule.value.lower()
    if rule.type is RuleType.SHORTER_THAN:
        assert rule.max_length is not None
        return len(message.content) < rule.max_length
    return re.search(rule.value, message.content) is not None  # RuleType.REGEX


def preview_rule(
    messages: Sequence[SiftBatchMessage],
    current_labels: Mapping[int, MessageLabel],
    rule: BulkRule,
) -> RulePreview:
    """How many batch messages a rule would trash, and how many of those
    already carry a `keep` label (a conflict the bulk-hide panel warns
    about before the maintainer confirms)."""
    matched = tuple(message.id for message in messages if rule_matches(rule, message))
    conflicts = tuple(
        message_id for message_id in matched if current_labels.get(message_id) is MessageLabel.KEEP
    )
    return RulePreview(matched=matched, conflicts=conflicts)


def save_bulk_rule(scratch_dir: Path, name: str, rule: BulkRule, at: datetime) -> Path:
    """Store a named bulk rule (issue #131) in the same directory
    `sift import --save-rules` uses (`infovore.sift.importer.
    save_trash_rules`, `TRASH_RULES_DIR_NAME`) for a human to review before
    any future corpus-wide use (issue #128's later audit item) — this PR,
    like that one, only stores it, applying it corpus-wide is a later
    issue. The file holds all five rule kinds (`type`/`value`/
    `max_length`), a richer schema than `save_trash_rules`' regex-only
    `patterns` list, since a rule saved here is only sometimes a regex."""
    rules_dir = scratch_dir / TRASH_RULES_DIR_NAME
    rules_dir.mkdir(parents=True, exist_ok=True)
    path = rules_dir / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "type": rule.type.value,
                "value": rule.value,
                "max_length": rule.max_length,
                "saved_at": to_db_time(at),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
