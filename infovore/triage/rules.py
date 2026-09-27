"""Triage rules as data (issue #95, deliverable 1).

The weights, caps, penalties, thresholds and term lists that
`infovore.triage.score.score_exchange` uses to compute a rule-based triage
score live in a TOML file (shipped as `infovore/triage/rules.toml`,
overridable with `INFOVORE_TRIAGE_RULES=<path>`) rather than as Python
constants, so an operator can tune them without a code change.

`TriageRules` is the loaded, validated, immutable result. Its `version` is a
short hash of the rules' canonical content, used as `TRIAGE_VERSION`: editing
the rules changes the hash, and everything keyed on the previous version is
treated as stale and rescored.
"""

import hashlib
import json
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Final

_SHIPPED_PACKAGE = "infovore.triage"
_SHIPPED_RESOURCE = "rules.toml"


class RulesError(Exception):
    """A triage rules file (shipped or overridden) is missing or malformed."""


@dataclass(frozen=True)
class TriageRules:
    """Immutable, validated triage rules, plus their derived `version`."""

    version: str

    domain_term_weight: float
    domain_term_cap: float
    irix_version_weight: float
    part_number_weight: float
    unix_path_weight: float
    code_weight: float
    archive_link_weight: float
    pdf_attachment_weight: float
    answered_question_weight: float
    agreed_answer_weight: float
    thread_weight: float
    substantial_weight: float

    tiny_penalty: float
    gif_penalty: float
    laughter_penalty: float

    substantial_characters: int
    answer_min_characters: int
    tiny_message_characters: int
    tiny_share_threshold: float
    laughter_share_threshold: float

    domain_terms: tuple[str, ...]
    archive_link_hosts: tuple[str, ...]
    gif_hosts: tuple[str, ...]
    laughter_tokens: tuple[str, ...]
    agreement_emoji: frozenset[str]


_FLOAT_KEYS: Final = (
    "domain_term_weight",
    "domain_term_cap",
    "irix_version_weight",
    "part_number_weight",
    "unix_path_weight",
    "code_weight",
    "archive_link_weight",
    "pdf_attachment_weight",
    "answered_question_weight",
    "agreed_answer_weight",
    "thread_weight",
    "substantial_weight",
    "tiny_penalty",
    "gif_penalty",
    "laughter_penalty",
    "tiny_share_threshold",
    "laughter_share_threshold",
)
_INT_KEYS: Final = (
    "substantial_characters",
    "answer_min_characters",
    "tiny_message_characters",
)
_LIST_KEYS: Final = (
    "domain_terms",
    "archive_link_hosts",
    "gif_hosts",
    "laughter_tokens",
    "agreement_emoji",
)
_ALL_KEYS: Final = frozenset(_FLOAT_KEYS + _INT_KEYS + _LIST_KEYS)


def _canonical_json(data: dict[str, object]) -> str:
    # sort_keys makes key order irrelevant; tomllib already discards whitespace
    # and comments, so this string is the same for any semantically identical
    # rules file.
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _version(data: dict[str, object]) -> str:
    digest = hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()
    return f"r-{digest[:12]}"


def _float_value(data: dict[str, object], key: str, source: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RulesError(f"triage rules ({source}): {key!r} must be a number, got {value!r}")
    return float(value)


def _int_value(data: dict[str, object], key: str, source: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RulesError(f"triage rules ({source}): {key!r} must be an integer, got {value!r}")
    return value


def _str_list_value(data: dict[str, object], key: str, source: str) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RulesError(f"triage rules ({source}): {key!r} must be a list of strings")
    return tuple(value)


def parse_rules(data: object, *, source: str) -> TriageRules:
    """Validate a TOML-decoded mapping into a `TriageRules`.

    Unknown keys, missing keys, and keys of the wrong type are all
    `RulesError`, with a message naming the offending key(s) and `source`
    (the shipped file or the `INFOVORE_TRIAGE_RULES` path) for a clear error.
    """
    if not isinstance(data, dict):
        raise RulesError(f"triage rules ({source}): must be a TOML table")
    keys = set(data)
    unknown = sorted(keys - _ALL_KEYS)
    if unknown:
        raise RulesError(f"triage rules ({source}): unknown key(s): {', '.join(unknown)}")
    missing = sorted(_ALL_KEYS - keys)
    if missing:
        raise RulesError(f"triage rules ({source}): missing key(s): {', '.join(missing)}")

    floats = {key: _float_value(data, key, source) for key in _FLOAT_KEYS}
    ints = {key: _int_value(data, key, source) for key in _INT_KEYS}
    lists: dict[str, tuple[str, ...] | frozenset[str]] = {
        key: _str_list_value(data, key, source) for key in _LIST_KEYS
    }
    lists["agreement_emoji"] = frozenset(lists["agreement_emoji"])

    return TriageRules(version=_version(data), **floats, **ints, **lists)  # type: ignore[arg-type]


def _shipped_text() -> str:
    return resources.files(_SHIPPED_PACKAGE).joinpath(_SHIPPED_RESOURCE).read_text(encoding="utf-8")


def load_rules(path: Path | str | None = None) -> TriageRules:
    """Load and validate triage rules from `path`, or the shipped default.

    Raises `RulesError` if the override file can't be read, isn't valid TOML,
    or fails `parse_rules`' validation.
    """
    if path is None:
        text = _shipped_text()
        source = "shipped rules.toml"
    else:
        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as error:
            raise RulesError(f"INFOVORE_TRIAGE_RULES: cannot read {file_path}: {error}") from error
        source = str(file_path)

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise RulesError(f"triage rules ({source}): invalid TOML: {error}") from error

    return parse_rules(data, source=source)


DEFAULT_RULES: Final[TriageRules] = load_rules()
