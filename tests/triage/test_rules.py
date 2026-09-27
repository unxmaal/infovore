from pathlib import Path

import pytest

from infovore.triage.rules import DEFAULT_RULES, RulesError, TriageRules, load_rules, parse_rules

VALID_DATA: dict[str, object] = {
    "domain_term_weight": 0.15,
    "domain_term_cap": 0.45,
    "irix_version_weight": 0.2,
    "part_number_weight": 0.3,
    "unix_path_weight": 0.15,
    "code_weight": 0.15,
    "archive_link_weight": 0.15,
    "pdf_attachment_weight": 0.15,
    "answered_question_weight": 0.2,
    "agreed_answer_weight": 0.05,
    "thread_weight": 0.05,
    "substantial_weight": 0.1,
    "tiny_penalty": -0.2,
    "gif_penalty": -0.1,
    "laughter_penalty": -0.1,
    "substantial_characters": 400,
    "answer_min_characters": 40,
    "tiny_message_characters": 20,
    "tiny_share_threshold": 0.7,
    "laughter_share_threshold": 0.3,
    "domain_terms": ["sgi", "irix"],
    "archive_link_hosts": ["bitsavers"],
    "gif_hosts": ["tenor\\.com"],
    "laughter_tokens": ["lol+"],
    "agreement_emoji": ["✅"],
}


def valid_data(**overrides: object) -> dict[str, object]:
    data = dict(VALID_DATA)
    data.update(overrides)
    return data


def write_toml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "rules.toml"
    path.write_text(text)
    return path


# --- parse_rules ------------------------------------------------------------


def test_parse_rules_builds_a_triage_rules() -> None:
    rules = parse_rules(valid_data(), source="test")
    assert isinstance(rules, TriageRules)
    assert rules.domain_term_weight == 0.15
    assert rules.domain_terms == ("sgi", "irix")
    assert rules.agreement_emoji == frozenset({"✅"})


def test_parse_rules_version_is_r_prefixed_hash() -> None:
    rules = parse_rules(valid_data(), source="test")
    assert rules.version.startswith("r-")
    assert len(rules.version) == len("r-") + 12


def test_parse_rules_rejects_non_table() -> None:
    with pytest.raises(RulesError, match="must be a TOML table"):
        parse_rules([1, 2, 3], source="test")


def test_parse_rules_rejects_unknown_key() -> None:
    with pytest.raises(RulesError, match="unknown key.*bogus"):
        parse_rules(valid_data(bogus=1), source="test")


def test_parse_rules_rejects_missing_key() -> None:
    data = valid_data()
    del data["domain_term_weight"]
    with pytest.raises(RulesError, match="missing key.*domain_term_weight"):
        parse_rules(data, source="test")


def test_parse_rules_rejects_wrong_type_float() -> None:
    with pytest.raises(RulesError, match="domain_term_weight.*must be a number"):
        parse_rules(valid_data(domain_term_weight="a lot"), source="test")


def test_parse_rules_rejects_bool_for_float_field() -> None:
    with pytest.raises(RulesError, match="domain_term_weight.*must be a number"):
        parse_rules(valid_data(domain_term_weight=True), source="test")


def test_parse_rules_accepts_int_for_float_field() -> None:
    rules = parse_rules(valid_data(domain_term_weight=1), source="test")
    assert rules.domain_term_weight == 1.0


def test_parse_rules_rejects_wrong_type_int() -> None:
    with pytest.raises(RulesError, match="substantial_characters.*must be an integer"):
        parse_rules(valid_data(substantial_characters=400.0), source="test")


def test_parse_rules_rejects_bool_for_int_field() -> None:
    with pytest.raises(RulesError, match="substantial_characters.*must be an integer"):
        parse_rules(valid_data(substantial_characters=True), source="test")


def test_parse_rules_rejects_wrong_type_list() -> None:
    with pytest.raises(RulesError, match="domain_terms.*must be a list of strings"):
        parse_rules(valid_data(domain_terms="sgi"), source="test")


def test_parse_rules_rejects_non_string_list_items() -> None:
    with pytest.raises(RulesError, match="domain_terms.*must be a list of strings"):
        parse_rules(valid_data(domain_terms=["sgi", 1]), source="test")


def test_parse_rules_error_message_includes_source() -> None:
    with pytest.raises(RulesError, match="my-custom-source"):
        parse_rules(valid_data(bogus=1), source="my-custom-source")


# --- load_rules ---------------------------------------------------------


def test_load_rules_default_matches_shipped_file() -> None:
    rules = load_rules()
    assert isinstance(rules, TriageRules)
    assert rules.domain_term_weight == 0.15
    assert rules.domain_term_cap == 0.45
    assert rules.tiny_penalty == -0.2
    assert "sgi" in rules.domain_terms
    assert rules.agreement_emoji == frozenset({"✅", "👍", "☑️", "✔️", "💯"})


def test_load_rules_none_equals_default_rules() -> None:
    assert load_rules(None) == DEFAULT_RULES


def test_load_rules_override_path_reads_custom_file(tmp_path: Path) -> None:
    text = "\n".join(f"{key} = {_toml_value(value)}" for key, value in VALID_DATA.items())
    path = write_toml(tmp_path, text)
    rules = load_rules(path)
    assert rules.domain_term_weight == 0.15
    assert rules != DEFAULT_RULES


def test_load_rules_override_path_accepts_str() -> None:
    text = "\n".join(f"{key} = {_toml_value(value)}" for key, value in VALID_DATA.items())
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rules.toml"
        path.write_text(text)
        rules = load_rules(str(path))
        assert rules.domain_term_weight == 0.15


def test_load_rules_missing_override_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RulesError, match="cannot read"):
        load_rules(tmp_path / "missing.toml")


def test_load_rules_invalid_toml_raises(tmp_path: Path) -> None:
    path = write_toml(tmp_path, "this is not [valid toml")
    with pytest.raises(RulesError, match="invalid TOML"):
        load_rules(path)


def test_load_rules_invalid_schema_raises(tmp_path: Path) -> None:
    path = write_toml(tmp_path, "domain_term_weight = 0.15\n")
    with pytest.raises(RulesError, match="missing key"):
        load_rules(path)


# --- version derivation ---------------------------------------------------


def test_version_ignores_whitespace_and_comments(tmp_path: Path) -> None:
    lines = [f"{key} = {_toml_value(value)}" for key, value in VALID_DATA.items()]
    plain_path = tmp_path / "plain.toml"
    plain_path.write_text("\n".join(lines))
    # Same values, but with extra blank lines, indentation, and comments.
    decorated_path = tmp_path / "decorated.toml"
    decorated_path.write_text(
        "# a comment\n\n"
        + "\n\n".join(f"  {line}   # trailing comment" for line in lines)
        + "\n\n# trailing comment\n"
    )
    plain_rules = load_rules(plain_path)
    decorated_rules = load_rules(decorated_path)
    assert plain_rules.version == decorated_rules.version


def test_version_changes_when_a_value_changes(tmp_path: Path) -> None:
    lines_a = [f"{key} = {_toml_value(value)}" for key, value in VALID_DATA.items()]
    path_a = write_toml(tmp_path, "\n".join(lines_a))
    changed = valid_data(domain_term_weight=0.99)
    path_b = tmp_path / "changed.toml"
    path_b.write_text("\n".join(f"{key} = {_toml_value(value)}" for key, value in changed.items()))
    assert load_rules(path_a).version != load_rules(path_b).version


def test_default_rules_is_reused() -> None:
    assert load_rules() == DEFAULT_RULES


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise TypeError(value)  # pragma: no cover - test data is well-formed
