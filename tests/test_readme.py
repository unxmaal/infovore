import re
from pathlib import Path

from infovore.db.connection import migrate, open_database

README = Path(__file__).resolve().parent.parent / "README.md"


def section(title: str) -> str:
    text = README.read_text()
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match is not None, f"README has no '## {title}' section"
    return match.group(1)


def test_every_schema_object_is_documented_in_the_data_model_section(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    # FTS5 creates four shadow tables per index (_data, _idx, _docsize,
    # _config). Derive the prefixes from the virtual tables that exist rather
    # than naming each index, or every new index needs this test edited.
    fts_prefixes = tuple(
        f"{row[0]}_"
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE '%USING fts5%'"
        )
    )
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            " AND name NOT LIKE 'sqlite_%'"
        )
        if not row[0].startswith(fts_prefixes)
    }
    documented = section("Data model")
    missing = sorted(name for name in names if f"`{name}`" not in documented)
    assert missing == []


def test_purpose_and_architecture_sections_are_written() -> None:
    assert len(section("Purpose").split()) > 40
    assert len(section("Architecture").split()) > 80


def test_every_builtin_command_is_documented_in_commands() -> None:
    from infovore.cli import builtin_commands

    commands = section("Commands")
    missing = [
        command.name for command in builtin_commands() if f"infovore {command.name}" not in commands
    ]
    assert missing == []


def test_configuration_documents_every_environment_variable() -> None:
    import re as _re

    source = (README.parent / "infovore" / "config.py").read_text()
    names = set(_re.findall(r'"(INFOVORE_[A-Z_]+)"', source)) - {"INFOVORE_TRIAGE_MIN_P_LORE"}
    configuration = section("Configuration")
    assert sorted(name for name in names if name not in configuration) == []


def test_readme_has_the_operator_sections() -> None:
    for title in ("Human judging workflow", "Measuring", "Privacy and opt-out", "Deployment"):
        assert section(title).strip()
