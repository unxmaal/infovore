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
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            " AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'claims_fts_%'"
        )
    }
    documented = section("Data model")
    missing = sorted(name for name in names if f"`{name}`" not in documented)
    assert missing == []


def test_purpose_and_architecture_sections_are_written() -> None:
    assert len(section("Purpose").split()) > 40
    assert len(section("Architecture").split()) > 80
