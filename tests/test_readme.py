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


def test_every_builtin_command_is_documented_in_running() -> None:
    from infovore.cli import builtin_commands

    running = section("Running")
    missing = [
        command.name
        for command in builtin_commands()
        if f"infovore {command.name}" not in running
    ]
    assert missing == []


def test_runbook_covers_first_run_to_steady_state() -> None:
    runbook = section("Running").split("### Runbook", 1)
    assert len(runbook) == 2, "README 'Running' needs a '### Runbook' subsection"
    body = runbook[1]
    for needle in (
        "Message Content",
        "Server Members",
        "no-archive",
        "Server notice",
        "infovore sync-optouts",
        "infovore backfill",
        "infovore chunk",
        "infovore extract --mode trial",
        "infovore probe",
        "infovore review",
        "infovore promote",
        "infovore run",
        "infovore snapshot",
    ):
        assert needle in body, needle
