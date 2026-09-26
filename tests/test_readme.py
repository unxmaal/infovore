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
        command.name for command in builtin_commands() if f"infovore {command.name}" not in running
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


def test_runbook_leads_with_a_one_time_export_and_documents_it_as_the_only_api_call() -> None:
    runbook = section("Running").split("### Runbook", 1)[1]
    for needle in (
        "fetch from the Discord API once",
        "DiscordChatExporter",
        "exportguild",
        "--include-threads all",
        "INFOVORE_SOURCE=export",
        "INFOVORE_EXPORT_DIR",
    ):
        assert needle in runbook, needle
    assert runbook.index("Export the server with DiscordChatExporter") < runbook.index("**2.")


def test_configuration_documents_export_source() -> None:
    configuration = section("Configuration")
    for needle in ("INFOVORE_SOURCE", "INFOVORE_EXPORT_DIR", "ExportDiscordSource"):
        assert needle in configuration, needle


def test_configuration_documents_the_shared_channel_allowlist() -> None:
    configuration = section("Configuration")
    running = section("Running")
    for needle in ("is_channel_allowed", "events_ignored"):
        assert needle in configuration or needle in running, needle
    assert "optional" in configuration


def test_triage_section_documents_every_signal_and_the_version() -> None:
    import inspect

    from infovore.triage import score

    triage = section("Triage")
    names = set(re.findall(r'reasons\.append\(\("(\w+)"', inspect.getsource(score)))
    names |= set(re.findall(r'\("(\w+)", [A-Z_]+, [A-Z_]+_WEIGHT\)', inspect.getsource(score)))
    assert names
    assert sorted(name for name in names if f"`{name}`" not in triage) == []
    assert score.TRIAGE_VERSION in triage
