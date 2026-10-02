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


def test_configuration_documents_the_channel_denylist() -> None:
    configuration = section("Configuration")
    running = section("Running")
    for needle in ("INFOVORE_EXCLUDE_CHANNELS", "claimable_exchanges"):
        assert needle in configuration, needle
    assert "INFOVORE_EXCLUDE_CHANNELS" in running


def test_configuration_documents_the_shared_channel_allowlist() -> None:
    configuration = section("Configuration")
    running = section("Running")
    for needle in ("is_channel_allowed", "events_ignored"):
        assert needle in configuration or needle in running, needle
    assert "optional" in configuration


def test_running_section_documents_the_shared_run_selector() -> None:
    running = section("Running")
    for needle in (
        "infovore.db.run_selection",
        "latest trial batch",
        "batch_id",
        "220-419",
    ):
        assert needle in running, needle


def test_tuning_loop_runbook_exists_and_covers_the_full_cycle() -> None:
    triage = section("Triage")
    parts = triage.split("### Tuning loop", 1)
    assert len(parts) == 2, "README 'Triage' needs a '### Tuning loop' subsection"
    body = parts[1]
    for needle in (
        "infovore extract --mode trial",
        "infovore probe",
        "infovore label --from-runs",
        "infovore triage --train",
        "infovore triage --signal-report",
        "infovore triage --suggest-terms",
        "INFOVORE_TRIAGE_RULES",
        "infovore triage --report",
        "human",
    ):
        assert needle in body, needle


def test_triage_section_documents_the_tuning_flags() -> None:
    triage = section("Triage")
    for needle in (
        "--signal-report",
        "--suggest-terms",
        "--min-support",
        "--max-corpus-df",
        "--fit-weights",
        "--out",
        "--force",
        "infovore/triage/logistic.py",
        "infovore.triage.tuning",
        "useless",
        "harmful",
        "class_imbalance",
        "BAYES_00",
    ):
        assert needle in triage, needle


def test_running_section_documents_strategy_mixed_and_the_batches_table() -> None:
    running = section("Running")
    for needle in ("mixed", "--mix", "extraction_batches", "sampled_by"):
        assert needle in running, needle


def test_triage_section_documents_the_uncertain_sampling_warning() -> None:
    triage = section("Triage")
    for needle in ("sampling_bias_warning", "sampled_by", "--strategy mixed"):
        assert needle in triage, needle


def test_sifting_section_documents_citations_train_and_report() -> None:
    running = section("Running")
    sifting = running.split("### Sifting", 1)
    assert len(sifting) == 2, "README 'Running' needs a '### Sifting' subsection"
    body = sifting[1]
    for needle in (
        "infovore sift citations",
        "infovore sift train",
        "message_model",
        "message_tokens",
        "message_combiner",
        "--human-weight",
        "citation model",
        "human model",
        "combiner",
        "fallback",
        "out-of-fold",
        "p_trash",
        "discard",
        "channel name",
        "AUC",
    ):
        assert needle in body, needle


def test_sifting_section_documents_context_features_and_ablation() -> None:
    running = section("Running")
    sifting = running.split("### Sifting", 1)[1]
    for needle in (
        "conversation context",
        "PREV_",
        "NEXT_",
        "REPLYTO_",
        "FEATURE_SET_VERSION",
        "feature_set_version",
        "FeatureSetMismatchError",
        "ablation",
        "exchange_context_tokens",
    ):
        assert needle in sifting, needle


def test_sifting_section_documents_the_three_feature_sets() -> None:
    running = section("Running")
    sifting = running.split("### Sifting", 1)[1]
    for needle in (
        "--features",
        "feature_set_name",
        "DEFAULT_FEATURE_SET",
        "structural",
        "FeatureSet",
    ):
        assert needle in sifting, needle


def test_sifting_section_documents_the_channels_filter() -> None:
    running = section("Running")
    sifting = running.split("### Sifting", 1)[1]
    for needle in ("--channels", "INFOVORE_EXCLUDE_CHANNELS"):
        assert needle in sifting, needle


def test_tuning_loop_recommends_mixed_rounds() -> None:
    triage = section("Triage")
    body = triage.split("### Tuning loop", 1)[1]
    assert "mixed" in body


def test_triage_section_documents_every_signal_and_the_version() -> None:
    import inspect

    from infovore.triage import score

    triage = section("Triage")
    names = set(re.findall(r'reasons\.append\(\("(\w+)"', inspect.getsource(score)))
    names |= set(re.findall(r'\("(\w+)", [\w.]+, rules\.\w+_weight\)', inspect.getsource(score)))
    assert names
    assert sorted(name for name in names if f"`{name}`" not in triage) == []
    assert "TRIAGE_VERSION" in triage
    assert re.search(r"r-[0-9a-f]{12}", triage) is not None
    assert "INFOVORE_TRIAGE_RULES" in triage
