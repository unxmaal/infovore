import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.connection import migrate, open_database
from infovore.db.status import collect_status
from infovore.triage.rules import DEFAULT_RULES
from infovore.triage.score import TRIAGE_VERSION

NOW = "2026-01-01T00:00:00+00:00"
LATER = "2026-01-02T00:00:00+00:00"


def fresh(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def test_fresh_database_reports_zeros(tmp_path: Path) -> None:
    report = collect_status(fresh(tmp_path))
    assert report.channels == 0
    assert report.messages == 0
    assert report.deleted_messages == 0
    assert report.exchanges_by_status == {}
    assert report.claims_by_novelty == {}
    assert report.retracted_claims == 0
    assert report.runs_by_outcome == {}
    assert report.last_extraction_at is None
    assert report.last_probe_at is None
    assert report.live_prompt_version is None
    assert report.triaged_exchanges == 0
    assert report.above_threshold_exchanges == 0
    assert report.labels_by_source == {}
    assert report.labels_effective == {}
    assert report.latest_model_version is None
    assert report.latest_model_labels_used is None
    assert report.p_lore_scored == 0
    assert report.passing_gate == 0
    assert report.excluded_by_denylist == 0


def test_triaged_and_above_threshold_counts(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    conn.executescript(
        f"""
        INSERT INTO channels (id, guild_id, name, kind) VALUES (1, 9, 'general', 'text');
        INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,
          created_at, content, ingested_at, raw_json)
          VALUES (1, 1, 9, 1, 'a', '{NOW}', 'x', '{NOW}', '{{}}'),
                 (2, 1, 9, 1, 'a', '{NOW}', 'y', '{NOW}', '{{}}'),
                 (3, 1, 9, 1, 'a', '{NOW}', 'z', '{NOW}', '{{}}');
        INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,
          ended_at, message_count, grouping_rule, content_hash, triage_score, triage_version)
          VALUES (1, 1, 1, '{NOW}', '{NOW}', 1, 'quiet_gap', 'a', 0.9, '{TRIAGE_VERSION}'),
                 (1, 2, 2, '{NOW}', '{NOW}', 1, 'quiet_gap', 'b', 0.1, '{TRIAGE_VERSION}'),
                 (1, 3, 3, '{NOW}', '{NOW}', 1, 'quiet_gap', 'c', NULL, NULL);
        """
    )
    report = collect_status(conn, triage_min_score=0.3)
    assert report.triaged_exchanges == 2
    assert report.above_threshold_exchanges == 1


def test_triaged_counts_use_the_provided_rules_version(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    custom = replace(DEFAULT_RULES, version="custom-status")
    conn.executescript(
        f"""
        INSERT INTO channels (id, guild_id, name, kind) VALUES (1, 9, 'general', 'text');
        INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,
          created_at, content, ingested_at, raw_json)
          VALUES (1, 1, 9, 1, 'a', '{NOW}', 'x', '{NOW}', '{{}}');
        INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,
          ended_at, message_count, grouping_rule, content_hash, triage_score, triage_version)
          VALUES (1, 1, 1, '{NOW}', '{NOW}', 1, 'quiet_gap', 'a', 0.9, 'custom-status');
        """
    )
    default_report = collect_status(conn, triage_min_score=0.3)
    custom_report = collect_status(conn, triage_min_score=0.3, rules=custom)
    assert default_report.triaged_exchanges == 0
    assert custom_report.triaged_exchanges == 1


def test_counts_and_last_runs(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    conn.executescript(
        f"""
        INSERT INTO channels (id, guild_id, name, kind) VALUES (1, 9, 'general', 'text');
        INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,
          created_at, content, ingested_at, raw_json, deleted_at)
          VALUES (1, 1, 9, 1, 'a', '{NOW}', 'x', '{NOW}', '{{}}', NULL),
                 (2, 1, 9, 1, 'a', '{NOW}', 'y', '{NOW}', '{{}}', '{NOW}');
        INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,
          ended_at, message_count, grouping_rule, content_hash, extraction_status)
          VALUES (1, 1, 1, '{NOW}', '{NOW}', 1, 'quiet_gap', 'a', 'pending'),
                 (1, 2, 2, '{NOW}', '{NOW}', 1, 'quiet_gap', 'b', 'done');
        INSERT INTO prompt_versions VALUES ('v1', 'sha', '{NOW}', '{NOW}');
        INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at, mode, outcome)
          VALUES (2, 'm', 'v1', '{NOW}', 'live', 'ok'), (1, 'm', 'v1', '{LATER}', 'live', 'failed');
        INSERT INTO claims (exchange_id, extraction_run_id, statement, subject, kind, confidence,
          probe_question, permalink, novelty, probed_at, retracted_at)
          VALUES (2, 1, 's', 'subj', 'fact', 0.5, 'q?', 'p', 'unknown', '{LATER}', NULL),
                 (2, 1, 't', 'subj', 'fact', 0.5, 'q?', 'p', 'unprobed', NULL, '{NOW}');
        INSERT INTO exchange_labels (exchange_id, label, source, labeled_at)
          VALUES (1, 'lore', 'llm', '{NOW}'),
                 (1, 'noise', 'human', '{LATER}'),
                 (2, 'noise', 'human', '{NOW}');
        """
    )
    report = collect_status(conn)
    assert report.channels == 1
    assert report.messages == 2
    assert report.deleted_messages == 1
    assert report.exchanges_by_status == {"pending": 1, "done": 1}
    assert report.claims_by_novelty == {"unknown": 1}
    assert report.retracted_claims == 1
    assert report.runs_by_outcome == {"ok": 1, "failed": 1}
    assert report.last_extraction_at == datetime(2026, 1, 2, tzinfo=UTC)
    assert report.last_probe_at == datetime(2026, 1, 2, tzinfo=UTC)
    assert report.live_prompt_version == "v1"
    assert report.labels_by_source == {"llm": {"lore": 1}, "human": {"noise": 2}}
    assert report.labels_effective == {"noise": 2}


def test_reports_latest_model_p_lore_scored_and_passing_gate(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    conn.executescript(
        f"""
        INSERT INTO channels (id, guild_id, name, kind) VALUES (1, 9, 'general', 'text');
        INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,
          created_at, content, ingested_at, raw_json)
          VALUES (1, 1, 9, 1, 'a', '{NOW}', 'x', '{NOW}', '{{}}'),
                 (2, 1, 9, 1, 'a', '{NOW}', 'y', '{NOW}', '{{}}');
        INSERT INTO triage_model (trained_at, labels_used, holdout_size, params_json)
          VALUES ('{NOW}', 40, 8, '{{}}');
        INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,
          ended_at, message_count, grouping_rule, content_hash, triage_score, triage_version,
          p_lore, p_lore_model)
          VALUES (1, 1, 1, '{NOW}', '{NOW}', 1, 'quiet_gap', 'a', 0.0, '{TRIAGE_VERSION}', 0.9, 1),
                 (1, 2, 2, '{NOW}', '{NOW}', 1, 'quiet_gap', 'b', 1.0, '{TRIAGE_VERSION}', NULL,
                  NULL);
        """
    )
    report = collect_status(conn, triage_min_score=0.3, triage_min_p_lore=0.5)
    assert report.latest_model_version == 1
    assert report.latest_model_labels_used == 40
    assert report.p_lore_scored == 1
    assert report.passing_gate == 2


def test_excluded_by_denylist_counts_gate_passing_exchanges_in_denylisted_channels(
    tmp_path: Path,
) -> None:
    conn = fresh(tmp_path)
    conn.executescript(
        f"""
        INSERT INTO channels (id, guild_id, name, kind) VALUES
          (1, 9, 'general', 'text'), (2, 9, 'food', 'text');
        INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,
          created_at, content, ingested_at, raw_json)
          VALUES (1, 1, 9, 1, 'a', '{NOW}', 'x', '{NOW}', '{{}}'),
                 (2, 2, 9, 1, 'a', '{NOW}', 'y', '{NOW}', '{{}}'),
                 (3, 2, 9, 1, 'a', '{NOW}', 'z', '{NOW}', '{{}}');
        INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,
          ended_at, message_count, grouping_rule, content_hash, triage_score, triage_version)
          VALUES (1, 1, 1, '{NOW}', '{NOW}', 1, 'quiet_gap', 'a', 0.9, '{TRIAGE_VERSION}'),
                 (2, 2, 2, '{NOW}', '{NOW}', 1, 'quiet_gap', 'b', 0.9, '{TRIAGE_VERSION}'),
                 (2, 3, 3, '{NOW}', '{NOW}', 1, 'quiet_gap', 'c', 0.0, '{TRIAGE_VERSION}');
        """
    )
    report = collect_status(
        conn, triage_min_score=0.3, exclude_channels=frozenset({"food"})
    )
    assert report.passing_gate == 2
    assert report.excluded_by_denylist == 1


def test_excluded_by_denylist_is_zero_without_a_denylist(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    conn.executescript(
        f"""
        INSERT INTO channels (id, guild_id, name, kind) VALUES (1, 9, 'general', 'text');
        INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,
          created_at, content, ingested_at, raw_json)
          VALUES (1, 1, 9, 1, 'a', '{NOW}', 'x', '{NOW}', '{{}}');
        INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,
          ended_at, message_count, grouping_rule, content_hash, triage_score, triage_version)
          VALUES (1, 1, 1, '{NOW}', '{NOW}', 1, 'quiet_gap', 'a', 0.9, '{TRIAGE_VERSION}');
        """
    )
    report = collect_status(conn, triage_min_score=0.3)
    assert report.excluded_by_denylist == 0
