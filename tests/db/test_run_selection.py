import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.claims import NewClaim, record_run, register_prompt_version, set_batch_id
from infovore.db.connection import migrate, open_database
from infovore.db.run_selection import (
    InvalidRunSelectorError,
    NoTrialBatchError,
    latest_trial_batch_run_ids,
    parse_run_tokens,
    resolve_run_selector,
)
from infovore.rows import ClaimKind, ExtractionRunRow, RunMode, RunOutcome

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_parse_run_tokens_accepts_bare_ids() -> None:
    assert parse_run_tokens(["5", "7", "3"]) == [3, 5, 7]


def test_parse_run_tokens_expands_an_inclusive_range() -> None:
    assert parse_run_tokens(["220-224"]) == [220, 221, 222, 223, 224]


def test_parse_run_tokens_mixes_ids_and_ranges_deduped_and_sorted() -> None:
    assert parse_run_tokens(["10-12", "5", "11"]) == [5, 10, 11, 12]


def test_parse_run_tokens_rejects_a_backwards_range() -> None:
    with pytest.raises(InvalidRunSelectorError):
        parse_run_tokens(["9-3"])


def test_parse_run_tokens_rejects_garbage() -> None:
    with pytest.raises(InvalidRunSelectorError) as excinfo:
        parse_run_tokens(["abc"])
    assert "abc" in str(excinfo.value)


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def seed_run(conn: sqlite3.Connection, exchange_id: int, mode: RunMode = RunMode.TRIAL) -> int:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, 1, 9, 1, 'a', ?, 'x', ?, '{}')",
        (exchange_id, NOW.isoformat(), NOW.isoformat()),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, 1, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
        (
            exchange_id,
            exchange_id,
            exchange_id,
            NOW.isoformat(),
            NOW.isoformat(),
            f"h{exchange_id}",
        ),
    )
    register_prompt_version(conn, "v1", "sha", NOW)
    recorded = record_run(
        conn,
        ExtractionRunRow(
            id=None,
            exchange_id=exchange_id,
            model="m",
            prompt_version="v1",
            started_at=NOW,
            finished_at=NOW,
            input_tokens=None,
            output_tokens=None,
            mode=mode,
            outcome=RunOutcome.OK,
            error=None,
        ),
        [
            NewClaim(
                exchange_id=exchange_id,
                statement="s",
                subject="subj",
                kind=ClaimKind.FACT,
                confidence=0.9,
                probe_question="q?",
                permalink="https://discord.com/channels/1/1/1",
                supersedes_claim_id=None,
                source_message_ids=(exchange_id,),
            )
        ],
    )
    return recorded.run_id


def test_latest_trial_batch_run_ids_raises_when_nothing_has_a_batch(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_run(conn, exchange_id=1)
    with pytest.raises(NoTrialBatchError):
        latest_trial_batch_run_ids(conn)


def test_latest_trial_batch_run_ids_returns_only_the_newest_batch(tmp_path: Path) -> None:
    conn = db(tmp_path)
    old_run = seed_run(conn, exchange_id=1)
    new_run = seed_run(conn, exchange_id=2)
    set_batch_id(conn, [old_run], "2026-01-01T00:00:00+00:00")
    set_batch_id(conn, [new_run], "2026-01-02T00:00:00+00:00")

    assert latest_trial_batch_run_ids(conn) == [new_run]


def test_latest_trial_batch_run_ids_ignores_live_mode_runs(tmp_path: Path) -> None:
    conn = db(tmp_path)
    live_run = seed_run(conn, exchange_id=1, mode=RunMode.LIVE)
    trial_run = seed_run(conn, exchange_id=2, mode=RunMode.TRIAL)
    set_batch_id(conn, [live_run], "2026-01-02T00:00:00+00:00")
    set_batch_id(conn, [trial_run], "2026-01-01T00:00:00+00:00")

    assert latest_trial_batch_run_ids(conn) == [trial_run]


def test_resolve_run_selector_returns_latest_batch_when_tokens_is_none(tmp_path: Path) -> None:
    conn = db(tmp_path)
    run_id = seed_run(conn, exchange_id=1)
    set_batch_id(conn, [run_id], "2026-01-01T00:00:00+00:00")

    assert resolve_run_selector(conn, None) == [run_id]


def test_resolve_run_selector_returns_latest_batch_when_tokens_is_empty(tmp_path: Path) -> None:
    conn = db(tmp_path)
    run_id = seed_run(conn, exchange_id=1)
    set_batch_id(conn, [run_id], "2026-01-01T00:00:00+00:00")

    assert resolve_run_selector(conn, []) == [run_id]


def test_resolve_run_selector_parses_explicit_tokens(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert resolve_run_selector(conn, ["5", "10-12"]) == [5, 10, 11, 12]
