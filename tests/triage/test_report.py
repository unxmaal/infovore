import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import insert_exchange
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule, MessageRow
from infovore.triage.report import compute_triage_stats

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def a_message(message_id: int, channel_id: int) -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=channel_id,
        guild_id=9,
        author_id=1,
        author_name_at_time="alice",
        author_is_bot=False,
        created_at=NOW,
        edited_at=None,
        content="x",
        reply_to_id=None,
        thread_id=None,
        deleted_at=None,
        ingested_at=NOW,
        raw_json="{}",
    )


def seed_scored_exchange(
    conn: sqlite3.Connection,
    message_id: int,
    channel_id: int,
    score: float,
    reasons: str,
    version: str = "t1",
) -> None:
    message = a_message(message_id, channel_id)
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            message.id,
            message.channel_id,
            message.guild_id,
            message.author_id,
            message.author_name_at_time,
            int(message.author_is_bot),
            message.created_at.isoformat(),
            message.content,
            message.ingested_at.isoformat(),
            message.raw_json,
        ),
    )
    row = ExchangeRow(
        id=None,
        channel_id=channel_id,
        thread_id=None,
        first_message_id=message_id,
        last_message_id=message_id,
        started_at=NOW,
        ended_at=NOW,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=f"c{message_id}",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    exchange_id = insert_exchange(conn, row, [message_id])
    conn.execute(
        "UPDATE exchanges SET triage_score = ?, triage_reasons = ?, triage_version = ?"
        " WHERE id = ?",
        (score, reasons, version, exchange_id),
    )


def test_histogram_buckets_scores_in_tenths(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_scored_exchange(conn, 1, 1, 0.05, "[]")
    seed_scored_exchange(conn, 2, 1, 0.15, "[]")
    seed_scored_exchange(conn, 3, 1, 1.0, "[]")

    stats = compute_triage_stats(conn, min_score=0.3)

    assert stats.histogram["0.0-0.1"] == 1
    assert stats.histogram["0.1-0.2"] == 1
    assert stats.histogram["0.9-1.0"] == 1


def test_channel_stats_report_mean_and_count(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_scored_exchange(conn, 1, 1, 0.2, "[]")
    seed_scored_exchange(conn, 2, 1, 0.4, "[]")
    seed_scored_exchange(conn, 3, 2, 0.9, "[]")

    stats = compute_triage_stats(conn, min_score=0.3)

    mean1, count1 = stats.channel_stats[1]
    assert mean1 == pytest.approx(0.3)
    assert count1 == 2
    mean2, count2 = stats.channel_stats[2]
    assert mean2 == 0.9
    assert count2 == 1


def test_above_and_below_threshold_counts(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_scored_exchange(conn, 1, 1, 0.5, "[]")
    seed_scored_exchange(conn, 2, 1, 0.2, "[]")
    seed_scored_exchange(conn, 3, 1, 0.3, "[]")

    stats = compute_triage_stats(conn, min_score=0.3)

    assert stats.above_threshold == 2
    assert stats.below_threshold == 1


def test_top_reasons_ordered_by_frequency_then_name(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_scored_exchange(conn, 1, 1, 0.5, '[["code", 0.15], ["thread", 0.05]]')
    seed_scored_exchange(conn, 2, 1, 0.5, '[["code", 0.15]]')
    seed_scored_exchange(conn, 3, 1, 0.5, '[["archive_link", 0.15]]')

    stats = compute_triage_stats(conn, min_score=0.3)

    assert stats.top_reasons[0] == ("code", 2)
    names = [name for name, _ in stats.top_reasons]
    assert set(names) == {"code", "thread", "archive_link"}


def test_top_reasons_limited_to_ten(tmp_path: Path) -> None:
    conn = db(tmp_path)
    for i in range(15):
        seed_scored_exchange(conn, i + 1, 1, 0.5, f'[["reason{i}", 0.1]]')

    stats = compute_triage_stats(conn, min_score=0.3)

    assert len(stats.top_reasons) == 10


def test_untriaged_exchanges_excluded_from_stats(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_scored_exchange(conn, 1, 1, 0.5, "[]", version="t0")

    stats = compute_triage_stats(conn, min_score=0.3)

    assert stats.above_threshold == 0
    assert stats.below_threshold == 0
    assert stats.channel_stats == {}


def test_empty_database_reports_zeros(tmp_path: Path) -> None:
    conn = db(tmp_path)

    stats = compute_triage_stats(conn, min_score=0.3)

    assert stats.histogram == {}
    assert stats.channel_stats == {}
    assert stats.above_threshold == 0
    assert stats.below_threshold == 0
    assert stats.top_reasons == []
