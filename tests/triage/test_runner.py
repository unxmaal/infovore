import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import get_exchange, insert_exchange
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule, MessageRow
from infovore.triage.rules import DEFAULT_RULES
from infovore.triage.runner import (
    CHANNEL_PRIOR_CAP,
    CHANNEL_PRIOR_WEIGHT,
    TriageExchangeScored,
    TriagePriorsApplied,
    TriageStarted,
    triage_pending,
)
from infovore.triage.score import TRIAGE_VERSION, score_exchange

NOW = datetime(2026, 1, 1, tzinfo=UTC)

ZERO_CONTENT = "just chatting, nothing to see"
FULL_CONTENT_A = "Octane Fuel Tezro Onyx Origin IP30 hinv PROM 6.5.22 060-0035-003 /usr/sbin/inst"
FULL_CONTENT_B = "```\nnvram netaddr\n``` https://bitsavers.org/pdf/sgi/x.pdf " * 20
HALF_CONTENT = "Octane needs 6.5.22, check /usr/var/log"


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def a_message(message_id: int, content: str, channel_id: int, author_id: int = 1) -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=channel_id,
        guild_id=9,
        author_id=author_id,
        author_name_at_time="alice",
        author_is_bot=False,
        created_at=NOW,
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        deleted_at=None,
        ingested_at=NOW,
        raw_json="{}",
    )


def seed_exchange(
    conn: sqlite3.Connection, messages: list[MessageRow], content_hash: str
) -> ExchangeRow:
    for message in messages:
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
        channel_id=messages[0].channel_id,
        thread_id=None,
        first_message_id=messages[0].id,
        last_message_id=messages[-1].id,
        started_at=NOW,
        ended_at=NOW,
        message_count=len(messages),
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=content_hash,
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    exchange_id = insert_exchange(conn, row, [m.id for m in messages])
    exchange = get_exchange(conn, exchange_id)
    assert exchange is not None
    return exchange


def zero_exchange(conn: sqlite3.Connection, message_id: int, channel_id: int) -> ExchangeRow:
    return seed_exchange(conn, [a_message(message_id, ZERO_CONTENT, channel_id)], f"z{message_id}")


def full_exchange(conn: sqlite3.Connection, message_id: int, channel_id: int) -> ExchangeRow:
    messages = [
        a_message(message_id, FULL_CONTENT_A, channel_id),
        a_message(message_id + 1, FULL_CONTENT_B, channel_id, author_id=2),
    ]
    return seed_exchange(conn, messages, f"f{message_id}")


def half_exchange(conn: sqlite3.Connection, message_id: int, channel_id: int) -> ExchangeRow:
    return seed_exchange(conn, [a_message(message_id, HALF_CONTENT, channel_id)], f"h{message_id}")


def clamp01(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 4)


def clipped_delta(diff: float) -> float:
    delta = CHANNEL_PRIOR_WEIGHT * diff
    return max(-CHANNEL_PRIOR_CAP, min(CHANNEL_PRIOR_CAP, delta))


def test_zero_content_scores_zero_raw() -> None:
    assert score_exchange([a_message(1, ZERO_CONTENT, 1)]).score == 0.0


def test_full_content_scores_one_raw() -> None:
    messages = [a_message(1, FULL_CONTENT_A, 1), a_message(2, FULL_CONTENT_B, 1, author_id=2)]
    assert score_exchange(messages).score == 1.0


def test_half_content_scores_point_five_raw() -> None:
    assert score_exchange([a_message(1, HALF_CONTENT, 1)]).score == 0.5


def test_triage_pending_scores_every_candidate_and_sets_version(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = zero_exchange(conn, 1, channel_id=1)
    assert exchange.id is not None

    report = triage_pending(conn)

    assert report.candidates == 1
    assert report.scored == 1
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.triage_version == TRIAGE_VERSION
    assert updated.triage_score == 0.0
    assert updated.triage_reasons is not None


def test_triage_pending_is_idempotent_when_version_unchanged(tmp_path: Path) -> None:
    conn = db(tmp_path)
    zero_exchange(conn, 1, channel_id=1)
    half_exchange(conn, 2, channel_id=2)

    first = triage_pending(conn)
    before = [
        (row["id"], row["triage_score"], row["triage_reasons"], row["triage_version"])
        for row in conn.execute(
            "SELECT id, triage_score, triage_reasons, triage_version FROM exchanges ORDER BY id"
        )
    ]

    second = triage_pending(conn)
    after = [
        (row["id"], row["triage_score"], row["triage_reasons"], row["triage_version"])
        for row in conn.execute(
            "SELECT id, triage_score, triage_reasons, triage_version FROM exchanges ORDER BY id"
        )
    ]

    assert second.candidates == 0
    assert second.scored == 0
    assert before == after
    assert first.channel_means == second.channel_means
    assert first.channel_priors == second.channel_priors


def test_triage_pending_rescans_after_version_bump(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = zero_exchange(conn, 1, channel_id=1)
    assert exchange.id is not None
    triage_pending(conn)

    bumped_rules = replace(DEFAULT_RULES, version="t2")
    report = triage_pending(conn, rules=bumped_rules)

    assert report.candidates == 1
    assert report.scored == 1
    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.triage_version == "t2"


def test_triage_pending_defaults_to_default_rules(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = zero_exchange(conn, 1, channel_id=1)
    assert exchange.id is not None

    triage_pending(conn)

    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.triage_version == DEFAULT_RULES.version


def test_triage_pending_scores_with_the_provided_rules(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = half_exchange(conn, 1, channel_id=1)
    assert exchange.id is not None

    boosted = replace(DEFAULT_RULES, domain_term_weight=1.0, version="boosted")
    triage_pending(conn, rules=boosted)

    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.triage_version == "boosted"
    assert updated.triage_score == 0.8


def test_channel_prior_shifts_score_toward_channel_mean(tmp_path: Path) -> None:
    conn = db(tmp_path)
    high_a = half_exchange(conn, 1, channel_id=1)
    high_b = zero_exchange(conn, 2, channel_id=1)
    low_a = zero_exchange(conn, 3, channel_id=2)
    low_b = zero_exchange(conn, 4, channel_id=2)
    assert high_a.id is not None and high_b.id is not None
    assert low_a.id is not None and low_b.id is not None

    triage_pending(conn)

    channel1_raw = (0.5 + 0.0) / 2
    channel2_raw = 0.0
    global_mean = (0.5 + 0.0 + 0.0 + 0.0) / 4
    delta1 = clipped_delta(channel1_raw - global_mean)
    delta2 = clipped_delta(channel2_raw - global_mean)
    assert abs(delta1) < CHANNEL_PRIOR_CAP
    assert abs(delta2) < CHANNEL_PRIOR_CAP

    half_updated = get_exchange(conn, high_a.id)
    assert half_updated is not None
    assert half_updated.triage_score == pytest.approx(clamp01(0.5 + delta1))
    zero_in_channel1 = get_exchange(conn, high_b.id)
    assert zero_in_channel1 is not None
    assert zero_in_channel1.triage_score == pytest.approx(clamp01(0.0 + delta1))
    zero_in_channel2 = get_exchange(conn, low_a.id)
    assert zero_in_channel2 is not None
    assert zero_in_channel2.triage_score == pytest.approx(clamp01(0.0 + delta2))


def test_channel_prior_is_capped_for_large_gaps(tmp_path: Path) -> None:
    conn = db(tmp_path)
    high = full_exchange(conn, 1, channel_id=1)
    low1 = zero_exchange(conn, 3, channel_id=2)
    zero_exchange(conn, 4, channel_id=2)
    zero_exchange(conn, 5, channel_id=2)
    assert high.id is not None
    assert low1.id is not None

    report = triage_pending(conn)

    global_mean = 1.0 / 4
    expected_delta_high = clipped_delta(1.0 - global_mean)
    expected_delta_low = clipped_delta(0.0 - global_mean)
    assert expected_delta_high == CHANNEL_PRIOR_CAP
    assert abs(expected_delta_low) < CHANNEL_PRIOR_CAP
    assert report.channel_priors[1] == pytest.approx(CHANNEL_PRIOR_CAP)
    assert report.channel_priors[2] == pytest.approx(expected_delta_low)
    updated_high = get_exchange(conn, high.id)
    assert updated_high is not None
    assert updated_high.triage_score == 1.0
    updated_low = get_exchange(conn, low1.id)
    assert updated_low is not None
    assert updated_low.triage_score == pytest.approx(clamp01(0.0 + expected_delta_low))


def test_channel_prior_reason_recorded_in_triage_reasons(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = zero_exchange(conn, 1, channel_id=1)
    assert exchange.id is not None

    triage_pending(conn)

    updated = get_exchange(conn, exchange.id)
    assert updated is not None
    assert updated.triage_reasons is not None
    reasons = dict(json.loads(updated.triage_reasons))
    assert "channel_prior" in reasons


def test_triage_pending_streams_progress_events(tmp_path: Path) -> None:
    conn = db(tmp_path)
    zero_exchange(conn, 1, channel_id=1)
    zero_exchange(conn, 2, channel_id=2)
    events: list[object] = []

    triage_pending(conn, progress=events.append)

    assert isinstance(events[0], TriageStarted)
    assert events[0].total == 2
    scored_events = [event for event in events if isinstance(event, TriageExchangeScored)]
    assert len(scored_events) == 2
    assert [event.index for event in scored_events] == [1, 2]
    assert isinstance(events[-1], TriagePriorsApplied)


def test_triage_pending_empty_database(tmp_path: Path) -> None:
    conn = db(tmp_path)

    report = triage_pending(conn)

    assert report.candidates == 0
    assert report.scored == 0
    assert report.channel_means == {}
    assert report.channel_priors == {}
    assert report.global_mean == 0.0


def _insert_stub_model(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        "INSERT INTO triage_model (trained_at, labels_used, holdout_size, params_json)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, ?)",
        (json.dumps({"lore_documents": 10, "noise_documents": 10}),),
    )
    version = int(cursor.lastrowid or 0)
    conn.execute(
        "INSERT INTO triage_tokens (model_version, token, lore_count, noise_count)"
        " VALUES (?, 'octane', 10, 0)",
        (version,),
    )
    return version


def test_triage_pending_leaves_p_lore_null_without_a_trained_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = zero_exchange(conn, 1, channel_id=1)
    assert exchange.id is not None

    triage_pending(conn)

    row = conn.execute(
        "SELECT p_lore, p_lore_model FROM exchanges WHERE id = ?", (exchange.id,)
    ).fetchone()
    assert (row["p_lore"], row["p_lore_model"]) == (None, None)


def test_triage_pending_scores_p_lore_once_a_model_exists(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = full_exchange(conn, 1, channel_id=1)
    assert exchange.id is not None
    version = _insert_stub_model(conn)

    triage_pending(conn)

    row = conn.execute(
        "SELECT p_lore, p_lore_model FROM exchanges WHERE id = ?", (exchange.id,)
    ).fetchone()
    assert row["p_lore"] is not None
    assert row["p_lore_model"] == version


def test_triage_pending_only_rescopes_exchanges_behind_the_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange = full_exchange(conn, 1, channel_id=1)
    assert exchange.id is not None
    version = _insert_stub_model(conn)

    triage_pending(conn)
    conn.execute("UPDATE exchanges SET p_lore = -1.0 WHERE id = ?", (exchange.id,))

    triage_pending(conn)

    row = conn.execute("SELECT p_lore FROM exchanges WHERE id = ?", (exchange.id,)).fetchone()
    assert row["p_lore"] == -1.0
    assert version is not None
