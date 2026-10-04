import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import claimable_exchanges
from infovore.db.status import StatusReport, collect_status
from tests.cascade_marks import mark

NOW = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
MAX_RETRIES = 3


def fresh(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def iso(at: datetime) -> str:
    return at.isoformat()


def add_exchange(
    conn: sqlite3.Connection,
    exchange_id: int,
    *,
    status: str = "pending",
    cascade: str | None = "residue",
    retry_count: int = 0,
    channel_id: int = 1,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO channels (id, guild_id, name, kind) VALUES (?, 1, ?, 'text')",
        (channel_id, f"channel-{channel_id}"),
    )
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 1, 1, 'a', ?, 'x', ?, '{}')",
        (exchange_id, channel_id, iso(NOW), iso(NOW)),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash, extraction_status, retry_count)"
        " VALUES (?, ?, ?, ?, ?, ?, 1, 'quiet_gap', ?, ?, ?)",
        (
            exchange_id,
            channel_id,
            exchange_id,
            exchange_id,
            iso(NOW),
            iso(NOW),
            f"h{exchange_id}",
            status,
            retry_count,
        ),
    )
    if cascade is not None:
        mark(conn, exchange_id, cascade)
    conn.commit()


def add_run(
    conn: sqlite3.Connection,
    exchange_id: int,
    started_at: datetime,
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cost_usd: float | None = 0.01,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO prompt_versions (version, text_sha256, created_at)"
        " VALUES ('v1', 'sha', ?)",
        (iso(NOW),),
    )
    conn.execute(
        "INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at,"
        " finished_at, input_tokens, output_tokens, mode, outcome, cost_usd)"
        " VALUES (?, 'm', 'v1', ?, ?, ?, ?, 'live', 'ok', ?)",
        (
            exchange_id,
            iso(started_at),
            iso(started_at + timedelta(seconds=10)),
            input_tokens,
            output_tokens,
            cost_usd,
        ),
    )
    conn.commit()


def add_probe_run(
    conn: sqlite3.Connection,
    started_at: datetime,
    *,
    claim_count: int = 1,
    cost_usd: float | None = 0.02,
) -> None:
    conn.execute(
        "INSERT INTO probe_runs (probe_model, judge_model, started_at, finished_at, claim_count,"
        " batched, recall_input_tokens, recall_output_tokens, judge_input_tokens,"
        " judge_output_tokens, cost_usd, outcome)"
        " VALUES ('p', 'j', ?, ?, ?, ?, 10, 5, 8, 4, ?, 'ok')",
        (
            iso(started_at),
            iso(started_at + timedelta(seconds=5)),
            claim_count,
            1 if claim_count > 1 else 0,
            cost_usd,
        ),
    )
    conn.commit()


def status(
    conn: sqlite3.Connection,
    *,
    exclude_channels: frozenset[str] = frozenset(),
    throughput_window_hours: int = 6,
) -> StatusReport:
    return collect_status(
        conn,
        exclude_channels=exclude_channels,
        now=NOW,
        max_retries=MAX_RETRIES,
        throughput_window_hours=throughput_window_hours,
    )


def test_pending_archived_counts_only_what_extract_would_claim(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    add_exchange(conn, 1)  # passes
    add_exchange(conn, 2, cascade="embed_irrelevant")  # ruled out
    add_exchange(conn, 3, status="done")  # not pending
    add_exchange(conn, 4, retry_count=MAX_RETRIES)  # exhausted retries

    report = status(conn)

    assert report.pending_exchanges == 2  # pending with retries left, archive aside
    assert report.pending_archived == 1


def test_pending_archived_agrees_with_the_claim_query(tmp_path: Path) -> None:
    # The count and the queue must not drift apart: one is the promise the
    # other keeps.
    conn = fresh(tmp_path)
    add_exchange(conn, 1)
    add_exchange(conn, 2, cascade="embed_irrelevant")
    add_exchange(conn, 3, cascade="lexicon")
    add_exchange(conn, 4, cascade=None)
    add_exchange(conn, 5, retry_count=MAX_RETRIES)
    add_exchange(conn, 6, status="done")

    claimable = claimable_exchanges(conn, 1000, MAX_RETRIES)

    assert status(conn).pending_archived == len(claimable)


def test_pending_archived_honours_the_channel_denylist(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    add_exchange(conn, 1, channel_id=1)
    add_exchange(conn, 2, channel_id=2)

    report = status(conn, exclude_channels=frozenset({"channel-2"}))

    assert report.pending_archived == 1


def test_extraction_throughput_uses_a_trailing_window(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    add_exchange(conn, 1)
    for minutes in (10, 70, 130):  # three runs inside a 6 hour window
        add_run(conn, 1, NOW - timedelta(minutes=minutes))
    add_run(conn, 1, NOW - timedelta(hours=20))  # outside it

    report = status(conn, throughput_window_hours=6)

    assert report.extraction_per_hour == 0.5  # 3 runs / 6 hours


def test_probe_throughput_counts_claims_not_calls(tmp_path: Path) -> None:
    # A batched pair covers several claims; the rate people care about is
    # claims cleared per hour.
    conn = fresh(tmp_path)
    add_probe_run(conn, NOW - timedelta(minutes=30), claim_count=10)
    add_probe_run(conn, NOW - timedelta(minutes=45), claim_count=8)

    report = status(conn, throughput_window_hours=6)

    assert report.probe_per_hour == 3.0  # 18 claims / 6 hours


def test_eta_divides_the_archived_queue_by_the_trailing_rate(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    for exchange_id in range(1, 13):
        add_exchange(conn, exchange_id)
    add_run(conn, 1, NOW - timedelta(minutes=10))
    add_run(conn, 1, NOW - timedelta(minutes=20))

    report = status(conn, throughput_window_hours=6)

    # 12 pending, 2 runs / 6h = 0.3333/h
    assert report.extraction_per_hour is not None
    assert round(report.extraction_eta_hours or 0.0, 1) == 36.0


def test_no_recent_runs_means_no_rate_and_no_eta(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    add_exchange(conn, 1)
    add_run(conn, 1, NOW - timedelta(days=3))

    report = status(conn, throughput_window_hours=6)

    assert report.extraction_per_hour is None
    assert report.extraction_eta_hours is None
    assert report.probe_per_hour is None


def test_an_empty_queue_has_a_zero_eta_not_a_missing_one(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    add_exchange(conn, 1, status="done")
    add_run(conn, 1, NOW - timedelta(minutes=10))

    report = status(conn, throughput_window_hours=6)

    assert report.pending_archived == 0
    assert report.extraction_eta_hours == 0.0


def test_spend_totals_come_from_both_stages(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    add_exchange(conn, 1)
    add_run(conn, 1, NOW - timedelta(minutes=10), input_tokens=100, output_tokens=20, cost_usd=0.01)
    add_run(conn, 1, NOW - timedelta(minutes=20), input_tokens=200, output_tokens=30, cost_usd=0.02)
    add_probe_run(conn, NOW - timedelta(minutes=5), claim_count=4, cost_usd=0.04)

    report = status(conn)

    assert report.extraction_input_tokens == 300
    assert report.extraction_output_tokens == 50
    assert report.extraction_cost_usd == 0.03
    # recall 10 + judge 8 in, recall 5 + judge 4 out
    assert report.probe_input_tokens == 18
    assert report.probe_output_tokens == 9
    assert report.probe_cost_usd == 0.04
    assert report.probe_claims == 4


def test_unreported_cost_is_none_rather_than_zero(tmp_path: Path) -> None:
    # A backend that reports no cost must not make the run look free.
    conn = fresh(tmp_path)
    add_exchange(conn, 1)
    add_run(conn, 1, NOW - timedelta(minutes=10), cost_usd=None)

    report = status(conn)

    assert report.extraction_cost_usd is None
    assert report.probe_cost_usd is None
