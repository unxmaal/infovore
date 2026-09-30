import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.connection import migrate, open_database
from infovore.triage.yield_report import DEFAULT_BANDS, compute_yield_by_band

NOW = datetime(2026, 1, 1, tzinfo=UTC).isoformat()


def fresh(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO prompt_versions (version, text_sha256, created_at) VALUES ('v1', 's', ?)",
        (NOW,),
    )
    return conn


def add(
    conn: sqlite3.Connection,
    exchange_id: int,
    p_lore: float | None,
    claims: int,
    *,
    mode: str = "live",
    input_tokens: int = 1000,
    output_tokens: int = 400,
    cost_usd: float | None = 0.05,
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, 1, 1, 1, 'a', ?, 'x', ?, '{}')",
        (exchange_id, NOW, NOW),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash, p_lore)"
        " VALUES (?, 1, ?, ?, ?, ?, 1, 'quiet_gap', ?, ?)",
        (exchange_id, exchange_id, exchange_id, NOW, NOW, f"h{exchange_id}", p_lore),
    )
    cursor = conn.execute(
        "INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at,"
        " finished_at, input_tokens, output_tokens, mode, outcome, cost_usd)"
        " VALUES (?, 'm', 'v1', ?, ?, ?, ?, ?, 'ok', ?)",
        (exchange_id, NOW, NOW, input_tokens, output_tokens, mode, cost_usd),
    )
    for index in range(claims):
        conn.execute(
            "INSERT INTO claims (exchange_id, extraction_run_id, statement, subject, kind,"
            " confidence, probe_question, permalink, novelty)"
            " VALUES (?, ?, ?, 's', 'fact', 0.9, 'q?', 'p', 'unprobed')",
            (exchange_id, cursor.lastrowid, f"claim {exchange_id}.{index}"),
        )
    conn.commit()


def test_reports_realised_yield_per_band(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    add(conn, 1, 0.99999, claims=4)
    add(conn, 2, 0.99999, claims=0)
    add(conn, 3, 0.5, claims=2)

    bands = compute_yield_by_band(conn)
    top = next(band for band in bands if band.lower == 0.9999)

    assert top.exchanges == 2
    assert top.barren == 1
    assert top.claims == 4
    assert top.claims_per_exchange == 2.0
    assert top.barren_rate == 0.5


def test_a_barren_band_is_visible_as_wasted_spend(tmp_path: Path) -> None:
    # The whole point: an exchange that cost tokens and produced nothing.
    conn = fresh(tmp_path)
    add(conn, 1, 0.99999, claims=0, input_tokens=7000, output_tokens=300, cost_usd=0.02)

    top = next(band for band in compute_yield_by_band(conn) if band.lower == 0.9999)

    assert top.barren == 1
    assert top.input_tokens == 7000
    assert top.cost_usd == 0.02
    assert top.cost_per_claim is None  # no claims: cost per claim is undefined, not zero


def test_cost_per_claim_is_reported_where_claims_exist(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    add(conn, 1, 0.99999, claims=4, cost_usd=0.08)

    top = next(band for band in compute_yield_by_band(conn) if band.lower == 0.9999)

    assert top.cost_per_claim == 0.02


def test_trial_runs_are_excluded(tmp_path: Path) -> None:
    # Trial runs deliberately sample across the score range, so mixing them
    # in makes the live gate look like it discriminates when it does not.
    conn = fresh(tmp_path)
    add(conn, 1, 0.99999, claims=1, mode="live")
    add(conn, 2, 0.2, claims=0, mode="trial")

    bands = compute_yield_by_band(conn)

    assert sum(band.exchanges for band in bands) == 1


def test_unscored_exchanges_are_counted_separately(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    add(conn, 1, None, claims=3)

    bands = compute_yield_by_band(conn)
    unscored = next(band for band in bands if band.lower is None)

    assert unscored.exchanges == 1
    assert unscored.claims == 3


def test_every_band_is_present_even_when_empty(tmp_path: Path) -> None:
    conn = fresh(tmp_path)

    bands = compute_yield_by_band(conn)

    assert len(bands) == len(DEFAULT_BANDS) + 1  # plus the unscored row
    assert all(band.exchanges == 0 for band in bands)
    assert all(band.claims_per_exchange is None for band in bands)


def test_a_score_below_every_band_lands_in_the_lowest(tmp_path: Path) -> None:
    conn = fresh(tmp_path)
    add(conn, 1, 0.2, claims=1)

    bands = compute_yield_by_band(conn, bands=(0.5, 0.9))

    assert bands[0].lower == 0.5
    assert bands[0].exchanges == 1


def test_an_unreported_cost_leaves_the_band_cost_unknown(tmp_path: Path) -> None:
    # One run with no cost must not make the band look free.
    conn = fresh(tmp_path)
    add(conn, 1, 0.99999, claims=1, cost_usd=None)

    top = next(band for band in compute_yield_by_band(conn) if band.lower == 0.9999)

    assert top.cost_usd is None
    assert top.cost_per_claim is None
