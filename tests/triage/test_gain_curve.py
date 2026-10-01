import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.triage.gain_curve import (
    MIN_PLAUSIBLE_INPUT_TOKENS,
    ORACLE,
    RANDOM,
    Candidate,
    GainReport,
    compute_gain_curve,
    format_gain_report,
    gain_points,
)

AT = datetime(2026, 1, 1, tzinfo=UTC)


def _c(tokens: int, claims: int, p_lore: float) -> Candidate:
    return Candidate(exchange_id=tokens, tokens=tokens, claims=claims, p_lore=p_lore)


def test_a_perfect_ranker_front_loads_the_claims() -> None:
    """Ten equal-cost exchanges, all the claims in the first one. A ranker that
    puts it first recovers everything for a tenth of the budget."""
    rows = [_c(100, 10, 1.0)] + [_c(100, 0, 0.0) for _ in range(9)]

    points = {p.label: p for p in gain_points(rows, seed=0)}

    assert points["p_lore"].claims_at_10_percent == pytest.approx(100.0)
    assert points["p_lore"].budget_for_90_percent == pytest.approx(10.0)


def test_an_inverted_ranker_pays_almost_everything() -> None:
    """Same corpus, but p_lore ranks the one valuable exchange last."""
    rows = [_c(100, 10, 0.0)] + [_c(100, 0, 1.0) for _ in range(9)]

    points = {p.label: p for p in gain_points(rows, seed=0)}

    assert points["p_lore"].claims_at_10_percent == pytest.approx(0.0)
    assert points["p_lore"].budget_for_90_percent == pytest.approx(100.0)


def test_the_random_control_is_always_reported() -> None:
    """p_lore at 12.7% looks respectable until random scores 9.8%. The control
    is structural, not an option a caller can forget."""
    rows = [_c(100, 1, 0.5) for _ in range(10)]

    labels = [p.label for p in gain_points(rows, seed=0)]

    assert RANDOM in labels
    assert ORACLE in labels


def test_the_oracle_is_an_upper_bound_on_every_measure() -> None:
    rows = [_c(10 * i, i % 4, (i * 7 % 10) / 10) for i in range(1, 40)]

    points = {p.label: p for p in gain_points(rows, seed=0)}

    assert points[ORACLE].claims_at_10_percent >= points["p_lore"].claims_at_10_percent
    assert points[ORACLE].budget_for_90_percent <= points["p_lore"].budget_for_90_percent


def test_the_random_control_is_reproducible_for_a_seed() -> None:
    rows = [_c(10 * i, i % 3, 0.5) for i in range(1, 30)]

    first = {p.label: p for p in gain_points(rows, seed=7)}[RANDOM]
    second = {p.label: p for p in gain_points(rows, seed=7)}[RANDOM]

    assert first == second


def test_a_corpus_with_no_claims_reports_zero_not_a_crash() -> None:
    rows = [_c(100, 0, 0.5) for _ in range(5)]

    points = {p.label: p for p in gain_points(rows, seed=0)}

    assert points["p_lore"].claims_at_10_percent == 0.0
    assert points["p_lore"].budget_for_90_percent == 100.0


def test_an_empty_corpus_reports_nothing() -> None:
    assert gain_points([], seed=0) == ()


def test_an_exchange_too_expensive_for_the_budget_is_not_counted() -> None:
    """The budget is a token ceiling, so an exchange that would overrun it
    contributes nothing at that budget even if it is ranked first."""
    rows = [_c(1000, 5, 1.0), _c(10, 1, 0.0)]

    points = {p.label: p for p in gain_points(rows, seed=0)}

    assert points["p_lore"].claims_at_10_percent == pytest.approx(0.0)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    connection.execute(
        "INSERT INTO prompt_versions (version, text_sha256, created_at) VALUES ('v5', 's', ?)",
        (AT.isoformat(),),
    )
    return connection


def _run(
    conn: sqlite3.Connection,
    run_id: int,
    mode: str,
    tokens: int,
    claims: int,
    p_lore: float | None,
) -> None:
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash, p_lore)"
        " VALUES (?, 1, 1, 1, ?, ?, 1, 'quiet_gap', ?, ?)",
        (run_id, AT.isoformat(), AT.isoformat(), f"h{run_id}", p_lore),
    )
    conn.execute(
        "INSERT INTO extraction_runs (id, exchange_id, model, prompt_version, started_at,"
        " mode, outcome, input_tokens, output_tokens)"
        " VALUES (?, ?, 'm', 'v5', ?, ?, 'ok', ?, 0)",
        (run_id, run_id, AT.isoformat(), mode, tokens),
    )
    for index in range(claims):
        conn.execute(
            "INSERT INTO claims (exchange_id, extraction_run_id, statement, subject, kind,"
            " confidence, probe_question, permalink)"
            " VALUES (?, ?, ?, 'sub', 'fact', 0.9, '', 'p')",
            (run_id, run_id, f"claim {run_id}-{index}"),
        )


def test_compute_reads_only_the_requested_mode(conn: sqlite3.Connection) -> None:
    _run(conn, 1, "trial", 20_000, 3, 0.9)
    _run(conn, 2, "live", 20_000, 3, 1.0)

    report = compute_gain_curve(conn, mode="trial", seed=0)

    assert report.exchanges == 1
    assert report.total_claims == 3


def test_compute_skips_exchanges_with_no_score(conn: sqlite3.Connection) -> None:
    """An unscored exchange cannot be ranked by the gate, so including it
    would silently credit the gate with a random placement."""
    _run(conn, 1, "trial", 20_000, 3, 0.9)
    _run(conn, 2, "trial", 20_000, 3, None)

    report = compute_gain_curve(conn, mode="trial", seed=0)

    assert report.exchanges == 1


def test_compute_reports_the_controls_alongside_the_gate(conn: sqlite3.Connection) -> None:
    for index in range(1, 11):
        _run(conn, index, "trial", 20_000, index % 3, index / 10)

    report = compute_gain_curve(conn, mode="trial", seed=0)

    assert {point.label for point in report.points} == {"p_lore", ORACLE, RANDOM}


def test_a_free_exchange_is_always_inside_the_budget() -> None:
    """A zero-token candidate never overruns, so the loop runs to completion
    rather than breaking out of it."""
    rows = [Candidate(exchange_id=1, tokens=0, claims=4, p_lore=1.0)]

    points = {p.label: p for p in gain_points(rows, seed=0)}

    assert points["p_lore"].claims_at_10_percent == pytest.approx(100.0)
    assert points["p_lore"].budget_for_90_percent == pytest.approx(100.0)


def test_the_report_formats_as_a_table() -> None:
    rows = [_c(100, 2, 0.9), _c(100, 0, 0.1)]
    report = GainReport(
        mode="trial",
        exchanges=2,
        total_tokens=200,
        total_claims=2,
        points=gain_points(rows, seed=0),
    )

    lines = format_gain_report(report)

    assert "trial runs: 2 exchanges, 2 claims, 200 tokens" in lines[0]
    assert "ordering" in lines[1]
    assert any(line.strip().startswith("p_lore") for line in lines)
    assert any(RANDOM in line for line in lines)


def test_an_empty_report_says_so_rather_than_printing_a_bare_table() -> None:
    report = GainReport(mode="trial", exchanges=0, total_tokens=0, total_claims=0, points=())

    lines = format_gain_report(report)

    assert lines == [
        "gain curve, trial runs: no scored runs with recorded tokens yet,"
        " so there is nothing to rank"
    ]


def test_a_run_too_cheap_to_be_real_is_excluded(conn: sqlite3.Connection) -> None:
    """Prompt v1/v2-era trial runs recorded 2 input tokens. A run billed less
    than the system prompt provably occupies cannot have real accounting, and
    at 2 tokens it is FREE, so it sorts to the front of any claims-per-token
    ordering and distorts every arm of the curve."""
    _run(conn, 1, "trial", 20_000, 3, 0.9)
    _run(conn, 2, "trial", 2, 5, 0.1)

    report = compute_gain_curve(conn, mode="trial", seed=0)

    assert report.exchanges == 1
    assert report.excluded_implausible == 1


def test_the_exclusion_count_is_reported_not_hidden(conn: sqlite3.Connection) -> None:
    _run(conn, 1, "trial", 20_000, 3, 0.9)
    _run(conn, 2, "trial", 2, 5, 0.1)

    lines = format_gain_report(compute_gain_curve(conn, mode="trial", seed=0))

    assert any("1 excluded" in line for line in lines)


def test_nothing_is_said_about_exclusions_when_there_are_none(conn: sqlite3.Connection) -> None:
    _run(conn, 1, "trial", 20_000, 3, 0.9)

    lines = format_gain_report(compute_gain_curve(conn, mode="trial", seed=0))

    assert not any("excluded" in line for line in lines)


def test_the_floor_is_derived_from_the_prompt_not_guessed() -> None:
    """A hard-coded number would rot the moment the prompt changed."""
    from infovore.extract.prompt import system_prompt

    assert len(system_prompt("v5")) // 8 <= MIN_PLAUSIBLE_INPUT_TOKENS
    assert len(system_prompt("v5")) >= MIN_PLAUSIBLE_INPUT_TOKENS


def test_a_stricter_floor_can_be_requested(conn: sqlite3.Connection) -> None:
    """The v2-era runs average 997 input tokens, above the provable floor but
    still far below the ~9,500 a real v3+ call costs, so a caller measuring
    that era needs to say so."""
    _run(conn, 1, "trial", 20_000, 3, 0.9)
    _run(conn, 2, "trial", 1_000, 5, 0.1)

    report = compute_gain_curve(conn, mode="trial", seed=0, min_input_tokens=3_000)

    assert report.exchanges == 1
    assert report.excluded_implausible == 1
