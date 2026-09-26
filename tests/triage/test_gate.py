import sqlite3
from datetime import UTC, datetime
from itertools import product

from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule
from infovore.triage.gate import gate_sql, passes_gate

NOW = datetime(2026, 1, 1, tzinfo=UTC)
MIN_SCORE = 0.3
MIN_P_LORE = 0.5


def make_exchange(
    exchange_id: int, triage_score: float | None, p_lore: float | None
) -> ExchangeRow:
    return ExchangeRow(
        id=exchange_id,
        channel_id=1,
        thread_id=None,
        first_message_id=1,
        last_message_id=1,
        started_at=NOW,
        ended_at=NOW,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=f"h{exchange_id}",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
        triage_score=triage_score,
        triage_reasons=None,
        triage_version="t1",
        p_lore=p_lore,
        p_lore_model=None if p_lore is None else 1,
    )


def test_passes_gate_on_rule_score_when_p_lore_is_none() -> None:
    exchange = make_exchange(1, triage_score=0.5, p_lore=None)
    assert passes_gate(exchange, min_score=MIN_SCORE, min_p_lore=MIN_P_LORE) is True


def test_fails_gate_on_rule_score_below_threshold_when_p_lore_is_none() -> None:
    exchange = make_exchange(1, triage_score=0.1, p_lore=None)
    assert passes_gate(exchange, min_score=MIN_SCORE, min_p_lore=MIN_P_LORE) is False


def test_fails_gate_when_both_scores_are_none() -> None:
    exchange = make_exchange(1, triage_score=None, p_lore=None)
    assert passes_gate(exchange, min_score=MIN_SCORE, min_p_lore=MIN_P_LORE) is False


def test_passes_gate_on_p_lore_even_when_rule_score_would_fail() -> None:
    exchange = make_exchange(1, triage_score=0.0, p_lore=0.9)
    assert passes_gate(exchange, min_score=MIN_SCORE, min_p_lore=MIN_P_LORE) is True


def test_fails_gate_on_p_lore_even_when_rule_score_would_pass() -> None:
    exchange = make_exchange(1, triage_score=1.0, p_lore=0.1)
    assert passes_gate(exchange, min_score=MIN_SCORE, min_p_lore=MIN_P_LORE) is False


def test_passes_gate_when_p_lore_equals_threshold() -> None:
    exchange = make_exchange(1, triage_score=None, p_lore=MIN_P_LORE)
    assert passes_gate(exchange, min_score=MIN_SCORE, min_p_lore=MIN_P_LORE) is True


def test_passes_gate_when_rule_score_equals_threshold() -> None:
    exchange = make_exchange(1, triage_score=MIN_SCORE, p_lore=None)
    assert passes_gate(exchange, min_score=MIN_SCORE, min_p_lore=MIN_P_LORE) is True


SCORE_CASES = (None, 0.0, MIN_SCORE, 1.0)
P_LORE_CASES = (None, 0.0, MIN_P_LORE, 1.0)


def test_gate_sql_matches_passes_gate_over_every_case() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE exchanges (id INTEGER PRIMARY KEY, triage_score REAL, p_lore REAL)")

    exchanges = [
        make_exchange(index, triage_score=score, p_lore=p_lore)
        for index, (score, p_lore) in enumerate(product(SCORE_CASES, P_LORE_CASES), start=1)
    ]
    conn.executemany(
        "INSERT INTO exchanges (id, triage_score, p_lore) VALUES (?, ?, ?)",
        [(exchange.id, exchange.triage_score, exchange.p_lore) for exchange in exchanges],
    )

    expected = {
        exchange.id
        for exchange in exchanges
        if passes_gate(exchange, min_score=MIN_SCORE, min_p_lore=MIN_P_LORE)
    }

    clause, params = gate_sql(MIN_SCORE, MIN_P_LORE)
    actual = {row["id"] for row in conn.execute(f"SELECT id FROM exchanges WHERE {clause}", params)}

    assert actual == expected
    assert 0 < len(expected) < len(exchanges)
