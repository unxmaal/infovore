from infovore.rows import ExchangeRow

GATE_SQL_CLAUSE = "((p_lore IS NOT NULL AND p_lore >= ?) OR (p_lore IS NULL AND triage_score >= ?))"


def gate_sql(min_score: float, min_p_lore: float) -> tuple[str, tuple[float, float]]:
    return GATE_SQL_CLAUSE, (min_p_lore, min_score)


def passes_gate(exchange: ExchangeRow, min_score: float, min_p_lore: float) -> bool:
    if exchange.p_lore is not None:
        return exchange.p_lore >= min_p_lore
    return exchange.triage_score is not None and exchange.triage_score >= min_score
