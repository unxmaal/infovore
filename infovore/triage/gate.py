from infovore.rows import ExchangeRow


def passes_gate(exchange: ExchangeRow, min_score: float) -> bool:
    score = exchange.triage_score
    return score is not None and score >= min_score
