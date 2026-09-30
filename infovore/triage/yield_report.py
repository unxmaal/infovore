"""Gate evaluation from realised yield rather than from labels.

`triage --report` measures the classifier against labels, which says how it
ranks across the whole score range. It cannot say whether the gate is buying
anything AT ITS OPERATING POINT, because above a saturated threshold every
exchange scores alike. This asks the outcome question instead: of the
exchanges extraction actually ran on, how many produced nothing, and what did
that cost.
"""

import sqlite3
from dataclasses import dataclass

DEFAULT_BANDS: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999)


@dataclass(frozen=True)
class YieldBand:
    lower: float | None
    upper: float | None
    exchanges: int
    barren: int
    claims: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None

    @property
    def barren_rate(self) -> float | None:
        if self.exchanges == 0:
            return None
        return self.barren / self.exchanges

    @property
    def claims_per_exchange(self) -> float | None:
        if self.exchanges == 0:
            return None
        return self.claims / self.exchanges

    @property
    def cost_per_claim(self) -> float | None:
        if self.claims == 0 or self.cost_usd is None:
            return None
        return self.cost_usd / self.claims


def _band_of(p_lore: float | None, bands: tuple[float, ...]) -> int | None:
    if p_lore is None:
        return None
    eligible = [index for index, lower in enumerate(bands) if p_lore >= lower]
    return eligible[-1] if eligible else 0


def compute_yield_by_band(
    conn: sqlite3.Connection, bands: tuple[float, ...] = DEFAULT_BANDS
) -> list[YieldBand]:
    rows = conn.execute(
        "SELECT e.p_lore AS p_lore,"
        " r.input_tokens AS input_tokens,"
        " r.output_tokens AS output_tokens,"
        " r.cost_usd AS cost_usd,"
        " (SELECT COUNT(*) FROM claims c WHERE c.extraction_run_id = r.id) AS claims"
        " FROM extraction_runs r JOIN exchanges e ON e.id = r.exchange_id"
        " WHERE r.outcome = 'ok' AND r.mode = 'live'"
    ).fetchall()

    slots: dict[int | None, dict[str, float | int | None]] = {
        index: {"exchanges": 0, "barren": 0, "claims": 0, "input": 0, "output": 0, "cost": None}
        for index in [*range(len(bands)), None]
    }
    for row in rows:
        slot = slots[_band_of(row["p_lore"], bands)]
        slot["exchanges"] = int(slot["exchanges"] or 0) + 1
        slot["claims"] = int(slot["claims"] or 0) + int(row["claims"])
        if row["claims"] == 0:
            slot["barren"] = int(slot["barren"] or 0) + 1
        slot["input"] = int(slot["input"] or 0) + int(row["input_tokens"] or 0)
        slot["output"] = int(slot["output"] or 0) + int(row["output_tokens"] or 0)
        if row["cost_usd"] is not None:
            slot["cost"] = float(slot["cost"] or 0.0) + float(row["cost_usd"])

    result = [
        YieldBand(
            lower=bands[index],
            upper=bands[index + 1] if index + 1 < len(bands) else None,
            exchanges=int(slots[index]["exchanges"] or 0),
            barren=int(slots[index]["barren"] or 0),
            claims=int(slots[index]["claims"] or 0),
            input_tokens=int(slots[index]["input"] or 0),
            output_tokens=int(slots[index]["output"] or 0),
            cost_usd=None if slots[index]["cost"] is None else float(slots[index]["cost"] or 0.0),
        )
        for index in range(len(bands))
    ]
    unscored = slots[None]
    result.append(
        YieldBand(
            lower=None,
            upper=None,
            exchanges=int(unscored["exchanges"] or 0),
            barren=int(unscored["barren"] or 0),
            claims=int(unscored["claims"] or 0),
            input_tokens=int(unscored["input"] or 0),
            output_tokens=int(unscored["output"] or 0),
            cost_usd=None if unscored["cost"] is None else float(unscored["cost"] or 0.0),
        )
    )
    return result
