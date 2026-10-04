import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

from infovore.triage.cascade import Outcome, write_outcomes
from infovore.triage.human import HumanFit
from infovore.triage.lexicon import load_lexicon

AT = datetime(2026, 1, 1, tzinfo=UTC)
_FIT = cast(HumanFit, SimpleNamespace(recipe={"model": "stub"}))
_OUTCOMES = {
    "lexicon": ("lexicon", "relevant", 0.9, None),
    "bayes_relevant": ("bayes", "relevant", 0.1, 0.95),
    "bayes_irrelevant": ("bayes", "irrelevant", 0.1, 0.02),
    "residue": ("residue", "residue", 0.1, 0.5),
    "denylist": ("denylist", "irrelevant", 0.0, None),
    "no_text": ("no_text", "no_text", 0.0, None),
}
ARCHIVED_KINDS = ("lexicon", "bayes_relevant", "residue")
RULED_OUT_KINDS = ("bayes_irrelevant", "denylist", "no_text")


def mark_all(
    conn: sqlite3.Connection, exchange_ids: Iterable[int], kind: str, at: datetime = AT
) -> None:
    """Record cascade outcomes through the real writer, as `relevance cascade --write` does."""
    stage, decision, share, p_bayes = _OUTCOMES[kind]
    outcomes = [Outcome(eid, share, 1, 3, p_bayes, stage, decision) for eid in exchange_ids]
    write_outcomes(conn, outcomes, load_lexicon(), 0.3, _FIT, at)


def mark(
    conn: sqlite3.Connection, exchange_id: int, kind: str = "residue", at: datetime = AT
) -> None:
    mark_all(conn, [exchange_id], kind, at)
