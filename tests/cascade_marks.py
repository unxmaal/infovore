import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime

from infovore.triage.cascade import EmbedStage, Outcome, write_outcomes
from infovore.triage.lexicon import load_lexicon

AT = datetime(2026, 1, 1, tzinfo=UTC)
_FIT = EmbedStage(lambda ids: {}, 0.0, 1.01, {"model": "stub"})
_OUTCOMES = {
    "lexicon": ("lexicon", "relevant", 0.9, None),
    "embed_relevant": ("embed", "relevant", 0.1, 0.95),
    "embed_irrelevant": ("embed", "irrelevant", 0.1, 0.02),
    "residue": ("residue", "residue", 0.1, 0.5),
    "denylist": ("denylist", "irrelevant", 0.0, None),
    "no_text": ("no_text", "no_text", 0.0, None),
}
ARCHIVED_KINDS = ("lexicon", "embed_relevant", "residue")
RULED_OUT_KINDS = ("embed_irrelevant", "denylist", "no_text")


def mark_all(
    conn: sqlite3.Connection, exchange_ids: Iterable[int], kind: str, at: datetime = AT
) -> None:
    """Record cascade outcomes through the real writer, as `relevance cascade --write` does."""
    stage, decision, share, p_embed = _OUTCOMES[kind]
    outcomes = [Outcome(eid, share, 1, 3, p_embed, stage, decision) for eid in exchange_ids]
    write_outcomes(conn, outcomes, load_lexicon(), 0.3, _FIT, at)


def mark(
    conn: sqlite3.Connection, exchange_id: int, kind: str = "residue", at: datetime = AT
) -> None:
    mark_all(conn, [exchange_id], kind, at)
