import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from infovore.db.batch import SQLITE_MAX_VARIABLES
from infovore.db.exchange_text import text_message_counts
from infovore.reputation.stats import Interval, Scored, auc_interval
from infovore.rows import Label

LEXICON_UNDECIDED: Final = "lexicon-undecided"
LEXICON_WRONG: Final = "lexicon-undecided-or-wrong"
DENYLIST_WRONG: Final = "denylist-wrong"
MANY_MESSAGES: Final = "3+ text messages"
SUBSETS: Final = (LEXICON_UNDECIDED, LEXICON_WRONG, DENYLIST_WRONG, MANY_MESSAGES)
MIN_TEXT_MESSAGES: Final = 3
_DECIDED: Final = ("relevant", "irrelevant")


@dataclass(frozen=True)
class PopulationReport:
    overall: Interval
    subsets: dict[str, Interval]


def stage_labels(conn: sqlite3.Connection, stage: str, ids: Sequence[int]) -> dict[int, str | None]:
    latest: dict[int, str | None] = {}
    for start in range(0, len(ids), SQLITE_MAX_VARIABLES):
        chunk = list(ids[start : start + SQLITE_MAX_VARIABLES])
        marks = ",".join("?" for _ in chunk)
        for row in conn.execute(
            "SELECT subject_id, label FROM annotations WHERE scorer = ?"
            f" AND subject_kind = 'exchange' AND subject_id IN ({marks}) ORDER BY id",
            (f"relevance_{stage}", *chunk),
        ):
            latest[row["subject_id"]] = row["label"]
    return latest


def strata(conn: sqlite3.Connection, labels: Mapping[int, Label]) -> dict[str, list[int]]:
    ids = sorted(labels)
    lexicon = stage_labels(conn, "lexicon", ids)
    denylist = stage_labels(conn, "denylist", ids)
    counts = text_message_counts(conn, ids)

    def human(eid: int) -> str:
        return "relevant" if labels[eid] is Label.LORE else "irrelevant"

    undecided = [eid for eid in ids if lexicon.get(eid) not in _DECIDED]
    return {
        LEXICON_UNDECIDED: undecided,
        LEXICON_WRONG: [
            eid for eid in ids if lexicon.get(eid) not in _DECIDED or lexicon[eid] != human(eid)
        ],
        DENYLIST_WRONG: [
            eid for eid in ids if denylist.get(eid) == "irrelevant" and labels[eid] is Label.LORE
        ],
        MANY_MESSAGES: [eid for eid in ids if counts[eid] >= MIN_TEXT_MESSAGES],
    }


def scored_pairs(
    scores: Mapping[int, float], labels: Mapping[int, Label], ids: Sequence[int]
) -> Scored:
    return [(scores[eid], labels[eid]) for eid in ids if eid in scores and eid in labels]


def evaluate_population(
    conn: sqlite3.Connection,
    scores: Mapping[int, float],
    labels: Mapping[int, Label],
    seed: int,
) -> PopulationReport:
    overall = auc_interval(scored_pairs(scores, labels, sorted(labels)), seed)
    subsets = {
        name: auc_interval(scored_pairs(scores, labels, ids), seed)
        for name, ids in strata(conn, labels).items()
    }
    return PopulationReport(overall, subsets)
