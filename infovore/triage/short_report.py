import sqlite3
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple

from infovore.db.batch import exchange_inputs_for_ids
from infovore.rows import Label
from infovore.triage.cascade import RESIDUE, EmbedStage, run_cascade, text_chars
from infovore.triage.human import held_out_ids, training_labels
from infovore.triage.lexicon import Lexicon

EDGES: Final = (100, 200, 500, 1000)
BUCKETS: Final = ("<100", "100-200", "200-500", "500-1000", "1000+")
PRECISION: Final = 0.95
MIN_LABELLED: Final = 20
BATCH: Final = 2000


class Row(NamedTuple):
    exchange_id: int
    chars: int
    hits: int


@dataclass(frozen=True)
class BucketLine:
    bucket: str
    tech: bool
    corpus: int
    relevant: int
    irrelevant: int
    held_relevant: int
    held_irrelevant: int

    @property
    def percent_irrelevant(self) -> float | None:
        labelled = self.relevant + self.irrelevant
        return 100 * self.irrelevant / labelled if labelled else None


@dataclass(frozen=True)
class Cutoff:
    limit: int
    precision: float
    n: int


def bucket_of(chars: int) -> str:
    return BUCKETS[sum(chars >= edge for edge in EDGES)]


def bucket_table(
    rows: Sequence[Row], labels: dict[int, Label], held: Collection[int]
) -> list[BucketLine]:
    lines = []
    for bucket in BUCKETS:
        for tech in (False, True):
            mine = [r for r in rows if bucket_of(r.chars) == bucket and (r.hits > 0) == tech]
            marked = [
                (r.exchange_id in held, labels[r.exchange_id])
                for r in mine
                if r.exchange_id in labels
            ]
            lines.append(
                BucketLine(
                    bucket,
                    tech,
                    len(mine),
                    sum(1 for h, lab in marked if not h and lab is Label.LORE),
                    sum(1 for h, lab in marked if not h and lab is not Label.LORE),
                    sum(1 for h, lab in marked if h and lab is Label.LORE),
                    sum(1 for h, lab in marked if h and lab is not Label.LORE),
                )
            )
    return lines


def best_cutoff(
    rows: Sequence[Row],
    labels: dict[int, Label],
    held: Collection[int],
    precision: float = PRECISION,
    minimum: int = MIN_LABELLED,
) -> Cutoff | None:
    build = [
        r for r in rows if r.hits == 0 and r.exchange_id in labels and r.exchange_id not in held
    ]
    for limit in sorted({r.chars + 1 for r in build}, reverse=True):
        below = [labels[r.exchange_id] is not Label.LORE for r in build if r.chars < limit]
        if len(below) >= minimum and sum(below) / len(below) >= precision:
            return Cutoff(limit, sum(below) / len(below), len(below))
    return None


def reached_stage_four(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    lexicon: Lexicon,
    t_high: float,
    exclude: frozenset[str],
    progress: Callable[[int, int], None],
) -> list[Row]:
    abstain = EmbedStage.abstaining("short-report")
    rows: list[Row] = []
    for start in range(0, len(ids), BATCH):
        chunk = ids[start : start + BATCH]
        kept = [
            o
            for o in run_cascade(conn, chunk, lexicon, t_high, abstain, exclude)
            if o.stage == RESIDUE
        ]
        inputs = exchange_inputs_for_ids(conn, [o.exchange_id for o in kept])
        for o in kept:
            rows.append(Row(o.exchange_id, text_chars(inputs[o.exchange_id].messages), o.hits))
        progress(min(start + BATCH, len(ids)), len(ids))
    return rows


def cutoff_line(cutoff: Cutoff | None) -> str:
    if cutoff is None:
        return f"no cutoff qualifies (precision >= {PRECISION}, at least {MIN_LABELLED} labelled)\n"
    return f"cutoff L={cutoff.limit} precision={cutoff.precision:.3f} n={cutoff.n}\n"


def short_limit(
    conn: sqlite3.Connection, lexicon: Lexicon, t_high: float, exclude: frozenset[str]
) -> int | None:
    labels, _ = training_labels(conn, exclude_channels=exclude)
    held = held_out_ids(conn)
    rows = reached_stage_four(conn, sorted(labels), lexicon, t_high, exclude, lambda *_: None)
    cutoff = best_cutoff(rows, labels, held)
    return None if cutoff is None else cutoff.limit
