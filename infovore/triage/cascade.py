import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from infovore.db.annotations import Annotation, record_annotation
from infovore.db.batch import exchange_inputs_for_ids
from infovore.db.channel_filter import excluded_exchange_ids
from infovore.db.exchange_text import text_message_counts
from infovore.eval.slices import BUILD
from infovore.rows import Label
from infovore.triage.bayes import p_lore
from infovore.triage.human import (
    MIN_PER_CLASS,
    HumanFit,
    InsufficientHumanLabelsError,
    fit_human,
    held_out_ids,
    human_features,
    training_labels,
)
from infovore.triage.lexicon import Lexicon, LexiconScore, score_lexicon

PRECISION_TARGET: Final = 0.97
BAYES_HIGH: Final = 0.9
BAYES_LOW: Final = 0.1
STAGES: Final = ("denylist", "no_text", "lexicon", "bayes", "residue")
SCORERS: Final = {stage: f"relevance_{stage}" for stage in STAGES}
RELEVANT: Final = "relevant"
IRRELEVANT: Final = "irrelevant"
RESIDUE: Final = "residue"
NO_TEXT: Final = "no_text"
UNSCORED: Final = (RESIDUE, NO_TEXT)


@dataclass(frozen=True)
class Outcome:
    exchange_id: int
    share: float
    hits: int
    messages: int
    p_bayes: float | None
    stage: str
    decision: str


@dataclass(frozen=True)
class StageReport:
    stage: str
    population: int
    decided: int
    relevant: int
    irrelevant: int
    labelled: int
    correct: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.labelled if self.labelled else None

    @property
    def share(self) -> float:
        return self.decided / self.population if self.population else 0.0


def tune_high(samples: Sequence[tuple[float, bool]], target: float = PRECISION_TARGET) -> float:
    for threshold in sorted({share for share, _ in samples if share > 0}):
        above = [relevant for share, relevant in samples if share >= threshold]
        if sum(above) / len(above) >= target:
            return threshold
    return 1.01


def decide_lexicon(score: LexiconScore, t_high: float) -> str | None:
    return RELEVANT if score.hits and score.share >= t_high else None


def decide_bayes(p: float | None) -> str | None:
    if p is None:
        return None
    if p >= BAYES_HIGH:
        return RELEVANT
    return IRRELEVANT if p <= BAYES_LOW else None


def tuning_samples(
    conn: sqlite3.Connection, lexicon: Lexicon, exclude_channels: frozenset[str] = frozenset()
) -> list[tuple[float, bool]]:
    held = held_out_ids(conn)
    labels, _ = training_labels(conn, exclude_channels=exclude_channels)
    build = {
        row["exchange_id"]
        for row in conn.execute(
            "SELECT exchange_id FROM current_slice_members WHERE name = ?", (BUILD,)
        )
    } - held
    ids = sorted(eid for eid in build if eid in labels)
    inputs = exchange_inputs_for_ids(conn, ids)
    return [
        (score_lexicon(lexicon, inputs[eid].messages).share, labels[eid] is Label.LORE)
        for eid in ids
    ]


def try_fit(
    conn: sqlite3.Connection,
    exclude_channels: frozenset[str] = frozenset(),
    minimum: int | None = None,
) -> tuple[HumanFit | None, str]:
    try:
        floor = MIN_PER_CLASS if minimum is None else minimum
        return fit_human(conn, minimum=floor, exclude_channels=exclude_channels), ""
    except InsufficientHumanLabelsError as error:
        return None, str(error)


def run_cascade(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    lexicon: Lexicon,
    t_high: float,
    fit: HumanFit | None,
    exclude_channels: frozenset[str] = frozenset(),
) -> list[Outcome]:
    denied = excluded_exchange_ids(conn, exclude_channels)
    live = [eid for eid in ids if eid not in denied]
    text = text_message_counts(conn, live)
    inputs = exchange_inputs_for_ids(conn, [eid for eid in live if text[eid]])
    outcomes = []
    for eid in ids:
        if eid in denied:
            outcomes.append(Outcome(eid, 0.0, 0, 0, None, "denylist", IRRELEVANT))
            continue
        if not text[eid]:
            outcomes.append(Outcome(eid, 0.0, 0, 0, None, NO_TEXT, NO_TEXT))
            continue
        one = inputs[eid]
        score = score_lexicon(lexicon, one.messages)
        decision = decide_lexicon(score, t_high)
        stage, p = "lexicon", None
        if decision is None and fit is not None:
            tokens = human_features(one.messages, one.reactions, one.attachments)
            p = p_lore(fit.model, tokens)
            stage, decision = "bayes", decide_bayes(p)
        if decision is None:
            stage, decision = "residue", RESIDUE
        outcomes.append(Outcome(eid, score.share, score.hits, score.messages, p, stage, decision))
    return outcomes


def current_exchange_ids(conn: sqlite3.Connection) -> list[int]:
    return [row["id"] for row in conn.execute("SELECT id FROM current_exchanges ORDER BY id")]


def run_cascade_batched(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    lexicon: Lexicon,
    t_high: float,
    fit: HumanFit | None,
    exclude_channels: frozenset[str],
    batch_size: int,
    progress: Callable[[int, int], None],
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        outcomes.extend(run_cascade(conn, chunk, lexicon, t_high, fit, exclude_channels))
        progress(len(outcomes), len(ids))
    return outcomes


def residue_channels(
    conn: sqlite3.Connection, outcomes: Sequence[Outcome], limit: int
) -> list[tuple[str, int]]:
    residue = [o.exchange_id for o in outcomes if o.stage == "residue"]
    counts: dict[str, int] = {}
    for start in range(0, len(residue), 500):
        chunk = residue[start : start + 500]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT COALESCE(c.name, CAST(e.channel_id AS TEXT)) AS name, COUNT(*) AS n"
            f" FROM current_exchanges e LEFT JOIN channels c ON c.id = e.channel_id"
            f" WHERE e.id IN ({marks}) GROUP BY name",
            chunk,
        )
        for row in rows:
            counts[row["name"]] = counts.get(row["name"], 0) + row["n"]
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:limit]


def stage_reports(outcomes: Sequence[Outcome], labels: dict[int, Label]) -> list[StageReport]:
    reports = []
    for stage in STAGES:
        mine = [o for o in outcomes if o.stage == stage]
        pairs = [
            (o.decision == RELEVANT, labels[o.exchange_id] is Label.LORE)
            for o in mine
            if o.exchange_id in labels
        ]
        tp = sum(1 for p, h in pairs if p and h)
        fp = sum(1 for p, h in pairs if p and not h)
        fn = sum(1 for p, h in pairs if not p and h)
        tn = sum(1 for p, h in pairs if not p and not h)
        reports.append(
            StageReport(
                stage=stage,
                population=len(outcomes),
                decided=len(mine),
                relevant=sum(1 for o in mine if o.decision == RELEVANT),
                irrelevant=sum(1 for o in mine if o.decision == IRRELEVANT),
                labelled=len(pairs) if stage not in UNSCORED else 0,
                correct=tp + tn if stage not in UNSCORED else 0,
                tp=tp if stage not in UNSCORED else 0,
                fp=fp if stage not in UNSCORED else 0,
                fn=fn if stage not in UNSCORED else 0,
                tn=tn if stage not in UNSCORED else 0,
            )
        )
    return reports


def _next_version(conn: sqlite3.Connection, scorer: str) -> int:
    row = conn.execute(
        "SELECT MAX(scorer_version) AS v FROM annotations WHERE scorer = ?", (scorer,)
    ).fetchone()
    return int(row["v"] or 0) + 1


def write_outcomes(
    conn: sqlite3.Connection,
    outcomes: Sequence[Outcome],
    lexicon: Lexicon,
    t_high: float,
    fit: HumanFit | None,
    at: datetime,
) -> dict[str, int]:
    versions = {stage: _next_version(conn, SCORERS[stage]) for stage in STAGES}
    base = {
        "lexicon_version": lexicon.version,
        "t_high": t_high,
        "bayes_band": [BAYES_LOW, BAYES_HIGH],
    }
    for o in outcomes:
        if o.stage in ("denylist", NO_TEXT):
            denied = Annotation(
                subject_kind="exchange",
                subject_id=o.exchange_id,
                scorer=SCORERS[o.stage],
                scorer_version=versions[o.stage],
                reproducibility="derived",
                score=None,
                label=o.decision,
                recipe=json.loads(json.dumps(base)),
                source_ref="relevance-cascade",
            )
            record_annotation(conn, denied, at)
            continue
        rows: list[tuple[str, float | None, str | None, dict[str, object]]] = [
            (
                "lexicon",
                o.share,
                o.decision if o.stage == "lexicon" else None,
                {**base, "hits": o.hits, "messages": o.messages},
            )
        ]
        if o.p_bayes is not None and fit is not None:
            rows.append(
                (
                    "bayes",
                    o.p_bayes,
                    o.decision if o.stage == "bayes" else None,
                    {**base, "model": fit.recipe},
                )
            )
        if o.stage == "residue":
            rows.append(("residue", None, RESIDUE, base))
        for stage, score, label, recipe in rows:
            record_annotation(
                conn,
                Annotation(
                    subject_kind="exchange",
                    subject_id=o.exchange_id,
                    scorer=SCORERS[stage],
                    scorer_version=versions[stage],
                    reproducibility="derived",
                    score=score,
                    label=label,
                    recipe=json.loads(json.dumps(recipe)),
                    source_ref="relevance-cascade",
                ),
                at,
            )
    conn.commit()
    return versions
