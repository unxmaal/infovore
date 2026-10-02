import hashlib
import json
import re
import sqlite3
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from importlib import resources
from typing import Final

from infovore.db.annotations import Annotation, record_annotation
from infovore.db.batch import BATCH_SIZE, exchange_inputs_for_ids
from infovore.rows import AttachmentRow, Label, MessageRow, ReactionRow
from infovore.triage.bayes import (
    HOLDOUT_BUCKETS,
    MAX_TOKEN_LENGTH,
    TOKEN,
    TRAILING_PUNCTUATION,
    Metrics,
    Model,
    auc,
    evaluate,
    in_holdout,
    p_lore,
    train,
)
from infovore.triage.rules import DEFAULT_RULES, TriageRules
from infovore.triage.score import score_exchange

HUMAN_SCORER: Final = "human_exchange"
SCORER: Final = "p_relevant_human"
MIN_PER_CLASS: Final = 200
EVAL_THRESHOLDS: Final = tuple(round(i / 10, 1) for i in range(1, 10))
SHARE_THRESHOLD: Final = 0.5
VERY_SHORT_CHARACTERS: Final = 30
_LABELS: Final = {"relevant": Label.LORE, "irrelevant": Label.NOISE}


class LlmLabelsNotTrainableError(ValueError):
    pass


class GazetteerError(ValueError):
    pass


class InsufficientHumanLabelsError(Exception):
    def __init__(self, relevant: int, irrelevant: int, minimum: int) -> None:
        self.relevant = relevant
        self.irrelevant = irrelevant
        self.minimum = minimum
        self.more_relevant = max(0, minimum - relevant)
        self.more_irrelevant = max(0, minimum - irrelevant)
        super().__init__(
            f"need at least {minimum} relevant and {minimum} irrelevant human exchange labels"
            f" (have relevant={relevant} irrelevant={irrelevant}); label"
            f" {self.more_relevant} more relevant and {self.more_irrelevant} more irrelevant"
        )


@dataclass(frozen=True)
class Gazetteer:
    version: str
    text: dict[str, re.Pattern[str]]
    message_share: dict[str, re.Pattern[str]]


def _compile(section: object, name: str, flags: int) -> dict[str, re.Pattern[str]]:
    if not isinstance(section, dict):
        raise GazetteerError(f"gazetteer: [{name}] must be a table")
    compiled: dict[str, re.Pattern[str]] = {}
    for key, patterns in section.items():
        if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
            raise GazetteerError(f"gazetteer: {name}.{key} must be a list of strings")
        compiled[key] = re.compile("|".join(f"(?:{item})" for item in patterns), flags)
    return compiled


def parse_gazetteer(text: str) -> Gazetteer:
    data = tomllib.loads(text)
    if set(data) != {"text", "message_share"}:
        raise GazetteerError("gazetteer: needs exactly [text] and [message_share]")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return Gazetteer(
        version="g-" + hashlib.sha256(canonical.encode()).hexdigest()[:12],
        text=_compile(data["text"], "text", re.IGNORECASE | re.MULTILINE),
        message_share=_compile(data["message_share"], "message_share", re.IGNORECASE),
    )


@lru_cache(maxsize=1)
def load_gazetteer() -> Gazetteer:
    path = resources.files("infovore.triage").joinpath("gazetteer.toml")
    return parse_gazetteer(path.read_text(encoding="utf-8"))


def _share(messages: Sequence[MessageRow], pattern: re.Pattern[str]) -> float:
    return sum(1 for message in messages if pattern.search(message.content.strip())) / len(messages)


def rule_names(
    messages: Sequence[MessageRow],
    reactions: Sequence[ReactionRow] = (),
    attachments: Sequence[AttachmentRow] = (),
    rules: TriageRules = DEFAULT_RULES,
    gazetteer: Gazetteer | None = None,
) -> frozenset[str]:
    if not messages:
        return frozenset()
    gazetteer = gazetteer or load_gazetteer()
    text = "\n".join(message.content for message in messages)
    hits = {name for name, _ in score_exchange(messages, reactions, attachments, rules).reasons}
    hits |= {f"gaz_{name}" for name, pattern in gazetteer.text.items() if pattern.search(text)}
    hits |= {
        f"gaz_{name}"
        for name, pattern in gazetteer.message_share.items()
        if _share(messages, pattern) > SHARE_THRESHOLD
    }
    if sum(1 for message in messages if message.author_is_bot) / len(messages) > SHARE_THRESHOLD:
        hits.add("gaz_bot_author")
    if len(text.strip()) < VERY_SHORT_CHARACTERS:
        hits.add("gaz_very_short")
    return frozenset(hits)


def human_features(
    messages: Sequence[MessageRow],
    reactions: Sequence[ReactionRow] = (),
    attachments: Sequence[AttachmentRow] = (),
    rules: TriageRules = DEFAULT_RULES,
    gazetteer: Gazetteer | None = None,
) -> frozenset[str]:
    words = {
        token.rstrip(TRAILING_PUNCTUATION)
        for message in messages
        for token in TOKEN.findall(message.content.lower())
    }
    words = {word for word in words if word and len(word) <= MAX_TOKEN_LENGTH}
    hits = rule_names(messages, reactions, attachments, rules, gazetteer)
    return frozenset(words | {f"__RULE_{name}" for name in hits})


def training_labels(
    conn: sqlite3.Connection, source: str = HUMAN_SCORER
) -> tuple[dict[int, Label], int]:
    if source != HUMAN_SCORER:
        raise LlmLabelsNotTrainableError(
            f"labels from {source!r} cannot train the gate; only {HUMAN_SCORER!r} annotations"
        )
    labels: dict[int, Label] = {}
    last_id = 0
    rows = conn.execute(
        "SELECT id, subject_id, label FROM annotations WHERE scorer = ?"
        " AND subject_kind = 'exchange' AND reproducibility = 'recorded' ORDER BY id",
        (HUMAN_SCORER,),
    )
    for row in rows:
        last_id = row["id"]
        if row["label"] in _LABELS:
            labels[row["subject_id"]] = _LABELS[row["label"]]
        else:
            labels.pop(row["subject_id"], None)
    return labels, last_id


@dataclass(frozen=True)
class HumanReport:
    relevant: int
    irrelevant: int
    holdout_size: int
    auc: float | None
    metrics: tuple[Metrics, ...]


@dataclass(frozen=True)
class HumanFit:
    model: Model
    report: HumanReport
    recipe: dict[str, object]


def fit_human(
    conn: sqlite3.Connection,
    rules: TriageRules = DEFAULT_RULES,
    minimum: int = MIN_PER_CLASS,
) -> HumanFit:
    labels, last_id = training_labels(conn)
    relevant = sum(1 for label in labels.values() if label is Label.LORE)
    irrelevant = len(labels) - relevant
    if relevant < minimum or irrelevant < minimum:
        raise InsufficientHumanLabelsError(relevant, irrelevant, minimum)
    gazetteer = load_gazetteer()
    ids = sorted(labels)
    examples: list[tuple[int, frozenset[str], Label]] = []
    for start in range(0, len(ids), BATCH_SIZE):
        batch = ids[start : start + BATCH_SIZE]
        inputs = exchange_inputs_for_ids(conn, batch)
        for exchange_id in batch:
            one = inputs[exchange_id]
            tokens = human_features(one.messages, one.reactions, one.attachments, rules, gazetteer)
            examples.append((exchange_id, tokens, labels[exchange_id]))
    model = train((tokens, label) for eid, tokens, label in examples if not in_holdout(eid))
    held = [(p_lore(model, tokens), label) for eid, tokens, label in examples if in_holdout(eid)]
    report = HumanReport(
        relevant=relevant,
        irrelevant=irrelevant,
        holdout_size=len(held),
        auc=auc(held),
        metrics=tuple(evaluate(held, EVAL_THRESHOLDS)),
    )
    recipe: dict[str, object] = {
        "label_scorer": HUMAN_SCORER,
        "labels_through_annotation_id": last_id,
        "holdout": f"sha256(exchange_id)[0] % {HOLDOUT_BUCKETS} == 0",
        "rules_version": rules.version,
        "gazetteer_version": gazetteer.version,
        "min_per_class": minimum,
        "features": "text tokens + __RULE_<name> rule hits",
    }
    return HumanFit(model, report, recipe)


def next_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT MAX(scorer_version) AS v FROM annotations WHERE scorer = ?", (SCORER,)
    ).fetchone()
    return int(row["v"] or 0) + 1


def score_human(
    conn: sqlite3.Connection,
    fit: HumanFit,
    at: datetime,
    rules: TriageRules = DEFAULT_RULES,
    limit: int | None = None,
) -> tuple[int, int]:
    version = next_version(conn)
    gazetteer = load_gazetteer()
    cap = -1 if limit is None else limit
    ids = [
        row["id"] for row in conn.execute("SELECT id FROM exchanges ORDER BY id LIMIT ?", (cap,))
    ]
    written = 0
    for start in range(0, len(ids), BATCH_SIZE):
        batch = ids[start : start + BATCH_SIZE]
        inputs = exchange_inputs_for_ids(conn, batch)
        for exchange_id in batch:
            one = inputs[exchange_id]
            tokens = human_features(one.messages, one.reactions, one.attachments, rules, gazetteer)
            record_annotation(
                conn,
                Annotation(
                    subject_kind="exchange",
                    subject_id=exchange_id,
                    scorer=SCORER,
                    scorer_version=version,
                    reproducibility="derived",
                    score=p_lore(fit.model, tokens),
                    recipe=fit.recipe,
                    source_ref="train-human",
                ),
                at,
            )
            written += 1
        conn.commit()
    return version, written
