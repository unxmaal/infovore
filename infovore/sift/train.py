"""The message-level trash classifier (issue #128 PR 3).

Trains a naive Bayes model over `message_labels` (human + citation),
reusing `infovore.triage.bayes`'s Robinson/Fisher combining and
sha256-based holdout split unchanged, rather than duplicating that engine
for messages.

**Reuse mapping.** `infovore.triage.bayes.train`/`p_lore`/`evaluate`/`auc`
are written generically against `infovore.rows.Label.LORE`/`.NOISE` (the
"positive"/"negative" class a score predicts) -- they never look at an
exchange specifically. This module scores `p_trash`, so `MessageLabel.TRASH`
is mapped onto `Label.LORE` (the positive class the score predicts) and
`MessageLabel.KEEP` onto `Label.NOISE` (`_bayes_label`, used everywhere a
label crosses into `bayes.py`). `p_trash` for a token set is then exactly
`bayes.p_lore(model, tokens)` -- no inversion anywhere. One consequence,
purely a bookkeeping artifact of this reuse: `Model.lore_documents` and the
"lore" side of `Model.counts` count *trash* documents throughout this
module; `message_model`/`message_tokens` (see the migration) use their own
`trash_count`/`keep_count` column names so nothing about the stored schema
is confusing, even though the in-memory `Model` field names are inherited
as-is from `bayes.py`.

**Weighting.** A human sift decision is direct, deliberate evidence; a
citation label is a side effect of what one extraction run happened to
need -- an uncited message in a successfully extracted exchange is not
necessarily trash, just not cited by any claim that run made. Naive Bayes
here counts *documents*, so the simplest way to weight a class of examples
without touching the counting math in `bayes.train` is to repeat each
non-holdout human example `human_weight` times (default `DEFAULT_HUMAN_WEIGHT
= 5`) before handing the list to `train()`. This never touches the holdout
set: every holdout example is scored exactly once, unweighted, so the
reported metrics are an honest read of held-out performance regardless of
`human_weight`.
"""

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from infovore.db.batch import BATCH_SIZE, SQLITE_MAX_VARIABLES
from infovore.db.codec import to_db_time
from infovore.db.connection import transaction
from infovore.db.message_labels import effective_message_labels_with_source
from infovore.db.raw import messages_by_ids
from infovore.rows import MessageLabel, MessageLabelSource
from infovore.sift.features import message_features
from infovore.timing import Clock
from infovore.triage.bayes import (
    Label,
    Metrics,
    Model,
    auc,
    evaluate,
    in_holdout,
    p_lore,
    token_probability,
    train,
)
from infovore.triage.parallel import ChunkPool

MIN_LABELS_PER_CLASS = 10
DEFAULT_HUMAN_WEIGHT = 5
EVAL_THRESHOLDS = tuple(round(i / 10, 1) for i in range(1, 10))
DEFAULT_CONFUSION_THRESHOLD = 0.5
TOP_TOKENS_LIMIT = 15
DISCARD_THRESHOLDS = (0.5, 0.7, 0.9)

__all__ = [
    "DEFAULT_CONFUSION_THRESHOLD",
    "DEFAULT_HUMAN_WEIGHT",
    "DISCARD_THRESHOLDS",
    "DiscardRow",
    "InsufficientLabelsError",
    "MessageExample",
    "MessageTokenInfo",
    "MessageTrainReport",
    "build_message_examples",
    "load_latest_message_model",
    "score_all",
    "score_stale",
    "train_and_store",
]


class InsufficientLabelsError(Exception):
    def __init__(self, keep_count: int, trash_count: int) -> None:
        super().__init__(
            f"need at least {MIN_LABELS_PER_CLASS} labels of each class to train"
            f" the message classifier (have keep={keep_count} trash={trash_count})"
        )
        self.keep_count = keep_count
        self.trash_count = trash_count


@dataclass(frozen=True)
class MessageExample:
    message_id: int
    channel_id: int
    tokens: frozenset[str]
    label: MessageLabel
    source: MessageLabelSource


@dataclass(frozen=True)
class MessageTokenInfo:
    token: str
    probability: float
    trash_count: int
    keep_count: int


@dataclass(frozen=True)
class DiscardRow:
    threshold: float
    human_keep_discarded: float | None
    citation_keep_discarded: float | None


@dataclass(frozen=True)
class MessageTrainReport:
    version: int
    labels_used: int
    holdout_size: int
    human_weight: int
    overall_metrics: tuple[Metrics, ...]
    confusion: Metrics
    overall_auc: float | None
    human_metrics: tuple[Metrics, ...]
    human_auc: float | None
    citation_metrics: tuple[Metrics, ...]
    citation_auc: float | None
    channel_stats: dict[int, Metrics]
    discard_pile: tuple[DiscardRow, ...]
    top_tokens: tuple[MessageTokenInfo, ...]


def _bayes_label(label: MessageLabel) -> Label:
    return Label.LORE if label is MessageLabel.TRASH else Label.NOISE


def build_message_examples(conn: sqlite3.Connection) -> list[MessageExample]:
    labeled = effective_message_labels_with_source(conn)
    ids = sorted(labeled)
    examples: list[MessageExample] = []
    for start in range(0, len(ids), SQLITE_MAX_VARIABLES):
        chunk = ids[start : start + SQLITE_MAX_VARIABLES]
        for message in messages_by_ids(conn, chunk):
            label, source = labeled[message.id]
            tokens = message_features(message, message.channel_id)
            examples.append(MessageExample(message.id, message.channel_id, tokens, label, source))
    return examples


def _top_message_tokens(model: Model) -> tuple[MessageTokenInfo, ...]:
    infos = [
        MessageTokenInfo(token, token_probability(model, token), trash, keep)
        for token, (trash, keep) in model.counts.items()
    ]
    infos.sort(key=lambda info: abs(info.probability - 0.5), reverse=True)
    return tuple(infos[:TOP_TOKENS_LIMIT])


def _share_at_or_above(scores: Sequence[float], threshold: float) -> float | None:
    if not scores:
        return None
    return sum(1 for score in scores if score >= threshold) / len(scores)


def train_and_store(
    conn: sqlite3.Connection,
    clock: Clock,
    confusion_threshold: float = DEFAULT_CONFUSION_THRESHOLD,
    human_weight: int = DEFAULT_HUMAN_WEIGHT,
) -> MessageTrainReport:
    examples = build_message_examples(conn)
    keep_count = sum(1 for example in examples if example.label is MessageLabel.KEEP)
    trash_count = sum(1 for example in examples if example.label is MessageLabel.TRASH)
    if keep_count < MIN_LABELS_PER_CLASS or trash_count < MIN_LABELS_PER_CLASS:
        raise InsufficientLabelsError(keep_count, trash_count)

    holdout_examples = [example for example in examples if in_holdout(example.message_id)]
    train_pairs: list[tuple[frozenset[str], Label]] = []
    for example in examples:
        if in_holdout(example.message_id):
            continue
        repeats = human_weight if example.source is MessageLabelSource.HUMAN else 1
        generic = _bayes_label(example.label)
        train_pairs.extend((example.tokens, generic) for _ in range(repeats))

    model = train(train_pairs)

    holdout_scored = [(p_lore(model, example.tokens), example) for example in holdout_examples]

    overall_pairs = [(score, _bayes_label(example.label)) for score, example in holdout_scored]
    overall_metrics = tuple(evaluate(overall_pairs, EVAL_THRESHOLDS))
    if confusion_threshold in EVAL_THRESHOLDS:
        confusion = next(
            metric for metric in overall_metrics if metric.threshold == confusion_threshold
        )
    else:
        confusion = evaluate(overall_pairs, [confusion_threshold])[0]
    overall_auc = auc(overall_pairs)

    channel_groups: dict[int, list[tuple[float, Label]]] = {}
    for score, example in holdout_scored:
        channel_groups.setdefault(example.channel_id, []).append(
            (score, _bayes_label(example.label))
        )
    channel_stats = {
        channel_id: evaluate(pairs, [confusion_threshold])[0]
        for channel_id, pairs in channel_groups.items()
    }

    human_pairs = [
        (score, _bayes_label(example.label))
        for score, example in holdout_scored
        if example.source is MessageLabelSource.HUMAN
    ]
    citation_pairs = [
        (score, _bayes_label(example.label))
        for score, example in holdout_scored
        if example.source is MessageLabelSource.CITATION
    ]
    human_metrics = tuple(evaluate(human_pairs, EVAL_THRESHOLDS))
    citation_metrics = tuple(evaluate(citation_pairs, EVAL_THRESHOLDS))
    human_auc = auc(human_pairs)
    citation_auc = auc(citation_pairs)

    human_keep_scores = [
        score
        for score, example in holdout_scored
        if example.label is MessageLabel.KEEP and example.source is MessageLabelSource.HUMAN
    ]
    citation_keep_scores = [
        score
        for score, example in holdout_scored
        if example.label is MessageLabel.KEEP and example.source is MessageLabelSource.CITATION
    ]
    discard_pile = tuple(
        DiscardRow(
            threshold=threshold,
            human_keep_discarded=_share_at_or_above(human_keep_scores, threshold),
            citation_keep_discarded=_share_at_or_above(citation_keep_scores, threshold),
        )
        for threshold in DISCARD_THRESHOLDS
    )

    params = {
        "trash_documents": model.lore_documents,
        "keep_documents": model.noise_documents,
        "human_weight": human_weight,
    }
    with transaction(conn):
        cursor = conn.execute(
            "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json)"
            " VALUES (?, ?, ?, ?)",
            (to_db_time(clock.now()), len(examples), len(holdout_examples), json.dumps(params)),
        )
        version = int(cursor.lastrowid or 0)
        for token, (trash, keep) in model.counts.items():
            conn.execute(
                "INSERT INTO message_tokens (model_version, token, trash_count, keep_count)"
                " VALUES (?, ?, ?, ?)",
                (version, token, trash, keep),
            )

    return MessageTrainReport(
        version=version,
        labels_used=len(examples),
        holdout_size=len(holdout_examples),
        human_weight=human_weight,
        overall_metrics=overall_metrics,
        confusion=confusion,
        overall_auc=overall_auc,
        human_metrics=human_metrics,
        human_auc=human_auc,
        citation_metrics=citation_metrics,
        citation_auc=citation_auc,
        channel_stats=channel_stats,
        discard_pile=discard_pile,
        top_tokens=_top_message_tokens(model),
    )


def load_latest_message_model(conn: sqlite3.Connection) -> tuple[int, Model] | None:
    row = conn.execute(
        "SELECT version, params_json FROM message_model ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    params = json.loads(row["params_json"])
    counts: dict[str, tuple[int, int]] = {}
    for token_row in conn.execute(
        "SELECT token, trash_count, keep_count FROM message_tokens WHERE model_version = ?",
        (row["version"],),
    ):
        counts[token_row["token"]] = (token_row["trash_count"], token_row["keep_count"])
    model = Model(
        lore_documents=params["trash_documents"],
        noise_documents=params["keep_documents"],
        counts=counts,
    )
    return int(row["version"]), model


# Module-level (spawn-safe) worker state for `ChunkPool`, mirroring
# `infovore.triage.train`'s `_worker_model`/`_init_p_lore_worker` pattern
# (issue #113): the model is sent to each worker once, via the pool's
# initializer, not per task.
_worker_model: Model | None = None


def _init_p_trash_worker(model: Model) -> None:
    global _worker_model
    _worker_model = model


def _p_trash_worker_chunk(items: list[tuple[int, frozenset[str]]]) -> list[tuple[int, float]]:
    model = _worker_model
    assert model is not None, "ChunkPool must call _init_p_trash_worker before this"
    return [(message_id, p_lore(model, tokens)) for message_id, tokens in items]


def _score_rows(
    conn: sqlite3.Connection,
    model: Model,
    model_version: int,
    rows: list[sqlite3.Row],
    workers: int = 1,
) -> int:
    count = 0
    with ChunkPool(_p_trash_worker_chunk, workers, _init_p_trash_worker, (model,)) as pool:
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            message_rows = messages_by_ids(conn, [row["id"] for row in batch])
            by_id = {message.id: message for message in message_rows}
            items = [
                (row["id"], message_features(by_id[row["id"]], row["channel_id"]))
                for row in batch
                if row["id"] in by_id
            ]
            updates = [
                (score, model_version, message_id)
                for chunk_result in pool.map_chunks(items)
                for message_id, score in chunk_result
            ]
            with transaction(conn):
                conn.executemany(
                    "UPDATE messages SET p_trash = ?, p_trash_model = ? WHERE id = ?", updates
                )
            count += len(updates)
    return count


def score_all(conn: sqlite3.Connection, model: Model, model_version: int, workers: int = 1) -> int:
    """Score every message that belongs to an exchange (issue #128 PR 3
    scores messages *in exchanges* only -- a message never grouped into an
    exchange was never a candidate for extraction and isn't part of this
    corpus)."""
    rows = conn.execute(
        "SELECT DISTINCT m.id AS id, m.channel_id AS channel_id"
        " FROM messages m JOIN exchange_messages em ON em.message_id = m.id"
        " ORDER BY m.id"
    ).fetchall()
    return _score_rows(conn, model, model_version, rows, workers)


def score_stale(
    conn: sqlite3.Connection, model: Model, model_version: int, workers: int = 1
) -> int:
    rows = conn.execute(
        "SELECT DISTINCT m.id AS id, m.channel_id AS channel_id"
        " FROM messages m JOIN exchange_messages em ON em.message_id = m.id"
        " WHERE m.p_trash_model IS NULL OR m.p_trash_model != ?"
        " ORDER BY m.id",
        (model_version,),
    ).fetchall()
    return _score_rows(conn, model, model_version, rows, workers)
