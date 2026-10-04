import math
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from infovore.rows import Label
from infovore.triage.cascade import PRECISION_TARGET, EmbedStage, tune_high
from infovore.triage.embed import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    MAX_CHARS,
    RELEVANT_RECALL_FLOOR,
    Embedder,
    EmbeddingCache,
    check_counts,
    embed_exchanges,
    embed_recipe,
    fit_head,
    oof_embed,
    predict_head,
    prepare_labelled,
    stratified_folds,
)
from infovore.triage.embed_backend import load_embedder
from infovore.triage.human import trainable_labels

FOLDS: Final = 5
POOL: Final = "first"
CACHE_NAME: Final = "embed-cache.db"


def default_cache_path(db_path: Path) -> Path:
    return db_path.parent / CACHE_NAME


def irrelevant_threshold(relevant_scores: Sequence[float], recall: float) -> float:
    ordered = sorted(relevant_scores)
    return ordered[math.floor((1 - recall) * len(ordered))]


def fit_embed_stage(
    conn: sqlite3.Connection,
    embedder: Embedder,
    cache: EmbeddingCache,
    exclude_channels: frozenset[str],
    max_chars: int = MAX_CHARS,
    folds: int = FOLDS,
) -> EmbedStage:
    trainable, _ = trainable_labels(conn, exclude_channels)
    labels, vectors = prepare_labelled(conn, trainable, embedder, cache, max_chars, POOL)
    check_counts(labels, folds)
    ids = sorted(labels)
    scores = oof_embed(ids, labels, vectors, stratified_folds(labels, folds), folds)
    relevant = [scores[eid] for eid in ids if labels[eid] is Label.LORE]
    t_irrelevant = irrelevant_threshold(relevant, RELEVANT_RECALL_FLOOR)
    t_relevant = max(
        t_irrelevant,
        tune_high([(scores[eid], labels[eid] is Label.LORE) for eid in ids], PRECISION_TARGET),
    )
    head = fit_head([vectors[eid] for eid in ids], [labels[eid] is Label.LORE for eid in ids])

    def score(wanted: Sequence[int]) -> dict[int, float]:
        found = embed_exchanges(conn, wanted, embedder, cache, max_chars, pool=POOL)
        return {eid: predict_head(head, vector) for eid, vector in found.items()}

    recipe = embed_recipe(embedder, folds, max_chars, POOL)
    recipe["thresholds"] = {
        "irrelevant_below": t_irrelevant,
        "relevant_at_or_above": t_relevant,
        "relevant_recall_floor": RELEVANT_RECALL_FLOOR,
        "relevant_precision_target": PRECISION_TARGET,
    }
    recipe["labels"] = {"relevant": len(relevant), "irrelevant": len(ids) - len(relevant)}
    return EmbedStage(score, t_irrelevant, t_relevant, recipe)


def build_embed_stage(
    conn: sqlite3.Connection,
    exclude_channels: frozenset[str],
    cache_path: Path,
    model: str = DEFAULT_MODEL,
    revision: str = DEFAULT_REVISION,
    loader: Callable[[str, str], Embedder] | None = None,
) -> EmbedStage:
    trainable, _ = trainable_labels(conn, exclude_channels)
    relevant = sum(1 for label in trainable.values() if label is Label.LORE)
    irrelevant = len(trainable) - relevant
    if min(relevant, irrelevant) < FOLDS:
        return EmbedStage.abstaining(
            f"need {FOLDS} trainable labels per class (relevant={relevant} irrelevant={irrelevant})"
        )
    embedder = (loader or load_embedder)(model, revision)
    return fit_embed_stage(
        conn, embedder, EmbeddingCache(cache_path), exclude_channels, folds=FOLDS
    )
