import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from infovore.reputation.stats import Interval, auc_interval, rank_average
from infovore.rows import Label
from infovore.triage.embed import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    MAX_CHARS,
    Embedder,
    EmbeddingCache,
    oof_embed,
    prepare_labelled,
    stratified_folds,
)
from infovore.triage.embed_backend import load_embedder
from infovore.triage.embed_stage import FOLDS, POOL, fit_embed_stage
from infovore.triage.human import trainable_labels


@dataclass(frozen=True)
class Comparison:
    embed: Interval
    reputation: Interval
    combined: Interval


@dataclass(frozen=True)
class EmbedScores:
    held_out: dict[int, float]
    out_of_fold: dict[int, float]


def compare(
    embed: Mapping[int, float],
    reputation: Mapping[int, float],
    labels: Mapping[int, Label],
    seed: int,
) -> Comparison:
    ids = sorted(eid for eid in labels if eid in embed and eid in reputation)
    first = [embed[eid] for eid in ids]
    second = [reputation[eid] for eid in ids]
    blend = rank_average(first, second)
    marks = [labels[eid] for eid in ids]
    return Comparison(
        auc_interval(list(zip(first, marks, strict=True)), seed),
        auc_interval(list(zip(second, marks, strict=True)), seed),
        auc_interval(list(zip(blend, marks, strict=True)), seed),
    )


def embed_scores(
    conn: sqlite3.Connection,
    exclude_channels: frozenset[str],
    cache_path: Path,
    held_out: Sequence[int],
    loader: Callable[[str, str], Embedder] | None = None,
) -> EmbedScores | None:
    trainable, _ = trainable_labels(conn, exclude_channels)
    relevant = sum(1 for label in trainable.values() if label is Label.LORE)
    if min(relevant, len(trainable) - relevant) < FOLDS:
        return None
    embedder = (loader or load_embedder)(DEFAULT_MODEL, DEFAULT_REVISION)
    cache = EmbeddingCache(cache_path)
    stage = fit_embed_stage(conn, embedder, cache, exclude_channels)
    labels, vectors = prepare_labelled(conn, trainable, embedder, cache, MAX_CHARS, POOL)
    folds = stratified_folds(labels, FOLDS)
    out_of_fold = oof_embed(sorted(labels), labels, vectors, folds, FOLDS)
    return EmbedScores(stage.score(sorted(held_out)), out_of_fold)
