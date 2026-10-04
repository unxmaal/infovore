import hashlib
import sqlite3
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from operator import mul
from pathlib import Path
from typing import Final, Protocol

from infovore.config import ConfigError
from infovore.db.batch import BATCH_SIZE, exchange_inputs_for_ids
from infovore.db.channel_filter import excluded_exchange_ids
from infovore.db.exchange_text import text_message_counts
from infovore.rows import Label, MessageRow
from infovore.triage.bayes import auc, candidate_thresholds, evaluate, p_lore, train
from infovore.triage.cascade import decide_lexicon, tune_high, tuning_samples
from infovore.triage.human import human_features, trainable_labels, training_labels
from infovore.triage.lexicon import Lexicon, load_lexicon, score_lexicon
from infovore.triage.logistic import sigmoid

SCORER: Final = "p_relevant_embed"
DEFAULT_MODEL: Final = "BAAI/bge-small-en-v1.5"
DEFAULT_REVISION: Final = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MAX_CHARS: Final = 2000
EMBED_BATCH: Final = 64
RELEVANT_RECALL_FLOOR: Final = 0.95
FOLD_SEED: Final = "embed-cv-v1"
HEAD_ITERATIONS: Final = 300
HEAD_LEARNING_RATE: Final = 0.01
HEAD_L2: Final = 1.0


class NotEnoughLabelsError(ConfigError):
    pass


class Embedder(Protocol):
    model_id: str
    revision: str
    max_tokens: int

    def token_count(self, text: str) -> int: ...

    def split(self, text: str, limit: int) -> list[str]: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


POOLS: Final = ("first", "mean", "max")


def message_parts(messages: Sequence[MessageRow]) -> list[str]:
    return [message.content for message in messages if message.content.strip()]


def render_exchange(messages: Sequence[MessageRow], max_chars: int) -> str:
    return "\n".join(message_parts(messages))[:max_chars]


def window_parts(parts: Sequence[str], embedder: Embedder) -> list[str]:
    limit = embedder.max_tokens
    windows: list[str] = []
    current: list[str] = []
    used = 0
    for part in parts:
        size = embedder.token_count(part)
        pieces = [part] if size <= limit else embedder.split(part, limit)
        for piece in pieces:
            count = size if len(pieces) == 1 else embedder.token_count(piece)
            if current and used + count > limit:
                windows.append("\n".join(current))
                current, used = [], 0
            current.append(piece)
            used += count
    if current:
        windows.append("\n".join(current))
    return windows


def pool_vectors(vectors: Sequence[Sequence[float]], pool: str) -> list[float]:
    if pool == "first":
        return list(vectors[0])
    columns = list(zip(*vectors, strict=True))
    if pool == "mean":
        return [sum(column) / len(vectors) for column in columns]
    if pool == "max":
        return [max(column) for column in columns]
    raise ValueError(f"unknown pool {pool!r}; expected one of {', '.join(POOLS)}")


def cache_model(embedder: Embedder, pool: str) -> str:
    if pool == "first":
        return embedder.model_id
    return f"{embedder.model_id}#windows{embedder.max_tokens}-{pool}"


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (model TEXT, revision TEXT,"
            " exchange_id INTEGER, content_hash TEXT, vector BLOB,"
            " PRIMARY KEY (model, revision, exchange_id, content_hash))"
        )

    def get(
        self, model: str, revision: str, exchange_id: int, content_hash: str
    ) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE model = ? AND revision = ?"
            " AND exchange_id = ? AND content_hash = ?",
            (model, revision, exchange_id, content_hash),
        ).fetchone()
        if row is None:
            return None
        return _unpack(row[0])

    def put(
        self,
        model: str,
        revision: str,
        exchange_id: int,
        content_hash: str,
        vector: Sequence[float],
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
            (model, revision, exchange_id, content_hash, array("f", vector).tobytes()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _unpack(blob: bytes) -> list[float]:
    vector = array("f")
    vector.frombytes(blob)
    return vector.tolist()


def embed_exchanges(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    embedder: Embedder,
    cache: EmbeddingCache,
    max_chars: int,
    batch_size: int = EMBED_BATCH,
    pool: str = "first",
) -> dict[int, list[float]]:
    pool_vectors([[0.0]], pool)
    model = cache_model(embedder, pool)
    found: dict[int, list[float]] = {}
    pending: list[tuple[int, str, list[str]]] = []
    for start in range(0, len(ids), BATCH_SIZE):
        chunk = ids[start : start + BATCH_SIZE]
        inputs = exchange_inputs_for_ids(conn, chunk)
        for eid in chunk:
            parts = message_parts(inputs[eid].messages)
            if not parts:
                continue
            if pool == "first":
                text = "\n".join(parts)[:max_chars]
                windows = [text]
            else:
                text = "\n".join(parts)
                windows = window_parts(parts, embedder)
            digest = hashlib.sha256(text.encode()).hexdigest()
            cached = cache.get(model, embedder.revision, eid, digest)
            if cached is None:
                pending.append((eid, digest, windows))
            else:
                found[eid] = cached
    flat = [(index, window) for index, (_, _, windows) in enumerate(pending) for window in windows]
    vectors: list[list[float]] = []
    for start in range(0, len(flat), batch_size):
        vectors.extend(embedder.embed([window for _, window in flat[start : start + batch_size]]))
    grouped: dict[int, list[list[float]]] = {}
    for (index, _), vector in zip(flat, vectors, strict=True):
        grouped.setdefault(index, []).append(vector)
    for index, (eid, digest, _) in enumerate(pending):
        pooled = array("f", pool_vectors(grouped[index], pool))
        cache.put(model, embedder.revision, eid, digest, pooled)
        found[eid] = pooled.tolist()
    return found


def stratified_folds(labels: dict[int, Label], folds: int) -> dict[int, int]:
    assigned: dict[int, int] = {}
    for label in (Label.LORE, Label.NOISE):
        members = sorted(
            (eid for eid, got in labels.items() if got is label),
            key=lambda eid: hashlib.sha256(f"{FOLD_SEED}:{eid}".encode()).digest(),
        )
        for position, eid in enumerate(members):
            assigned[eid] = position % folds
    return assigned


@dataclass(frozen=True)
class Head:
    bias: float
    weights: list[float]
    mean: list[float]
    scale: list[float]


def fit_head(
    xs: Sequence[Sequence[float]],
    ys: Sequence[bool],
    iterations: int = HEAD_ITERATIONS,
    learning_rate: float = HEAD_LEARNING_RATE,
    l2: float = HEAD_L2,
) -> Head:
    total = len(xs)
    if total == 0:
        return Head(0.0, [], [], [])
    columns = list(zip(*xs, strict=True))
    mean = [sum(column) / total for column in columns]
    scale = [
        (sum((value - m) ** 2 for value in column) / total) ** 0.5 or 1.0
        for column, m in zip(columns, mean, strict=True)
    ]
    rows = [_standardize(x, mean, scale) for x in xs]
    zcols = list(zip(*rows, strict=True))
    targets = [1.0 if y else 0.0 for y in ys]
    weights = [0.0] * len(mean)
    bias = 0.0
    for _ in range(iterations):
        errors = [
            sigmoid(bias + sum(map(mul, weights, row))) - target
            for row, target in zip(rows, targets, strict=True)
        ]
        weights = [
            w - learning_rate * (sum(map(mul, errors, column)) / total + l2 / total * w)
            for w, column in zip(weights, zcols, strict=True)
        ]
        bias -= learning_rate * sum(errors) / total
    return Head(bias, weights, mean, scale)


def _standardize(x: Sequence[float], mean: Sequence[float], scale: Sequence[float]) -> list[float]:
    return [(value - m) / s for value, m, s in zip(x, mean, scale, strict=True)]


def predict_head(head: Head, x: Sequence[float]) -> float:
    row = _standardize(x, head.mean, head.scale)
    return sigmoid(head.bias + sum(map(mul, head.weights, row)))


@dataclass(frozen=True)
class Summary:
    relevant: int
    irrelevant: int
    auc: float | None
    threshold: float | None
    precision: float
    recall: float
    f1: float
    irrelevant_recall: float | None


def summarize(
    scored: Sequence[tuple[float, Label]], relevant_recall: float = RELEVANT_RECALL_FLOOR
) -> Summary:
    relevant = sum(1 for _, label in scored if label is Label.LORE)
    irrelevant = len(scored) - relevant
    if relevant == 0 or irrelevant == 0:
        return Summary(relevant, irrelevant, None, None, 0.0, 0.0, 0.0, None)
    table = evaluate(scored, candidate_thresholds(scored))
    best = max(table, key=lambda row: row.f1)
    qualifying = [row for row in table if row.recall >= relevant_recall]
    floor = max(qualifying, key=lambda row: row.threshold)
    return Summary(
        relevant,
        irrelevant,
        auc(scored),
        best.threshold,
        best.precision,
        best.recall,
        best.f1,
        floor.tn / irrelevant,
    )


def residue_ids(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    lexicon: Lexicon,
    t_high: float,
    exclude_channels: frozenset[str],
) -> set[int]:
    denied = excluded_exchange_ids(conn, exclude_channels)
    live = [eid for eid in ids if eid not in denied]
    text = text_message_counts(conn, live)
    inputs = exchange_inputs_for_ids(conn, [eid for eid in live if text[eid]])
    return {
        eid
        for eid in live
        if text[eid]
        and decide_lexicon(score_lexicon(lexicon, inputs[eid].messages), t_high) is None
    }


@dataclass(frozen=True)
class Comparison:
    bayes: Summary
    embed: Summary


@dataclass(frozen=True)
class CrossValidation:
    folds: int
    all_labels: Comparison
    residue: Comparison | None
    recipe: dict[str, object]


def embed_recipe(embedder: Embedder, folds: int, max_chars: int, pool: str) -> dict[str, object]:
    return {
        "model": embedder.model_id,
        "revision": embedder.revision,
        "pool": pool,
        "max_chars": max_chars,
        "truncation": (
            f"rendered exchange text cut to first {max_chars} characters"
            if pool == "first"
            else "none"
        ),
        "windows": (
            "one window"
            if pool == "first"
            else f"messages packed into windows of <= {embedder.max_tokens} tokens;"
            " oversized message split by tokenizer; window vectors pooled"
        ),
        "text": "non-empty message contents joined by newline, in order",
        "head": {
            "kind": "standardized L2 logistic regression, full-batch gradient descent",
            "iterations": HEAD_ITERATIONS,
            "learning_rate": HEAD_LEARNING_RATE,
            "l2": HEAD_L2,
        },
        "folds": folds,
        "fold_seed": FOLD_SEED,
    }


def prepare_labelled(
    conn: sqlite3.Connection,
    labels: dict[int, Label],
    embedder: Embedder,
    cache: EmbeddingCache,
    max_chars: int,
    pool: str,
) -> tuple[dict[int, Label], dict[int, list[float]]]:
    vectors = embed_exchanges(conn, sorted(labels), embedder, cache, max_chars, pool=pool)
    return {eid: label for eid, label in labels.items() if eid in vectors}, vectors


def check_counts(labels: dict[int, Label], folds: int) -> None:
    relevant = sum(1 for label in labels.values() if label is Label.LORE)
    irrelevant = len(labels) - relevant
    if relevant < folds or irrelevant < folds:
        raise NotEnoughLabelsError(
            f"need at least {folds} relevant and {folds} irrelevant labelled exchanges"
            f" with text (have relevant={relevant} irrelevant={irrelevant})"
        )


def _held_out_folds(
    ids: Sequence[int], assigned: dict[int, int], folds: int
) -> list[tuple[list[int], list[int]]]:
    return [
        (
            [eid for eid in ids if assigned[eid] != fold],
            [eid for eid in ids if assigned[eid] == fold],
        )
        for fold in range(folds)
    ]


def oof_embed(
    ids: Sequence[int],
    labels: dict[int, Label],
    vectors: dict[int, list[float]],
    assigned: dict[int, int],
    folds: int,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for fit_ids, test_ids in _held_out_folds(ids, assigned, folds):
        head = fit_head(
            [vectors[eid] for eid in fit_ids], [labels[eid] is Label.LORE for eid in fit_ids]
        )
        for eid in test_ids:
            scores[eid] = predict_head(head, vectors[eid])
    return scores


def _oof_bayes(
    ids: Sequence[int],
    labels: dict[int, Label],
    tokens: dict[int, frozenset[str]],
    assigned: dict[int, int],
    folds: int,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for fit_ids, test_ids in _held_out_folds(ids, assigned, folds):
        model = train((tokens[eid], labels[eid]) for eid in fit_ids)
        for eid in test_ids:
            scores[eid] = p_lore(model, tokens[eid])
    return scores


def _tokens(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, frozenset[str]]:
    found: dict[int, frozenset[str]] = {}
    for start in range(0, len(ids), BATCH_SIZE):
        chunk = ids[start : start + BATCH_SIZE]
        for eid, one in exchange_inputs_for_ids(conn, chunk).items():
            found[eid] = human_features(one.messages, one.reactions, one.attachments)
    return found


def cross_validate(
    conn: sqlite3.Connection,
    embedder: Embedder,
    cache: EmbeddingCache,
    folds: int,
    max_chars: int,
    residue: bool = False,
    exclude_channels: frozenset[str] = frozenset(),
    pool: str = "first",
) -> CrossValidation:
    trainable, _ = trainable_labels(conn, exclude_channels)
    labels, vectors = prepare_labelled(conn, trainable, embedder, cache, max_chars, pool)
    check_counts(labels, folds)
    assigned = stratified_folds(labels, folds)
    ids = sorted(labels)
    tokens = _tokens(conn, ids)

    def compare(subset: Sequence[int]) -> Comparison:
        bayes = _oof_bayes(subset, labels, tokens, assigned, folds)
        embed = oof_embed(subset, labels, vectors, assigned, folds)
        return Comparison(
            summarize([(bayes[eid], labels[eid]) for eid in subset]),
            summarize([(embed[eid], labels[eid]) for eid in subset]),
        )

    lexicon = load_lexicon(conn)
    residue_comparison = None
    if residue:
        t_high = tune_high(tuning_samples(conn, lexicon, exclude_channels))
        subset = sorted(residue_ids(conn, ids, lexicon, t_high, exclude_channels))
        residue_comparison = compare(subset)
    return CrossValidation(
        folds, compare(ids), residue_comparison, embed_recipe(embedder, folds, max_chars, pool)
    )


def embed_scores(
    conn: sqlite3.Connection,
    embedder: Embedder,
    cache: EmbeddingCache,
    folds: int,
    max_chars: int,
    exclude_channels: frozenset[str] = frozenset(),
    pool: str = "first",
) -> tuple[dict[int, float], dict[str, object]]:
    trainable, _ = trainable_labels(conn, exclude_channels)
    labels, vectors = prepare_labelled(conn, trainable, embedder, cache, max_chars, pool)
    check_counts(labels, folds)
    ids = sorted(labels)
    scores = oof_embed(ids, labels, vectors, stratified_folds(labels, folds), folds)
    held_out = {
        eid: label
        for eid, label in training_labels(conn, exclude_channels=exclude_channels)[0].items()
        if eid not in trainable
    }
    _, held_vectors = prepare_labelled(conn, held_out, embedder, cache, max_chars, pool)
    head = fit_head([vectors[eid] for eid in ids], [labels[eid] is Label.LORE for eid in ids])
    scores.update({eid: predict_head(head, vector) for eid, vector in held_vectors.items()})
    recipe = embed_recipe(embedder, folds, max_chars, pool)
    recipe["scores"] = (
        "out-of-fold for trainable labels; held-out labels from a head fit on all trainable"
    )
    return scores, recipe
