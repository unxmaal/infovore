import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from infovore.db.batch import BATCH_SIZE, exchange_inputs_for_ids
from infovore.db.codec import to_db_time
from infovore.db.connection import transaction
from infovore.db.exchanges import get_exchange
from infovore.db.labels import effective_labels
from infovore.rows import Label
from infovore.timing import Clock
from infovore.triage.bayes import (
    Metrics,
    Model,
    candidate_thresholds,
    evaluate,
    features,
    format_threshold,
    in_holdout,
    p_lore,
    recommend_threshold,
    token_probability,
    train,
)
from infovore.triage.rules import DEFAULT_RULES, TriageRules

MIN_LABELS_PER_CLASS = 10
EVAL_THRESHOLDS = tuple(round(i / 10, 1) for i in range(1, 10))
DEFAULT_CONFUSION_THRESHOLD = 0.5
TOP_TOKENS_LIMIT = 15
RECALL_TARGETS: tuple[float, ...] = (0.95, 0.9, 0.8, 0.7, 0.5)


class InsufficientLabelsError(Exception):
    def __init__(self, lore_count: int, noise_count: int) -> None:
        super().__init__(
            f"need at least {MIN_LABELS_PER_CLASS} labels of each class to train"
            f" the triage classifier (have lore={lore_count} noise={noise_count})"
        )
        self.lore_count = lore_count
        self.noise_count = noise_count


class NoTrainedModelError(Exception):
    pass


@dataclass(frozen=True)
class Example:
    exchange_id: int
    channel_id: int
    tokens: frozenset[str]
    label: Label


@dataclass(frozen=True)
class TokenInfo:
    token: str
    probability: float
    lore_count: int
    noise_count: int


@dataclass(frozen=True)
class TrainReport:
    version: int
    labels_used: int
    holdout_size: int
    metrics: tuple[Metrics, ...]
    confusion: Metrics
    channel_stats: dict[int, Metrics]
    top_tokens: tuple[TokenInfo, ...]


def build_examples(conn: sqlite3.Connection, rules: TriageRules = DEFAULT_RULES) -> list[Example]:
    labels = effective_labels(conn)
    exchange_ids = sorted(labels)
    channel_ids: dict[int, int] = {}
    for exchange_id in exchange_ids:
        exchange = get_exchange(conn, exchange_id)
        assert exchange is not None
        channel_ids[exchange_id] = exchange.channel_id

    examples: list[Example] = []
    for start in range(0, len(exchange_ids), BATCH_SIZE):
        batch = exchange_ids[start : start + BATCH_SIZE]
        inputs = exchange_inputs_for_ids(conn, batch)
        for exchange_id in batch:
            channel_id = channel_ids[exchange_id]
            one = inputs[exchange_id]
            tokens = features(one.messages, channel_id, one.reactions, one.attachments, rules)
            examples.append(Example(exchange_id, channel_id, tokens, labels[exchange_id]))
    return examples


def _top_tokens(model: Model) -> tuple[TokenInfo, ...]:
    infos = [
        TokenInfo(token, token_probability(model, token), lore, noise)
        for token, (lore, noise) in model.counts.items()
    ]
    infos.sort(key=lambda info: abs(info.probability - 0.5), reverse=True)
    return tuple(infos[:TOP_TOKENS_LIMIT])


def train_and_store(
    conn: sqlite3.Connection,
    clock: Clock,
    confusion_threshold: float = DEFAULT_CONFUSION_THRESHOLD,
    rules: TriageRules = DEFAULT_RULES,
) -> TrainReport:
    examples = build_examples(conn, rules)
    lore_count = sum(1 for example in examples if example.label is Label.LORE)
    noise_count = sum(1 for example in examples if example.label is Label.NOISE)
    if lore_count < MIN_LABELS_PER_CLASS or noise_count < MIN_LABELS_PER_CLASS:
        raise InsufficientLabelsError(lore_count, noise_count)

    train_examples = [
        (example.tokens, example.label)
        for example in examples
        if not in_holdout(example.exchange_id)
    ]
    holdout_examples = [example for example in examples if in_holdout(example.exchange_id)]

    model = train(train_examples)

    holdout_scored = [
        (p_lore(model, example.tokens), example.label) for example in holdout_examples
    ]
    metrics = tuple(evaluate(holdout_scored, EVAL_THRESHOLDS))
    confusion_thresholds = [confusion_threshold]
    if confusion_threshold not in EVAL_THRESHOLDS:
        confusion = evaluate(holdout_scored, confusion_thresholds)[0]
    else:
        confusion = next(metric for metric in metrics if metric.threshold == confusion_threshold)

    scored_by_channel: dict[int, list[tuple[float, Label]]] = {}
    for example, scored in zip(holdout_examples, holdout_scored, strict=True):
        scored_by_channel.setdefault(example.channel_id, []).append(scored)
    channel_stats = {
        channel_id: evaluate(scored, [confusion_threshold])[0]
        for channel_id, scored in scored_by_channel.items()
    }

    params = {
        "lore_documents": model.lore_documents,
        "noise_documents": model.noise_documents,
        "rules_version": rules.version,
    }
    with transaction(conn):
        cursor = conn.execute(
            "INSERT INTO triage_model (trained_at, labels_used, holdout_size, params_json)"
            " VALUES (?, ?, ?, ?)",
            (
                to_db_time(clock.now()),
                len(examples),
                len(holdout_examples),
                json.dumps(params),
            ),
        )
        version = int(cursor.lastrowid or 0)
        for token, (lore, noise) in model.counts.items():
            conn.execute(
                "INSERT INTO triage_tokens (model_version, token, lore_count, noise_count)"
                " VALUES (?, ?, ?, ?)",
                (version, token, lore, noise),
            )

    return TrainReport(
        version=version,
        labels_used=len(examples),
        holdout_size=len(holdout_examples),
        metrics=metrics,
        confusion=confusion,
        channel_stats=channel_stats,
        top_tokens=_top_tokens(model),
    )


def load_latest_model(conn: sqlite3.Connection) -> tuple[int, Model] | None:
    row = conn.execute(
        "SELECT version, params_json FROM triage_model ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    params = json.loads(row["params_json"])
    counts: dict[str, tuple[int, int]] = {}
    for token_row in conn.execute(
        "SELECT token, lore_count, noise_count FROM triage_tokens WHERE model_version = ?",
        (row["version"],),
    ):
        counts[token_row["token"]] = (token_row["lore_count"], token_row["noise_count"])
    model = Model(
        lore_documents=params["lore_documents"],
        noise_documents=params["noise_documents"],
        counts=counts,
    )
    return int(row["version"]), model


def _score_rows(
    conn: sqlite3.Connection,
    model: Model,
    model_version: int,
    rows: list[sqlite3.Row],
    rules: TriageRules = DEFAULT_RULES,
) -> int:
    count = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        inputs = exchange_inputs_for_ids(conn, [row["id"] for row in batch])
        updates: list[tuple[float, int, int]] = []
        for row in batch:
            one = inputs[row["id"]]
            tokens = features(
                one.messages, row["channel_id"], one.reactions, one.attachments, rules
            )
            score = p_lore(model, tokens)
            updates.append((score, model_version, row["id"]))
        with transaction(conn):
            conn.executemany(
                "UPDATE exchanges SET p_lore = ?, p_lore_model = ? WHERE id = ?", updates
            )
        count += len(batch)
    return count


def score_all(
    conn: sqlite3.Connection, model: Model, model_version: int, rules: TriageRules = DEFAULT_RULES
) -> int:
    rows = conn.execute("SELECT id, channel_id FROM exchanges ORDER BY id").fetchall()
    return _score_rows(conn, model, model_version, rows, rules)


def score_stale(
    conn: sqlite3.Connection, model: Model, model_version: int, rules: TriageRules = DEFAULT_RULES
) -> int:
    rows = conn.execute(
        "SELECT id, channel_id FROM exchanges"
        " WHERE p_lore_model IS NULL OR p_lore_model != ? ORDER BY id",
        (model_version,),
    ).fetchall()
    return _score_rows(conn, model, model_version, rows, rules)


def model_rules_version(conn: sqlite3.Connection) -> str | None:
    """The `rules.version` the latest trained model's features were built under.

    `None` when no model has been trained yet. `""` for a model trained before
    this was recorded (params_json has no `rules_version` key) — treated as
    "definitely not the active rules" so a stale model from an old build still
    triggers the staleness warning rather than silently passing as current.
    """
    row = conn.execute(
        "SELECT params_json FROM triage_model ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    params = json.loads(row["params_json"])
    return str(params.get("rules_version", ""))


def _holdout_scored_for_model(
    conn: sqlite3.Connection, model: Model, rules: TriageRules = DEFAULT_RULES
) -> list[tuple[float, Label]]:
    examples = build_examples(conn, rules)
    return [
        (p_lore(model, example.tokens), example.label)
        for example in examples
        if in_holdout(example.exchange_id)
    ]


def _corpus_share(conn: sqlite3.Connection, threshold: float) -> float:
    total = int(
        conn.execute("SELECT COUNT(*) FROM exchanges WHERE p_lore IS NOT NULL").fetchone()[0]
    )
    if not total:
        return 0.0
    passing = int(
        conn.execute("SELECT COUNT(*) FROM exchanges WHERE p_lore >= ?", (threshold,)).fetchone()[0]
    )
    return passing / total


def recommend(
    conn: sqlite3.Connection, min_recall: float, rules: TriageRules = DEFAULT_RULES
) -> tuple[Metrics, float] | None:
    loaded = load_latest_model(conn)
    if loaded is None:
        raise NoTrainedModelError
    _, model = loaded
    holdout_scored = _holdout_scored_for_model(conn, model, rules)
    table = evaluate(holdout_scored, candidate_thresholds(holdout_scored))
    metric = recommend_threshold(table, min_recall)
    if metric is None:
        return None
    return metric, _corpus_share(conn, metric.threshold)


@dataclass(frozen=True)
class RecommendationRow:
    min_recall: float
    metric: Metrics | None
    share: float | None
    formatted_threshold: str | None


def recommend_table(
    conn: sqlite3.Connection,
    min_recalls: Sequence[float] = RECALL_TARGETS,
    rules: TriageRules = DEFAULT_RULES,
) -> list[RecommendationRow]:
    """Recommend a threshold for each of `min_recalls` in one holdout pass.

    Reuses the same candidate thresholds (the holdout's own distinct p_lore
    scores) for every target, so this is a single scoring/evaluation pass no
    matter how many recall targets are requested.
    """
    loaded = load_latest_model(conn)
    if loaded is None:
        raise NoTrainedModelError
    _, model = loaded
    holdout_scored = _holdout_scored_for_model(conn, model, rules)
    table = evaluate(holdout_scored, candidate_thresholds(holdout_scored))

    rows: list[RecommendationRow] = []
    for min_recall in min_recalls:
        metric = recommend_threshold(table, min_recall)
        if metric is None:
            rows.append(RecommendationRow(min_recall, None, None, None))
            continue
        formatted = format_threshold(metric.threshold, holdout_scored)
        share = _corpus_share(conn, float(formatted))
        rows.append(RecommendationRow(min_recall, metric, share, formatted))
    return rows
