"""The message-level trash classifier (issue #128 PR 3, redesigned by #135).

Trains **two** naive Bayes models, each with honest counts (no duplicated
examples), and combines them with a fitted logistic regression:

- a **citation model**, trained on citation-labeled messages only, excluding
  any message that also carries a human label (a human label always wins at
  read time -- `infovore.db.message_labels.effective_message_labels_with_source`
  -- so that message belongs to the human model, never both);
- a **human model**, trained on human-labeled messages only;
- a **combiner** (`infovore.triage.logistic`, extended by #135 to fit
  continuous-valued features) fit on `logit(p_citation)`, `logit(p_human)`,
  and a `human_seen` indicator, using out-of-fold `p_human` predictions so
  the combiner never sees the human model's in-sample overconfidence.

This replaces the old single-model design, which counted a non-holdout human
example `--human-weight` times by plain repetition: Naive Bayes has no
defense against duplicated evidence, so at a weight high enough for human
labels to matter at all, a single human-trashed message's distinctive words
could dominate the model outright (issue #135's motivating example: one
message mentioning an SGI "Judge" graphics card sent every unrelated message
containing "judge" to a high `p_trash`). With honest, non-duplicated counts
per model and a combiner fit on out-of-fold predictions, one human example is
just one example.

**Reuse mapping.** `infovore.triage.bayes.train`/`p_lore`/`evaluate`/`auc`
are written generically against `infovore.rows.Label.LORE`/`.NOISE` (the
"positive"/"negative" class a score predicts) -- they never look at an
exchange specifically. This module scores `p_trash`, so `MessageLabel.TRASH`
is mapped onto `Label.LORE` (the positive class the score predicts) and
`MessageLabel.KEEP` onto `Label.NOISE` (`_bayes_label`, used everywhere a
label crosses into `bayes.py`). `p_trash` for a token set is then exactly
`p_lore(model, tokens)` for a single model, or the combiner's own output for
the full ensemble -- no inversion anywhere.
"""

import hashlib
import json
import math
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from infovore.db.batch import BATCH_SIZE, SQLITE_MAX_VARIABLES
from infovore.db.codec import to_db_time
from infovore.db.connection import transaction
from infovore.db.message_labels import effective_message_labels_with_source
from infovore.db.raw import messages_by_ids
from infovore.privacy.optout import opted_out_user_ids
from infovore.rows import MessageLabel, MessageLabelSource
from infovore.sift.features import (
    DEFAULT_FEATURE_SET,
    FEATURE_SET_VERSION,
    FeatureSet,
    context_tokens_for_feature_set,
    exchange_context_tokens,
    message_features,
    structural_context_tokens,
)
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
from infovore.triage.logistic import FeatureValues, LogisticModel, predict_proba, train_logistic
from infovore.triage.parallel import ChunkPool

# Citation labels are weak but plentiful -- this is the same floor the old
# single-model design used, now scoped to citation-only examples (a message
# with both a citation and a human label counts toward the human model only).
MIN_LABELS_PER_CLASS = 10

# A human sift decision is direct, deliberate evidence, but there are always
# far fewer of them than citation labels. Below this many of either class,
# there isn't enough to fit a combiner honestly, so `sift train` falls back
# to the citation model alone (issue #135 design point 4).
DEFAULT_MIN_HUMAN_LABELS_PER_CLASS = 30

# K-fold split (by sha256 of the message id, the same hashing scheme as
# `infovore.triage.bayes.in_holdout`, generalized from one holdout bucket to
# `FOLDS` buckets) used to generate honest, out-of-fold `p_human` predictions
# for the combiner's training data, and to cross-validate the combiner's own
# fit for the out-of-fold report.
FOLDS = 5

# Probabilities are clipped to this range before taking a logit, so neither
# `citation_logit` nor `human_logit` ever reaches +/-inf.
CLIP_EPSILON = 1e-6

EVAL_THRESHOLDS = tuple(round(i / 10, 1) for i in range(1, 10))
DEFAULT_CONFUSION_THRESHOLD = 0.5
TOP_TOKENS_LIMIT = 15
DISCARD_THRESHOLDS = (0.5, 0.7, 0.9)

CITATION_LOGIT_FEATURE = "citation_logit"
HUMAN_LOGIT_FEATURE = "human_logit"
HUMAN_SEEN_FEATURE = "human_seen"

__all__ = [
    "CITATION_LOGIT_FEATURE",
    "CLIP_EPSILON",
    "DEFAULT_CONFUSION_THRESHOLD",
    "DEFAULT_MIN_HUMAN_LABELS_PER_CLASS",
    "DISCARD_THRESHOLDS",
    "FOLDS",
    "HUMAN_LOGIT_FEATURE",
    "HUMAN_SEEN_FEATURE",
    "MIN_LABELS_PER_CLASS",
    "DiscardRow",
    "Ensemble",
    "FeatureSetAblation",
    "FeatureSetMismatchError",
    "InsufficientLabelsError",
    "MessageExample",
    "MessageTokenInfo",
    "MessageTrainReport",
    "build_message_examples",
    "load_latest_message_model",
    "p_trash_for_tokens",
    "score_all",
    "score_stale",
    "train_and_store",
]


class InsufficientLabelsError(Exception):
    def __init__(self, keep_count: int, trash_count: int) -> None:
        super().__init__(
            f"need at least {MIN_LABELS_PER_CLASS} citation-labeled messages of each class"
            f" (excluding any message that also has a human label) to train the citation"
            f" model (have keep={keep_count} trash={trash_count})"
        )
        self.keep_count = keep_count
        self.trash_count = trash_count


class FeatureSetMismatchError(Exception):
    """Raised when scoring is attempted with an ensemble trained under a
    different `infovore.sift.features.FEATURE_SET_VERSION` (issue #141): the
    stored model's token counts assume a feature set that no longer matches
    what `message_features`/`exchange_context_tokens` build today, so
    scoring it would silently mix incompatible tokens rather than fail
    loudly. Retraining (`infovore sift train`) always stores the current
    version, so this only ever fires against a stale, pre-retrain model."""

    def __init__(self, stored_version: int, current_version: int) -> None:
        super().__init__(
            f"stored model was trained with feature set version {stored_version}, but the"
            f" current feature set is version {current_version}; run `infovore sift train`"
            " to retrain before scoring"
        )
        self.stored_version = stored_version
        self.current_version = current_version


@dataclass(frozen=True)
class MessageExample:
    """One labeled message's tokens under all three `FeatureSet`s at once
    (issue #144), built once per message so `train_and_store` can fit and
    compare all three without reloading or re-resolving any exchange:
    `plain_tokens` (no conversation context at all), `structural_tokens`
    (context signals only -- no neighbour word tokens), and
    `context_tokens` (everything, issue #141's original shape)."""

    message_id: int
    channel_id: int
    plain_tokens: frozenset[str]
    structural_tokens: frozenset[str]
    context_tokens: frozenset[str]
    label: MessageLabel
    source: MessageLabelSource


def _tokens_for_feature_set(example: MessageExample, feature_set: FeatureSet) -> frozenset[str]:
    """The exact token set `example` trains/scores under `feature_set`."""
    if feature_set is FeatureSet.PLAIN:
        return example.plain_tokens
    if feature_set is FeatureSet.STRUCTURAL:
        return example.structural_tokens
    return example.context_tokens


def _selector_for_feature_set(
    feature_set: FeatureSet,
) -> Callable[[MessageExample], frozenset[str]]:
    """A `token_selector` (`_fit_ablation`) bound to one `FeatureSet`, as a
    real function rather than a per-iteration lambda -- mypy can't infer a
    lambda's parameter types inside a dict/generator comprehension, and a
    named function is clearer at the call site anyway."""

    def selector(example: MessageExample) -> frozenset[str]:
        return _tokens_for_feature_set(example, feature_set)

    return selector


@dataclass(frozen=True)
class MessageTokenInfo:
    token: str
    probability: float
    trash_count: int
    keep_count: int


@dataclass(frozen=True)
class DiscardRow:
    threshold: float
    keep_lost: float | None
    trash_caught: float | None


@dataclass(frozen=True)
class Ensemble:
    """The full, persisted classifier state `load_latest_message_model`
    reconstructs: the citation model (always present), the human model and
    combiner (present unless `fallback`), the combiner's own version (what
    `messages.p_trash_model` records), and the feature set version it was
    trained under (issue #141 -- `_score_rows` refuses to score with this
    when it no longer matches `infovore.sift.features.FEATURE_SET_VERSION`,
    see `FeatureSetMismatchError`), and the named `feature_set` it was
    trained on (issue #144 -- resolved from `message_combiner.
    feature_set_name` when present, or from `feature_set_version` alone
    for a row stored before that column existed, see
    `load_latest_message_model`)."""

    citation_version: int
    citation_model: Model
    human_version: int | None
    human_model: Model | None
    combiner_version: int
    combiner: LogisticModel | None
    fallback: bool
    feature_set_version: int
    feature_set: FeatureSet


@dataclass(frozen=True)
class FeatureSetAblation:
    """One row of the issue #144 three-way ablation: one named `FeatureSet`'s
    out-of-fold combined AUC and discard pile, fit via `_fit_ablation` on
    the exact same folds and the exact same fallback decision as the other
    two rows (`train_and_store` calls `_fit_ablation` once per
    `FeatureSet`, each selecting a different `MessageExample` field via
    `_tokens_for_feature_set`) -- so all three rows are a true
    apples-to-apples comparison over identical data, never three separate,
    possibly-disagreeing runs."""

    feature_set: FeatureSet
    citation_auc: float | None
    human_auc: float | None
    combined_auc: float | None
    discard_pile: tuple[DiscardRow, ...]


@dataclass(frozen=True)
class MessageTrainReport:
    version: int
    citation_labels_used: int
    human_labels_used: int
    fallback: bool
    min_human_labels_per_class: int
    citation_auc: float | None
    human_auc: float | None
    combined_auc: float | None
    discard_pile: tuple[DiscardRow, ...]
    citation_top_tokens: tuple[MessageTokenInfo, ...]
    human_top_tokens: tuple[MessageTokenInfo, ...]
    # Secondary section: the citation model alone, against its own 1-in-5
    # holdout (issue #135 design point 6 keeps this as a sanity check).
    citation_holdout_metrics: tuple[Metrics, ...]
    citation_holdout_confusion: Metrics
    citation_holdout_auc: float | None
    channel_stats: dict[str, Metrics]
    # The persisted/scored feature set (issue #144 -- `--features`,
    # default `plain`) -- the figures above (`citation_auc`/`human_auc`/
    # `combined_auc`/`discard_pile`/the top-tokens and holdout sections)
    # are all this set's own fit.
    feature_set: FeatureSet
    # Three-way ablation (issue #144, superseding #141's two-way version):
    # the same out-of-fold AUC and discard pile as above, fit and
    # evaluated on the *same folds*, once per named `FeatureSet` -- so the
    # gain or loss any one set brings over the others is measured, not
    # assumed, on every run. One entry's figures always equal the fields
    # above exactly: whichever `FeatureSetAblation.feature_set` matches
    # `feature_set`.
    ablations: tuple[FeatureSetAblation, ...]


def _bayes_label(label: MessageLabel) -> Label:
    return Label.LORE if label is MessageLabel.TRASH else Label.NOISE


def _exchange_ids_for_messages(
    conn: sqlite3.Connection, message_ids: Sequence[int]
) -> dict[int, int]:
    result: dict[int, int] = {}
    for start in range(0, len(message_ids), SQLITE_MAX_VARIABLES):
        chunk = message_ids[start : start + SQLITE_MAX_VARIABLES]
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT message_id, exchange_id FROM exchange_messages"
            f" WHERE message_id IN ({placeholders})",
            chunk,
        ):
            result[row["message_id"]] = row["exchange_id"]
    return result


def build_message_examples(conn: sqlite3.Connection) -> list[MessageExample]:
    """Every labeled message's training example, under all three named
    `FeatureSet`s at once (issue #144): `plain_tokens` (no conversation
    context at all), `structural_tokens` (context signals only, filtered
    from the full context via `infovore.sift.features.
    structural_context_tokens`), and `context_tokens` (everything, issue
    #141's original shape). Each labeled message's exchange is loaded
    once, in position order, via
    `infovore.sift.features.exchange_context_tokens` (itself batched
    across every exchange these messages belong to, not one query per
    exchange) -- the one expensive, per-exchange step every feature set
    shares; deriving `structural_tokens` from the already-built full
    context is then just a cheap, pure filter, not another exchange load."""
    labeled = effective_message_labels_with_source(conn)
    ids = sorted(labeled)
    if not ids:
        return []
    opted_out = opted_out_user_ids(conn)
    exchange_id_by_message = _exchange_ids_for_messages(conn, ids)
    exchange_ids = sorted(set(exchange_id_by_message.values()))
    context_by_message = exchange_context_tokens(conn, exchange_ids, opted_out)

    examples: list[MessageExample] = []
    for start in range(0, len(ids), SQLITE_MAX_VARIABLES):
        chunk = ids[start : start + SQLITE_MAX_VARIABLES]
        for message in messages_by_ids(conn, chunk):
            label, source = labeled[message.id]
            full_context = context_by_message.get(message.id, frozenset())
            plain_tokens = message_features(message, message.channel_id)
            structural_tokens = message_features(
                message, message.channel_id, structural_context_tokens(full_context)
            )
            context_tokens = message_features(message, message.channel_id, full_context)
            examples.append(
                MessageExample(
                    message.id,
                    message.channel_id,
                    plain_tokens,
                    structural_tokens,
                    context_tokens,
                    label,
                    source,
                )
            )
    return examples


def _clip(probability: float) -> float:
    return min(max(probability, CLIP_EPSILON), 1.0 - CLIP_EPSILON)


def _logit(probability: float) -> float:
    clipped = _clip(probability)
    return math.log(clipped / (1.0 - clipped))


def _fold_of(message_id: int, folds: int = FOLDS) -> int:
    digest = hashlib.sha256(str(message_id).encode()).digest()
    return digest[0] % folds


def _combiner_features(p_citation: float, p_human: float, human_seen: bool) -> dict[str, float]:
    return {
        CITATION_LOGIT_FEATURE: _logit(p_citation),
        HUMAN_LOGIT_FEATURE: _logit(p_human),
        HUMAN_SEEN_FEATURE: 1.0 if human_seen else 0.0,
    }


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


def _channel_names(conn: sqlite3.Connection, channel_ids: Sequence[int]) -> dict[int, str]:
    ids = sorted(set(channel_ids))
    names: dict[int, str] = {}
    for start in range(0, len(ids), SQLITE_MAX_VARIABLES):
        chunk = ids[start : start + SQLITE_MAX_VARIABLES]
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT id, name FROM channels WHERE id IN ({placeholders})", chunk
        ):
            names[row["id"]] = row["name"]
    return names


def p_trash_for_tokens(ensemble: Ensemble, tokens: frozenset[str]) -> float:
    """`p_trash` for one message's tokens under the full ensemble: the
    citation model alone in fallback mode, otherwise the combiner's output
    over both models' scores."""
    p_citation = p_lore(ensemble.citation_model, tokens)
    if ensemble.fallback or ensemble.human_model is None or ensemble.combiner is None:
        return p_citation
    p_human = p_lore(ensemble.human_model, tokens)
    features = _combiner_features(p_citation, p_human, human_seen=p_human != 0.5)
    return predict_proba(ensemble.combiner, features)


@dataclass(frozen=True)
class _AblationFit:
    """One named feature set's fit for the issue #144 three-way ablation: a
    citation model, human model and combiner fit purely from a
    `token_selector` over `MessageExample` (`_tokens_for_feature_set` bound
    to one `FeatureSet`), plus the out-of-fold AUCs and discard pile that
    fitting produces. Every `FeatureSet`'s fit shares the exact same fold
    split (`_fold_of`, keyed by message id) and the exact same fallback
    decision (human label counts alone, independent of tokens), so all
    three are a true apples-to-apples comparison."""

    citation_model: Model
    human_model: Model | None
    combiner: LogisticModel | None
    citation_auc: float | None
    human_auc: float | None
    combined_auc: float | None
    discard_pile: tuple[DiscardRow, ...]


def _fit_ablation(
    citation_examples: Sequence[MessageExample],
    human_examples: Sequence[MessageExample],
    fallback: bool,
    token_selector: Callable[[MessageExample], frozenset[str]],
) -> _AblationFit:
    citation_train_pairs = [
        (token_selector(e), _bayes_label(e.label))
        for e in citation_examples
        if not in_holdout(e.message_id)
    ]
    citation_model = train(citation_train_pairs)

    if fallback:
        citation_pairs_vs_human = [
            (p_lore(citation_model, token_selector(e)), _bayes_label(e.label))
            for e in human_examples
        ]
        citation_auc = auc(citation_pairs_vs_human)
        discard_pile = tuple(
            DiscardRow(threshold=threshold, keep_lost=None, trash_caught=None)
            for threshold in DISCARD_THRESHOLDS
        )
        return _AblationFit(citation_model, None, None, citation_auc, None, None, discard_pile)

    folds_of_human = {e.message_id: _fold_of(e.message_id) for e in human_examples}
    p_human_oof: dict[int, float] = {}
    for fold in range(FOLDS):
        sub_train_pairs = [
            (token_selector(e), _bayes_label(e.label))
            for e in human_examples
            if folds_of_human[e.message_id] != fold
        ]
        sub_model = train(sub_train_pairs)
        for e in human_examples:
            if folds_of_human[e.message_id] == fold:
                p_human_oof[e.message_id] = p_lore(sub_model, token_selector(e))

    p_citation_for_human = {
        e.message_id: p_lore(citation_model, token_selector(e)) for e in human_examples
    }

    def _features_for(message_id: int) -> dict[str, float]:
        p_human = p_human_oof[message_id]
        return _combiner_features(
            p_citation_for_human[message_id], p_human, human_seen=p_human != 0.5
        )

    combiner_examples: list[tuple[FeatureValues, bool]] = [
        (_features_for(e.message_id), e.label is MessageLabel.TRASH) for e in human_examples
    ]
    combiner = train_logistic(combiner_examples)

    # Nested cross-validation, fold-excluding the combiner itself too
    # (cheap: it fits three already-computed features, no NB retraining),
    # so the reported "combined" AUC is honestly out-of-fold rather than
    # scored by a combiner that saw these exact rows during its own fit.
    combined_oof: dict[int, float] = {}
    for fold in range(FOLDS):
        fold_train_examples: list[tuple[FeatureValues, bool]] = [
            (_features_for(e.message_id), e.label is MessageLabel.TRASH)
            for e in human_examples
            if folds_of_human[e.message_id] != fold
        ]
        fold_combiner = train_logistic(fold_train_examples)
        for e in human_examples:
            if folds_of_human[e.message_id] == fold:
                combined_oof[e.message_id] = predict_proba(
                    fold_combiner, _features_for(e.message_id)
                )

    human_pairs = [(p_human_oof[e.message_id], _bayes_label(e.label)) for e in human_examples]
    citation_pairs = [
        (p_citation_for_human[e.message_id], _bayes_label(e.label)) for e in human_examples
    ]
    combined_pairs = [(combined_oof[e.message_id], _bayes_label(e.label)) for e in human_examples]
    human_auc = auc(human_pairs)
    citation_auc = auc(citation_pairs)
    combined_auc = auc(combined_pairs)

    keep_scores = [
        combined_oof[e.message_id] for e in human_examples if e.label is MessageLabel.KEEP
    ]
    trash_scores = [
        combined_oof[e.message_id] for e in human_examples if e.label is MessageLabel.TRASH
    ]
    discard_pile = tuple(
        DiscardRow(
            threshold=threshold,
            keep_lost=_share_at_or_above(keep_scores, threshold),
            trash_caught=_share_at_or_above(trash_scores, threshold),
        )
        for threshold in DISCARD_THRESHOLDS
    )

    # Final human model: retrained on *all* human labels -- the fold split
    # above only ever generated the combiner's training inputs.
    human_model = train([(token_selector(e), _bayes_label(e.label)) for e in human_examples])
    return _AblationFit(
        citation_model, human_model, combiner, citation_auc, human_auc, combined_auc, discard_pile
    )


def train_and_store(
    conn: sqlite3.Connection,
    clock: Clock,
    confusion_threshold: float = DEFAULT_CONFUSION_THRESHOLD,
    min_human_labels_per_class: int = DEFAULT_MIN_HUMAN_LABELS_PER_CLASS,
    feature_set: FeatureSet = DEFAULT_FEATURE_SET,
) -> MessageTrainReport:
    examples = build_message_examples(conn)
    human_examples = [e for e in examples if e.source is MessageLabelSource.HUMAN]
    human_ids = {e.message_id for e in human_examples}
    citation_examples = [
        e
        for e in examples
        if e.source is MessageLabelSource.CITATION and e.message_id not in human_ids
    ]

    citation_keep = sum(1 for e in citation_examples if e.label is MessageLabel.KEEP)
    citation_trash = sum(1 for e in citation_examples if e.label is MessageLabel.TRASH)
    if citation_keep < MIN_LABELS_PER_CLASS or citation_trash < MIN_LABELS_PER_CLASS:
        raise InsufficientLabelsError(citation_keep, citation_trash)

    human_keep = sum(1 for e in human_examples if e.label is MessageLabel.KEEP)
    human_trash = sum(1 for e in human_examples if e.label is MessageLabel.TRASH)
    fallback = human_keep < min_human_labels_per_class or human_trash < min_human_labels_per_class

    # Issue #144: fit all three named feature sets on the exact same folds
    # and the exact same fallback decision -- a true three-way ablation,
    # not three separate runs that might disagree on data. Only
    # `feature_set` (default `plain`) is actually persisted/scored below;
    # the other two are report-only.
    fits = {
        fs: _fit_ablation(
            citation_examples, human_examples, fallback, _selector_for_feature_set(fs)
        )
        for fs in FeatureSet
    }
    chosen_fit = fits[feature_set]

    citation_model = chosen_fit.citation_model
    human_model = chosen_fit.human_model
    combiner = chosen_fit.combiner

    # --- citation model, with its own internal holdout for the secondary section ---
    citation_holdout = [e for e in citation_examples if in_holdout(e.message_id)]
    citation_holdout_scored = [
        (p_lore(citation_model, _tokens_for_feature_set(e, feature_set)), e)
        for e in citation_holdout
    ]
    citation_secondary_pairs = [
        (score, _bayes_label(e.label)) for score, e in citation_holdout_scored
    ]
    citation_holdout_metrics = tuple(evaluate(citation_secondary_pairs, EVAL_THRESHOLDS))
    if confusion_threshold in EVAL_THRESHOLDS:
        citation_holdout_confusion = next(
            metric for metric in citation_holdout_metrics if metric.threshold == confusion_threshold
        )
    else:
        citation_holdout_confusion = evaluate(citation_secondary_pairs, [confusion_threshold])[0]
    citation_holdout_auc = auc(citation_secondary_pairs)

    channel_names = _channel_names(conn, [e.channel_id for e in citation_examples])
    channel_groups: dict[str, list[tuple[float, Label]]] = {}
    for score, e in citation_holdout_scored:
        name = channel_names.get(e.channel_id, f"channel {e.channel_id}")
        channel_groups.setdefault(name, []).append((score, _bayes_label(e.label)))
    channel_stats = {
        name: evaluate(pairs, [confusion_threshold])[0] for name, pairs in channel_groups.items()
    }

    citation_top_tokens = _top_message_tokens(citation_model)
    human_top_tokens = _top_message_tokens(human_model) if human_model is not None else ()

    now_text = to_db_time(clock.now())
    with transaction(conn):
        cursor = conn.execute(
            "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json, kind)"
            " VALUES (?, ?, ?, ?, 'citation')",
            (
                now_text,
                len(citation_examples),
                len(citation_holdout),
                json.dumps(
                    {
                        "trash_documents": citation_model.lore_documents,
                        "keep_documents": citation_model.noise_documents,
                    }
                ),
            ),
        )
        citation_version = int(cursor.lastrowid or 0)
        conn.executemany(
            "INSERT INTO message_tokens (model_version, token, trash_count, keep_count)"
            " VALUES (?, ?, ?, ?)",
            [
                (citation_version, token, trash, keep)
                for token, (trash, keep) in citation_model.counts.items()
            ],
        )

        human_version: int | None = None
        if human_model is not None:
            cursor = conn.execute(
                "INSERT INTO message_model"
                " (trained_at, labels_used, holdout_size, params_json, kind)"
                " VALUES (?, ?, 0, ?, 'human')",
                (
                    now_text,
                    len(human_examples),
                    json.dumps(
                        {
                            "trash_documents": human_model.lore_documents,
                            "keep_documents": human_model.noise_documents,
                        }
                    ),
                ),
            )
            human_version = int(cursor.lastrowid or 0)
            conn.executemany(
                "INSERT INTO message_tokens (model_version, token, trash_count, keep_count)"
                " VALUES (?, ?, ?, ?)",
                [
                    (human_version, token, trash, keep)
                    for token, (trash, keep) in human_model.counts.items()
                ],
            )

        combiner_params = {
            "intercept": combiner.intercept if combiner is not None else 0.0,
            "weights": dict(combiner.weights) if combiner is not None else {},
            "min_human_labels_per_class": min_human_labels_per_class,
            "human_keep": human_keep,
            "human_trash": human_trash,
        }
        cursor = conn.execute(
            "INSERT INTO message_combiner (trained_at, citation_model_version,"
            " human_model_version, fallback, params_json, feature_set_version,"
            " feature_set_name)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                now_text,
                citation_version,
                human_version,
                int(fallback),
                json.dumps(combiner_params),
                FEATURE_SET_VERSION,
                feature_set.value,
            ),
        )
        combiner_version = int(cursor.lastrowid or 0)

    ablations = tuple(
        FeatureSetAblation(
            feature_set=fs,
            citation_auc=fits[fs].citation_auc,
            human_auc=fits[fs].human_auc,
            combined_auc=fits[fs].combined_auc,
            discard_pile=fits[fs].discard_pile,
        )
        for fs in FeatureSet
    )

    return MessageTrainReport(
        version=combiner_version,
        citation_labels_used=len(citation_examples),
        human_labels_used=len(human_examples),
        fallback=fallback,
        min_human_labels_per_class=min_human_labels_per_class,
        citation_auc=chosen_fit.citation_auc,
        human_auc=chosen_fit.human_auc,
        combined_auc=chosen_fit.combined_auc,
        discard_pile=chosen_fit.discard_pile,
        citation_top_tokens=citation_top_tokens,
        human_top_tokens=human_top_tokens,
        citation_holdout_metrics=citation_holdout_metrics,
        citation_holdout_confusion=citation_holdout_confusion,
        citation_holdout_auc=citation_holdout_auc,
        channel_stats=channel_stats,
        feature_set=feature_set,
        ablations=ablations,
    )


def _load_model(conn: sqlite3.Connection, version: int) -> Model:
    row = conn.execute(
        "SELECT params_json FROM message_model WHERE version = ?", (version,)
    ).fetchone()
    params = json.loads(row["params_json"])
    counts: dict[str, tuple[int, int]] = {}
    for token_row in conn.execute(
        "SELECT token, trash_count, keep_count FROM message_tokens WHERE model_version = ?",
        (version,),
    ):
        counts[token_row["token"]] = (token_row["trash_count"], token_row["keep_count"])
    return Model(
        lore_documents=params["trash_documents"],
        noise_documents=params["keep_documents"],
        counts=counts,
    )


def _legacy_feature_set(version: int) -> FeatureSet:
    """Resolves a `message_combiner` row written before issue #144 (no
    `feature_set_name` column existed yet, only `feature_set_version`) to
    the named `FeatureSet` its version always meant: version 1 was #141's
    pre-context shape (`FeatureSet.PLAIN`), version 2 was #141's own
    full-context shape -- the only feature-set choice that existed before
    this issue, always on (`FeatureSet.CONTEXT`)."""
    return FeatureSet.PLAIN if version == 1 else FeatureSet.CONTEXT


def load_latest_message_model(conn: sqlite3.Connection) -> Ensemble | None:
    row = conn.execute(
        "SELECT version, citation_model_version, human_model_version, fallback, params_json,"
        " feature_set_version, feature_set_name FROM message_combiner"
        " ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    params = json.loads(row["params_json"])
    citation_version = int(row["citation_model_version"])
    citation_model = _load_model(conn, citation_version)

    human_version_raw = row["human_model_version"]
    human_version = int(human_version_raw) if human_version_raw is not None else None
    human_model = _load_model(conn, human_version) if human_version is not None else None

    fallback = bool(row["fallback"])
    combiner = (
        None
        if fallback
        else LogisticModel(intercept=params["intercept"], weights=params["weights"])
    )

    feature_set_version = int(row["feature_set_version"])
    feature_set_name = row["feature_set_name"]
    feature_set = (
        FeatureSet(feature_set_name)
        if feature_set_name is not None
        else _legacy_feature_set(feature_set_version)
    )

    return Ensemble(
        citation_version=citation_version,
        citation_model=citation_model,
        human_version=human_version,
        human_model=human_model,
        combiner_version=int(row["version"]),
        combiner=combiner,
        fallback=fallback,
        feature_set_version=feature_set_version,
        feature_set=feature_set,
    )


# Module-level (spawn-safe) worker state for `ChunkPool`, mirroring
# `infovore.triage.train`'s `_worker_model`/`_init_p_lore_worker` pattern
# (issue #113): the ensemble is sent to each worker once, via the pool's
# initializer, not per task.
_worker_ensemble: Ensemble | None = None


def _init_p_trash_worker(ensemble: Ensemble) -> None:
    global _worker_ensemble
    _worker_ensemble = ensemble


def _p_trash_worker_chunk(items: list[tuple[int, frozenset[str]]]) -> list[tuple[int, float]]:
    ensemble = _worker_ensemble
    assert ensemble is not None, "ChunkPool must call _init_p_trash_worker before this"
    return [(message_id, p_trash_for_tokens(ensemble, tokens)) for message_id, tokens in items]


def _score_rows(
    conn: sqlite3.Connection,
    ensemble: Ensemble,
    rows: list[sqlite3.Row],
    workers: int = 1,
) -> int:
    """Score `rows` (message id + channel id), `BATCH_SIZE` at a time: each
    batch's exchange context is loaded and built once, batched, in this
    (main) process (issue #141 -- `infovore.sift.features.
    exchange_context_tokens`), before the pure-Python `p_trash` scoring
    itself is handed to `ChunkPool`'s workers, exactly as `message_features`
    without context was before. Builds exactly `ensemble.feature_set`'s
    tokens (issue #144, `infovore.sift.features.
    context_tokens_for_feature_set`) -- the exchange-context lookup is
    skipped entirely when that set is `FeatureSet.PLAIN`, which needs none
    of it. Refuses up front if `ensemble` was trained under a feature set
    that no longer matches `infovore.sift.features.FEATURE_SET_VERSION`
    (`FeatureSetMismatchError`) -- scoring even one row with mismatched
    tokens would be silently wrong, not just imprecise."""
    if ensemble.feature_set_version != FEATURE_SET_VERSION:
        raise FeatureSetMismatchError(ensemble.feature_set_version, FEATURE_SET_VERSION)
    needs_context = ensemble.feature_set is not FeatureSet.PLAIN
    opted_out = opted_out_user_ids(conn) if needs_context else frozenset()
    count = 0
    with ChunkPool(_p_trash_worker_chunk, workers, _init_p_trash_worker, (ensemble,)) as pool:
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            message_ids = [row["id"] for row in batch]
            message_rows = messages_by_ids(conn, message_ids)
            by_id = {message.id: message for message in message_rows}
            if needs_context:
                exchange_id_by_message = _exchange_ids_for_messages(conn, message_ids)
                exchange_ids = sorted(set(exchange_id_by_message.values()))
                context_by_message = exchange_context_tokens(conn, exchange_ids, opted_out)
            else:
                context_by_message = {}
            items = [
                (
                    row["id"],
                    message_features(
                        by_id[row["id"]],
                        row["channel_id"],
                        context_tokens_for_feature_set(
                            ensemble.feature_set,
                            context_by_message.get(row["id"], frozenset()),
                        ),
                    ),
                )
                for row in batch
                if row["id"] in by_id
            ]
            updates = [
                (score, ensemble.combiner_version, message_id)
                for chunk_result in pool.map_chunks(items)
                for message_id, score in chunk_result
            ]
            with transaction(conn):
                conn.executemany(
                    "UPDATE messages SET p_trash = ?, p_trash_model = ? WHERE id = ?", updates
                )
            count += len(updates)
    return count


def score_all(conn: sqlite3.Connection, ensemble: Ensemble, workers: int = 1) -> int:
    """Score every message that belongs to an exchange (issue #128 PR 3
    scores messages *in exchanges* only -- a message never grouped into an
    exchange was never a candidate for extraction and isn't part of this
    corpus)."""
    rows = conn.execute(
        "SELECT DISTINCT m.id AS id, m.channel_id AS channel_id"
        " FROM messages m JOIN exchange_messages em ON em.message_id = m.id"
        " ORDER BY m.id"
    ).fetchall()
    return _score_rows(conn, ensemble, rows, workers)


def score_stale(conn: sqlite3.Connection, ensemble: Ensemble, workers: int = 1) -> int:
    rows = conn.execute(
        "SELECT DISTINCT m.id AS id, m.channel_id AS channel_id"
        " FROM messages m JOIN exchange_messages em ON em.message_id = m.id"
        " WHERE m.p_trash_model IS NULL OR m.p_trash_model != ?"
        " ORDER BY m.id",
        (ensemble.combiner_version,),
    ).fetchall()
    return _score_rows(conn, ensemble, rows, workers)
