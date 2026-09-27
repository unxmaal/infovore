"""Tune the programmatic triage rules from labeled outcomes (issue #95,
deliverables 2-5): `infovore triage --signal-report`, `--suggest-terms`,
`--fit-weights`, and the report-card additions to `--report`.

Two logistic fits are used, for different purposes:

- The **signal-only** fit (`_fit_signal_weights`) trains on just the
  `SIG_<name>` rule-signal indicators. It answers "how good are the rules,
  on their own" — `--signal-report`'s fitted-weight column, and the
  weights `--fit-weights` writes into a rules TOML (only `SIG_<name>`
  coefficients map onto that schema; `domain_terms` et al are copied
  unchanged, per the issue's "terms unchanged").
- The **combined** fit (`_fit_combined_weights`) additionally includes a
  binned `p_lore` feature (`BAYES_00".."BAYES_99`, SpamAssassin-style) when
  a trained Bayes model exists, and a `CHAN_<id>` channel feature. It
  follows the issue's SpamAssassin-inspired addendum: weight every source
  of evidence together rather than picking rule score or Bayes. Its
  `SIG_<name>` coefficients (after joint fitting with `p_lore` and channel)
  are what actually get written by `--fit-weights` — jointly fitting first
  keeps a rule's fitted weight from double-counting evidence the Bayes
  model already explains — while its full predictions (including the
  `BAYES_`/`CHAN_` features) are what the `--report` report card
  cross-validates as "the fitted combination".
"""

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

from infovore.rows import Label
from infovore.triage.bayes import auc, candidate_thresholds, evaluate, recommend_threshold
from infovore.triage.bayes import token_probability as bayes_token_probability
from infovore.triage.logistic import LogisticModel, predict_proba, train_logistic
from infovore.triage.rules import TriageRules
from infovore.triage.train import (
    Example,
    NoTrainedModelError,
    build_examples,
    load_latest_model,
)

# --- shared constants --------------------------------------------------------

# (reason name from infovore.triage.score.score_exchange, TriageRules field
# holding its weight/penalty). Order matches score.py's own signal order.
SIGNAL_WEIGHT_FIELDS: tuple[tuple[str, str], ...] = (
    ("domain_terms", "domain_term_weight"),
    ("irix_version", "irix_version_weight"),
    ("part_number", "part_number_weight"),
    ("unix_path", "unix_path_weight"),
    ("code", "code_weight"),
    ("archive_link", "archive_link_weight"),
    ("pdf_attachment", "pdf_attachment_weight"),
    ("answered_question", "answered_question_weight"),
    ("agreed_answer", "agreed_answer_weight"),
    ("thread", "thread_weight"),
    ("substantial", "substantial_weight"),
    ("mostly_tiny_messages", "tiny_penalty"),
    ("gif_links", "gif_penalty"),
    ("laughter", "laughter_penalty"),
)

MIN_SIGNAL_SUPPORT = 5
USELESS_LIFT_BAND = 0.15

FOLDS = 5
MIN_RECALL_FOR_REPORT_CARD = 0.8

# SpamAssassin-style BAYES_<pct> bins, named by each bin's lower bound * 100.
BAYES_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.01, "BAYES_00"),
    (0.01, 0.05, "BAYES_01"),
    (0.05, 0.20, "BAYES_05"),
    (0.20, 0.50, "BAYES_20"),
    (0.50, 0.80, "BAYES_50"),
    (0.80, 0.95, "BAYES_80"),
    (0.95, 0.99, "BAYES_95"),
    (0.99, 1.01, "BAYES_99"),  # upper bound > 1.0 so p_lore == 1.0 lands here
)

TARGET_MAX_WEIGHT = 0.3  # keeps fitted weights on the same scale as today's hand-tuned ones

MIN_TOKEN_LENGTH = 3
STRONG_LORE_PROBABILITY = 0.75
WEAK_LORE_PROBABILITY = 0.4
MAX_SUGGESTIONS = 30

# Common English function words and the exact "ordinary word" clues issue #95
# calls out (that's, works, running, since): tokens that pass this list still
# need a digit to qualify, since an all-letters token this common is unlikely
# to be a domain term even with strong lore evidence.
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "about",
        "after",
        "again",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "back",
        "be",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "came",
        "can",
        "come",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "down",
        "during",
        "each",
        "even",
        "every",
        "first",
        "for",
        "from",
        "get",
        "go",
        "going",
        "gone",
        "got",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "know",
        "like",
        "made",
        "make",
        "many",
        "may",
        "me",
        "might",
        "more",
        "most",
        "much",
        "must",
        "my",
        "need",
        "new",
        "no",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "one",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "really",
        "said",
        "same",
        "say",
        "see",
        "she",
        "should",
        "since",
        "so",
        "some",
        "still",
        "such",
        "take",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "things",
        "think",
        "this",
        "those",
        "thought",
        "through",
        "time",
        "to",
        "too",
        "under",
        "up",
        "us",
        "use",
        "used",
        "very",
        "want",
        "was",
        "way",
        "we",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "without",
        "working",
        "works",
        "would",
        "yeah",
        "yes",
        "yet",
        "you",
        "your",
        "running",
    ]
)


class NoLabelsError(Exception):
    pass


# --- feature construction ----------------------------------------------------


def _signal_tokens(tokens: frozenset[str]) -> frozenset[str]:
    return frozenset(token for token in tokens if token.startswith("SIG_"))


def bayes_bin(p_lore: float) -> str:
    """The SpamAssassin-style `BAYES_<pct>` bucket `p_lore` falls in."""
    for low, high, name in BAYES_BINS:
        if low <= p_lore < high:
            return name
    return BAYES_BINS[-1][2]  # pragma: no cover - unreachable, bins cover [0, 1] inclusive


def _fold_of(exchange_id: int, folds: int = FOLDS) -> int:
    """Same hashing scheme as `infovore.triage.bayes.in_holdout`
    (`sha256(str(exchange_id))`'s first byte), generalized from a single
    holdout bucket to `folds` buckets."""
    digest = hashlib.sha256(str(exchange_id).encode()).digest()
    return digest[0] % folds


def _combined_features(
    example: Example, p_lore_by_exchange: Mapping[int, float] | None
) -> frozenset[str]:
    active = set(_signal_tokens(example.tokens))
    if p_lore_by_exchange is not None and example.exchange_id in p_lore_by_exchange:
        active.add(bayes_bin(p_lore_by_exchange[example.exchange_id]))
    active.add(f"CHAN_{example.channel_id}")
    return frozenset(active)


def _fit_signal_weights(examples: Sequence[Example]) -> LogisticModel:
    return train_logistic(
        [(_signal_tokens(example.tokens), example.label is Label.LORE) for example in examples]
    )


def _fit_combined_weights(
    examples: Sequence[Example], p_lore_by_exchange: Mapping[int, float] | None
) -> LogisticModel:
    return train_logistic(
        [
            (_combined_features(example, p_lore_by_exchange), example.label is Label.LORE)
            for example in examples
        ]
    )


def _labeled_p_lore(conn: sqlite3.Connection, examples: Sequence[Example]) -> dict[int, float]:
    """`{exchange_id: p_lore}` for every one of `examples` that already has a
    `p_lore` (only ever called with a non-empty `examples`, from callers that
    already checked `NoLabelsError` themselves)."""
    ids = [example.exchange_id for example in examples]
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, p_lore FROM exchanges WHERE id IN ({placeholders}) AND p_lore IS NOT NULL",
        ids,
    ).fetchall()
    return {row["id"]: row["p_lore"] for row in rows}


# --- deliverable 2: --signal-report ------------------------------------------


@dataclass(frozen=True)
class SignalStats:
    name: str
    field: str
    fires_lore: int
    fires_noise: int
    fires_lore_rate: float
    fires_noise_rate: float
    precision: float
    lift: float
    current_weight: float
    fitted_weight: float
    flag: str  # "", "insufficient-data", "useless", or "harmful"


def _flag_signal(total_fires: int, lift: float, current_weight: float, rate_diff: float) -> str:
    if total_fires < MIN_SIGNAL_SUPPORT:
        return "insufficient-data"
    if abs(lift - 1.0) <= USELESS_LIFT_BAND:
        return "useless"
    sign_mismatch = (current_weight > 0 and rate_diff < 0) or (current_weight < 0 and rate_diff > 0)
    if sign_mismatch:
        return "harmful"
    return ""


def signal_report(conn: sqlite3.Connection, rules: TriageRules) -> tuple[SignalStats, ...]:
    """Per-rule-signal statistics over every labeled exchange: how often it
    fires on lore vs noise, its precision, its lift over the base lore rate,
    its current (rules.toml) weight, and a weight fit from the labels alone
    (`_fit_signal_weights`) — SpamAssassin-style rule diagnostics. Flags a
    signal `"useless"` (lift close to 1: firing tells you almost nothing
    about lore vs noise) or `"harmful"` (its current weight's sign disagrees
    with which class it actually fires more on), with fewer than
    `MIN_SIGNAL_SUPPORT` total fires flagged `"insufficient-data"` instead of
    either, since lift is noisy at very low counts.

    Raises `NoLabelsError` if no exchange has an effective label yet.
    """
    examples = build_examples(conn, rules)
    if not examples:
        raise NoLabelsError
    total_lore = sum(1 for example in examples if example.label is Label.LORE)
    total_noise = len(examples) - total_lore
    base_rate = total_lore / len(examples)
    fitted = _fit_signal_weights(examples)

    stats: list[SignalStats] = []
    for name, field in SIGNAL_WEIGHT_FIELDS:
        token = f"SIG_{name}"
        fires_lore = sum(
            1 for example in examples if token in example.tokens and example.label is Label.LORE
        )
        fires_noise = sum(
            1 for example in examples if token in example.tokens and example.label is Label.NOISE
        )
        total_fires = fires_lore + fires_noise
        fires_lore_rate = fires_lore / total_lore if total_lore else 0.0
        fires_noise_rate = fires_noise / total_noise if total_noise else 0.0
        precision = fires_lore / total_fires if total_fires else 0.0
        lift = precision / base_rate if base_rate else 0.0
        current_weight = float(getattr(rules, field))
        stats.append(
            SignalStats(
                name=name,
                field=field,
                fires_lore=fires_lore,
                fires_noise=fires_noise,
                fires_lore_rate=fires_lore_rate,
                fires_noise_rate=fires_noise_rate,
                precision=precision,
                lift=lift,
                current_weight=current_weight,
                fitted_weight=fitted.weights.get(token, 0.0),
                flag=_flag_signal(
                    total_fires, lift, current_weight, fires_lore_rate - fires_noise_rate
                ),
            )
        )
    return tuple(stats)


# --- deliverable 3: --suggest-terms -------------------------------------------


@dataclass(frozen=True)
class TermCandidate:
    token: str
    lore_count: int
    noise_count: int
    probability: float


@dataclass(frozen=True)
class SuggestTermsResult:
    additions: tuple[TermCandidate, ...]
    drops: tuple[TermCandidate, ...]
    snippet: str


_LITERAL_TERM = re.compile(r"^[a-z0-9]+$")
_VIRTUAL_TOKEN_PREFIXES = ("SIG_", "CHAN_", "LEN_")


def _is_domain_ish(token: str) -> bool:
    if len(token) < MIN_TOKEN_LENGTH:
        return False
    has_digit = any(character.isdigit() for character in token)
    return has_digit or token not in STOPWORDS


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _domain_terms_snippet(terms: Sequence[str]) -> str:
    lines = ["domain_terms = ["]
    lines.extend(f"    {_toml_string(term)}," for term in terms)
    lines.append("]")
    return "\n".join(lines)


def suggest_terms(
    conn: sqlite3.Connection, rules: TriageRules, min_support: int = MIN_SIGNAL_SUPPORT
) -> SuggestTermsResult:
    """Candidate `domain_terms` additions (strong lore evidence in the
    trained Bayes model's per-token counts, domain-ish looking, and not
    already matched by a current `domain_terms` rule) and drop candidates
    (a current *literal* domain term whose own token counts don't actually
    predict lore), plus a ready-to-paste `domain_terms = [...]` TOML
    snippet with the additions appended.

    Reuses the already-trained Bayes model's `triage_tokens` counts rather
    than rescanning raw messages: those counts already are "tokens with
    lore evidence", exactly what this is looking for. Raises
    `NoTrainedModelError` if no model has been trained yet.
    """
    loaded = load_latest_model(conn)
    if loaded is None:
        raise NoTrainedModelError
    _, model = loaded
    domain_pattern = re.compile(rf"\b(?:{'|'.join(rules.domain_terms)})\b", re.IGNORECASE)

    additions: list[TermCandidate] = []
    for token, (lore_count, noise_count) in model.counts.items():
        if token.startswith(_VIRTUAL_TOKEN_PREFIXES):
            continue
        support = lore_count + noise_count
        if support < min_support:
            continue
        if not _is_domain_ish(token):
            continue
        if domain_pattern.search(token):
            continue
        probability = bayes_token_probability(model, token)
        if probability < STRONG_LORE_PROBABILITY:
            continue
        additions.append(TermCandidate(token, lore_count, noise_count, probability))
    additions.sort(key=lambda candidate: candidate.probability, reverse=True)
    additions = additions[:MAX_SUGGESTIONS]

    drops: list[TermCandidate] = []
    for term in rules.domain_terms:
        if not _LITERAL_TERM.match(term):
            continue  # a regex fragment (e.g. `ip\d{2}`) isn't one token to look up
        counts = model.counts.get(term)
        if counts is None:
            continue
        lore_count, noise_count = counts
        support = lore_count + noise_count
        if support < min_support:
            continue
        probability = bayes_token_probability(model, term)
        if probability > WEAK_LORE_PROBABILITY:
            continue
        drops.append(TermCandidate(term, lore_count, noise_count, probability))
    drops.sort(key=lambda candidate: candidate.probability)
    drops = drops[:MAX_SUGGESTIONS]

    snippet = _domain_terms_snippet((*rules.domain_terms, *(c.token for c in additions)))
    return SuggestTermsResult(additions=tuple(additions), drops=tuple(drops), snippet=snippet)


# --- deliverable 4: --fit-weights ---------------------------------------------


@dataclass(frozen=True)
class WeightChange:
    name: str
    field: str
    before: float
    after: float


@dataclass(frozen=True)
class FitWeightsResult:
    rules_toml: str
    changes: tuple[WeightChange, ...]
    class_imbalance: float
    class_imbalance_warning: bool


CLASS_IMBALANCE_WARNING_THRESHOLD = 0.7


def _scale_signal_weights(fitted: LogisticModel) -> dict[str, float]:
    magnitudes = [abs(fitted.weights.get(f"SIG_{name}", 0.0)) for name, _ in SIGNAL_WEIGHT_FIELDS]
    largest = max(magnitudes, default=0.0)
    scale = TARGET_MAX_WEIGHT / largest if largest else 1.0
    return {
        field: round(scale * fitted.weights.get(f"SIG_{name}", 0.0), 4)
        for name, field in SIGNAL_WEIGHT_FIELDS
    }


def render_rules_toml(rules: TriageRules, overrides: Mapping[str, float]) -> str:
    """A complete, valid rules TOML: every `TriageRules` field (skipping the
    derived `version`), with `overrides` substituted for the named fields
    and everything else copied from `rules` unchanged. Loadable by
    `infovore.triage.rules.load_rules`, by construction (it writes exactly
    the keys `parse_rules` requires, using reflection over `TriageRules`'s
    own fields rather than a hand-kept key list, so it can't drift out of
    sync with the schema)."""
    lines = [
        "# Generated by `infovore triage --fit-weights` (issue #95).",
        "# Signal weights below are fitted from labeled data; term lists, caps, and",
        "# thresholds are copied unchanged from the rules this was fit against. A human",
        "# should review this (and the before/after weights table) before pointing",
        "# INFOVORE_TRIAGE_RULES at it.",
        "",
    ]
    for field in fields(TriageRules):
        if field.name == "version":
            continue
        value = overrides.get(field.name, getattr(rules, field.name))
        if field.name == "agreement_emoji":
            items = ", ".join(_toml_string(item) for item in sorted(value))
            lines.append(f"{field.name} = [{items}]")
        elif isinstance(value, tuple):
            items = ", ".join(_toml_string(item) for item in value)
            lines.append(f"{field.name} = [{items}]")
        elif isinstance(value, bool):  # pragma: no cover - no bool-typed rules fields today
            lines.append(f"{field.name} = {'true' if value else 'false'}")
        elif isinstance(value, int):
            lines.append(f"{field.name} = {value}")
        else:
            lines.append(f"{field.name} = {value!r}")
    return "\n".join(lines) + "\n"


def fit_weights(conn: sqlite3.Connection, rules: TriageRules) -> FitWeightsResult:
    """Fit a combined (rule signals + binned `p_lore`, when a model exists,
    + channel) logistic model on every labeled exchange, and write out a
    rules TOML with its `SIG_<name>` coefficients substituted for the
    matching weight/penalty fields (scaled so the largest fitted signal
    weight matches `TARGET_MAX_WEIGHT`, keeping the additive score on
    roughly today's 0-1ish scale — ranking, which is what the gate
    threshold is re-derived from, is invariant to that positive rescaling).
    Fitting jointly with `p_lore` first (when available) keeps a rule's
    fitted weight from re-claiming credit the Bayes model already explains
    for the same exchanges.

    Raises `NoLabelsError` if no exchange has an effective label yet.
    """
    examples = build_examples(conn, rules)
    if not examples:
        raise NoLabelsError
    p_lore_by_exchange = _labeled_p_lore(conn, examples) if load_latest_model(conn) else None
    fitted = _fit_combined_weights(examples, p_lore_by_exchange)
    scaled = _scale_signal_weights(fitted)

    changes = tuple(
        WeightChange(
            name=name, field=field, before=float(getattr(rules, field)), after=scaled[field]
        )
        for name, field in SIGNAL_WEIGHT_FIELDS
    )
    lore_count = sum(1 for example in examples if example.label is Label.LORE)
    class_imbalance = max(lore_count, len(examples) - lore_count) / len(examples)

    return FitWeightsResult(
        rules_toml=render_rules_toml(rules, scaled),
        changes=changes,
        class_imbalance=class_imbalance,
        class_imbalance_warning=class_imbalance > CLASS_IMBALANCE_WARNING_THRESHOLD,
    )


# --- deliverable 5: report card ----------------------------------------------


@dataclass(frozen=True)
class ScoreCard:
    label: str
    auc: float | None
    threshold: float | None
    recall_at_threshold: float | None
    corpus_share: float | None


@dataclass(frozen=True)
class ReportCard:
    rule_score: ScoreCard
    p_lore: ScoreCard | None
    fitted_combination: ScoreCard


def _score_card(label: str, scored: Sequence[tuple[float, Label]], min_recall: float) -> ScoreCard:
    area = auc(scored)
    table = evaluate(scored, candidate_thresholds(scored))
    metric = recommend_threshold(table, min_recall)
    return ScoreCard(
        label=label,
        auc=area,
        threshold=metric.threshold if metric else None,
        recall_at_threshold=metric.recall if metric else None,
        corpus_share=None,
    )


def _corpus_share(conn: sqlite3.Connection, column: str, threshold: float) -> float:
    # `column` is always one of the two literals below, never user input.
    assert column in ("triage_score", "p_lore")
    total = int(
        conn.execute(f"SELECT COUNT(*) FROM exchanges WHERE {column} IS NOT NULL").fetchone()[0]
    )
    if not total:  # pragma: no cover - unreachable: `threshold` only exists when >=1 row scored it
        return 0.0
    passing = int(
        conn.execute(
            f"SELECT COUNT(*) FROM exchanges WHERE {column} >= ?", (threshold,)
        ).fetchone()[0]
    )
    return passing / total


def _labeled_column_scores(
    conn: sqlite3.Connection, exchange_ids: Sequence[int], labels: Mapping[int, Label], column: str
) -> list[tuple[float, Label]]:
    # `column` is always one of the two literals below, never user input.
    assert column in ("triage_score", "p_lore")
    if not exchange_ids:  # pragma: no cover - unreachable: callers already checked labels exist
        return []
    placeholders = ",".join("?" * len(exchange_ids))
    rows = conn.execute(
        f"SELECT id, {column} AS value FROM exchanges"
        f" WHERE id IN ({placeholders}) AND {column} IS NOT NULL",
        exchange_ids,
    ).fetchall()
    return [(row["value"], labels[row["id"]]) for row in rows]


def compute_report_card(
    conn: sqlite3.Connection, rules: TriageRules, min_recall: float = MIN_RECALL_FOR_REPORT_CARD
) -> ReportCard | None:
    """Compares the rule score, `p_lore` (if a model is trained), and a
    freshly cross-validated combined score against the labels: each score's
    AUC, the highest threshold reaching `min_recall` recall on its own
    scored labels, and the corpus share (of every exchange that score has
    been computed for) passing that threshold. `None` when there are no
    labels at all — a report card needs something to score against.

    The combined score's AUC is 5-fold cross-validated (by
    `sha256(exchange_id)`, generalizing the existing single holdout split):
    it is fit fresh from the labels being evaluated, so an in-sample AUC
    would be inflated. The rule score isn't fit from labels at all (no
    leakage to guard against), and `p_lore` is evaluated over every labeled
    exchange's already-stored score, which can be mildly optimistic for the
    (roughly four-fifths of) labels the classifier itself trained on --
    documented in the README rather than adding a second holdout-only
    codepath for one column.
    """
    examples = build_examples(conn, rules)
    if not examples:
        return None
    labels = {example.exchange_id: example.label for example in examples}
    exchange_ids = list(labels)

    rule_scored = _labeled_column_scores(conn, exchange_ids, labels, "triage_score")
    rule_card = _score_card("rule score", rule_scored, min_recall)
    rule_card = ScoreCard(
        rule_card.label,
        rule_card.auc,
        rule_card.threshold,
        rule_card.recall_at_threshold,
        _corpus_share(conn, "triage_score", rule_card.threshold)
        if rule_card.threshold is not None
        else None,
    )

    p_lore_card: ScoreCard | None = None
    has_model = load_latest_model(conn) is not None
    if has_model:
        p_lore_scored = _labeled_column_scores(conn, exchange_ids, labels, "p_lore")
        if p_lore_scored:
            card = _score_card("p_lore", p_lore_scored, min_recall)
            p_lore_card = ScoreCard(
                card.label,
                card.auc,
                card.threshold,
                card.recall_at_threshold,
                _corpus_share(conn, "p_lore", card.threshold)
                if card.threshold is not None
                else None,
            )

    p_lore_by_exchange = _labeled_p_lore(conn, examples) if has_model else None
    fold_ids = {example.exchange_id: _fold_of(example.exchange_id) for example in examples}
    pooled: list[tuple[float, Label]] = []
    for fold in range(FOLDS):
        train_examples = [
            (_combined_features(example, p_lore_by_exchange), example.label is Label.LORE)
            for example in examples
            if fold_ids[example.exchange_id] != fold
        ]
        held_out = [example for example in examples if fold_ids[example.exchange_id] == fold]
        if not held_out:
            continue
        model = train_logistic(train_examples)
        pooled.extend(
            (predict_proba(model, _combined_features(example, p_lore_by_exchange)), example.label)
            for example in held_out
        )

    # `pooled` always has exactly `len(examples)` entries: each example falls
    # in exactly one fold, and every fold's held-out examples are pooled, so
    # (given `examples` is non-empty, checked above) at least one fold's
    # `held_out` — and so `pooled` — is always non-empty.
    combination_card = _score_card("fitted combination", pooled, min_recall)
    share = None
    if combination_card.threshold is not None:
        final_model = _fit_combined_weights(examples, p_lore_by_exchange)
        rows = conn.execute(
            "SELECT triage_reasons, p_lore, channel_id FROM exchanges"
            " WHERE triage_reasons IS NOT NULL"
        ).fetchall()
        if rows:
            passing = 0
            for row in rows:
                active = {
                    f"SIG_{name}"
                    for name, _ in json.loads(row["triage_reasons"])
                    if name != "channel_prior"
                }
                if has_model and row["p_lore"] is not None:
                    active.add(bayes_bin(row["p_lore"]))
                active.add(f"CHAN_{row['channel_id']}")
                if predict_proba(final_model, frozenset(active)) >= combination_card.threshold:
                    passing += 1
            share = passing / len(rows)
    fitted_card = ScoreCard(
        combination_card.label,
        combination_card.auc,
        combination_card.threshold,
        combination_card.recall_at_threshold,
        share,
    )

    return ReportCard(rule_score=rule_card, p_lore=p_lore_card, fitted_combination=fitted_card)


__all__ = [
    "BAYES_BINS",
    "CLASS_IMBALANCE_WARNING_THRESHOLD",
    "MAX_SUGGESTIONS",
    "MIN_RECALL_FOR_REPORT_CARD",
    "MIN_SIGNAL_SUPPORT",
    "SIGNAL_WEIGHT_FIELDS",
    "STOPWORDS",
    "USELESS_LIFT_BAND",
    "FitWeightsResult",
    "NoLabelsError",
    "ReportCard",
    "ScoreCard",
    "SignalStats",
    "SuggestTermsResult",
    "TermCandidate",
    "WeightChange",
    "bayes_bin",
    "compute_report_card",
    "fit_weights",
    "render_rules_toml",
    "signal_report",
    "suggest_terms",
]
