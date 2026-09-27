import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from infovore.rows import AttachmentRow, Label, MessageRow, ReactionRow
from infovore.triage.rules import DEFAULT_RULES, TriageRules
from infovore.triage.score import score_exchange

UNKNOWN_WORD_STRENGTH = 1.0
UNKNOWN_WORD_PROBABILITY = 0.5
MINIMUM_PROBABILITY_STRENGTH = 0.1
MAX_CLUES = 150
HOLDOUT_BUCKETS = 5
TOKEN = re.compile(r"[\w/][\w'./+-]*")
TRAILING_PUNCTUATION = ".,!?;:)'\"-"
MAX_TOKEN_LENGTH = 40

FORMAT_MIN_DIGITS = 6
FORMAT_MAX_DIGITS = 17

__all__ = [
    "Label",
    "Metrics",
    "Model",
    "auc",
    "candidate_thresholds",
    "chi2q",
    "evaluate",
    "features",
    "format_threshold",
    "in_holdout",
    "p_lore",
    "recommend_threshold",
    "token_probability",
    "train",
]


@dataclass
class Model:
    lore_documents: int = 0
    noise_documents: int = 0
    counts: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class Metrics:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def _length_bucket(count: int) -> str:
    if count <= 1:
        return "LEN_1"
    if count <= 5:
        return "LEN_2-5"
    if count <= 20:
        return "LEN_6-20"
    return "LEN_21+"


def features(
    messages: Sequence[MessageRow],
    channel_id: int,
    reactions: Sequence[ReactionRow] = (),
    attachments: Sequence[AttachmentRow] = (),
    rules: TriageRules = DEFAULT_RULES,
) -> frozenset[str]:
    words = {
        token.rstrip(TRAILING_PUNCTUATION)
        for message in messages
        for token in TOKEN.findall(message.content.lower())
    }
    words = {word for word in words if word and len(word) <= MAX_TOKEN_LENGTH}
    triage = score_exchange(messages, reactions, attachments, rules)
    virtual = {f"SIG_{name}" for name, _ in triage.reasons}
    return frozenset(words | virtual | {f"CHAN_{channel_id}", _length_bucket(len(messages))})


def train(examples: Iterable[tuple[frozenset[str], Label]]) -> Model:
    model = Model()
    for tokens, label in examples:
        if label is Label.LORE:
            model.lore_documents += 1
        else:
            model.noise_documents += 1
        for token in tokens:
            lore, noise = model.counts.get(token, (0, 0))
            model.counts[token] = (lore + 1, noise) if label is Label.LORE else (lore, noise + 1)
    return model


def token_probability(model: Model, token: str) -> float:
    if model.lore_documents == 0 or model.noise_documents == 0:
        return UNKNOWN_WORD_PROBABILITY
    lore, noise = model.counts.get(token, (0, 0))
    seen = lore + noise
    if seen == 0:
        return UNKNOWN_WORD_PROBABILITY
    lore_ratio = lore / model.lore_documents
    noise_ratio = noise / model.noise_documents
    raw = lore_ratio / (lore_ratio + noise_ratio)
    return (UNKNOWN_WORD_STRENGTH * UNKNOWN_WORD_PROBABILITY + seen * raw) / (
        UNKNOWN_WORD_STRENGTH + seen
    )


def chi2q(x2: float, degrees: int) -> float:
    half = x2 / 2.0
    term = math.exp(-half)
    total = term
    for index in range(1, degrees // 2):
        term *= half / index
        total += term
    return min(total, 1.0)


def p_lore(model: Model, tokens: frozenset[str], max_clues: int = MAX_CLUES) -> float:
    clues = sorted(
        (
            (abs(probability - 0.5), token, probability)
            for token in tokens
            if abs((probability := token_probability(model, token)) - 0.5)
            >= MINIMUM_PROBABILITY_STRENGTH
        ),
        reverse=True,
    )[:max_clues]
    if not clues:
        return 0.5
    lore_evidence = sum(math.log(1.0 - probability) for _, _, probability in clues)
    noise_evidence = sum(math.log(probability) for _, _, probability in clues)
    degrees = 2 * len(clues)
    lore_strength = 1.0 - chi2q(-2.0 * lore_evidence, degrees)
    noise_strength = 1.0 - chi2q(-2.0 * noise_evidence, degrees)
    return (lore_strength - noise_strength + 1.0) / 2.0


def in_holdout(exchange_id: int) -> bool:
    digest = hashlib.sha256(str(exchange_id).encode()).digest()
    return digest[0] % HOLDOUT_BUCKETS == 0


def evaluate(scored: Sequence[tuple[float, Label]], thresholds: Sequence[float]) -> list[Metrics]:
    table: list[Metrics] = []
    for threshold in thresholds:
        tp = sum(1 for p, label in scored if p >= threshold and label is Label.LORE)
        fp = sum(1 for p, label in scored if p >= threshold and label is Label.NOISE)
        fn = sum(1 for p, label in scored if p < threshold and label is Label.LORE)
        tn = sum(1 for p, label in scored if p < threshold and label is Label.NOISE)
        table.append(Metrics(threshold, tp=tp, fp=fp, fn=fn, tn=tn))
    return table


def auc(scored: Sequence[tuple[float, Label]]) -> float | None:
    """Area under the ROC curve for `scored`, computed rank-based (the
    Mann-Whitney U statistic) rather than by integrating `evaluate` over a
    threshold grid: exact regardless of how scores cluster or saturate, and
    ties (equal scores across the two classes) are handled by giving each
    member of a tied block the block's average rank, the standard tie
    correction (a tie contributes exactly 0.5 per pair, as it should — a
    threshold can't tell them apart).

    Returns `None` when either class is empty in `scored` (issue #95's
    report card: AUC is undefined with no labels of one class, distinct from
    a real 0.0 or 1.0)."""
    positive_count = sum(1 for _, label in scored if label is Label.LORE)
    negative_count = len(scored) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ranked = sorted(scored, key=lambda pair: pair[0])
    total = len(ranked)
    positive_rank_sum = 0.0
    index = 0
    while index < total:
        end = index
        while end + 1 < total and ranked[end + 1][0] == ranked[index][0]:
            end += 1
        # 1-based ranks index+1..end+1, averaged over the tied block.
        average_rank = (index + 1 + end + 1) / 2.0
        for tied in range(index, end + 1):
            if ranked[tied][1] is Label.LORE:
                positive_rank_sum += average_rank
        index = end + 1
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def recommend_threshold(table: Sequence[Metrics], min_recall: float) -> Metrics | None:
    qualifying = [row for row in table if row.recall >= min_recall]
    return max(qualifying, key=lambda row: row.threshold) if qualifying else None


def candidate_thresholds(scored: Sequence[tuple[float, Label]]) -> list[float]:
    """Distinct p_lore scores present in ``scored``, sorted ascending.

    Fisher-combined p_lore saturates near 0.0 and 1.0, so a fixed grid (e.g.
    0.1..0.9) can't express the cutoffs a well-trained model needs (holdout
    recall of 90% might require a threshold like 0.9994). Evaluating exactly
    the holdout's own scores as candidate thresholds lets `recommend_threshold`
    pick the highest one that still meets a recall target, using the same
    ``>=`` semantics as the live gate (`infovore.triage.gate.passes_gate`), so
    ties at the same score are handled consistently.
    """
    return sorted({p for p, _ in scored})


def format_threshold(threshold: float, scored: Sequence[tuple[float, Label]]) -> str:
    """Format ``threshold`` with the fewest significant digits (starting at
    `FORMAT_MIN_DIGITS`) that keep every entry of ``scored`` on the same side
    of the ``>=`` gate as the raw float.

    A naive fixed-precision format (e.g. ``.6g``) can round a saturated score
    like 0.9999999999646519 straight to "1", silently discarding the cutoff.
    Increasing precision until the formatted-and-reparsed value reproduces the
    exact same holdout gate outcomes guarantees the printed string is safe to
    paste into INFOVORE_TRIAGE_MIN_P_LORE.
    """
    baseline = [p >= threshold for p, _ in scored]
    for digits in range(FORMAT_MIN_DIGITS, FORMAT_MAX_DIGITS + 1):
        candidate = f"{threshold:.{digits}g}"
        if [p >= float(candidate) for p, _ in scored] == baseline:
            return candidate
    return repr(threshold)  # pragma: no cover - 17 significant digits always round-trips
