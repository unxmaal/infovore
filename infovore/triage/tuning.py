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
  predictions are what the `--report` report card cross-validates as "the
  fitted combination". Its coefficients are NOT written by `--fit-weights`:
  the written file drives the stand-alone rule score, which never sees
  `p_lore` or channel features, so its weights come from the signal-only fit.
"""

import hashlib
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields

from infovore.db.batch import BATCH_SIZE, SQLITE_MAX_VARIABLES
from infovore.rows import Label, LabelSource
from infovore.triage.bayes import (
    TOKEN,
    TRAILING_PUNCTUATION,
    auc,
    candidate_thresholds,
    evaluate,
    recommend_threshold,
)
from infovore.triage.bayes import token_probability as bayes_token_probability
from infovore.triage.logistic import LogisticModel, predict_proba, train_logistic
from infovore.triage.parallel import ChunkPool
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

# Default `--max-corpus-df` (issue #105): a candidate is dropped if it appears
# in more than 1% of *all* exchanges, labeled or not. Ordinary English words
# are common everywhere regardless of label -- the label-only strong-evidence
# check above can't tell "genuinely rare jargon" from "a common word that
# happens to appear in most of a handful of long lore exchanges and none of
# the short noise ones" -- but jargon really is rare corpus-wide, so this
# catches what the label-only check can't.
MAX_CORPUS_DF = 0.01

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
    corpus_df: float


@dataclass(frozen=True)
class SuggestTermsResult:
    additions: tuple[TermCandidate, ...]
    drops: tuple[TermCandidate, ...]
    snippet: str


_LITERAL_TERM = re.compile(r"^[a-z0-9]+$")
_VIRTUAL_TOKEN_PREFIXES = ("SIG_", "CHAN_", "LEN_")


def _is_domain_ish(token: str) -> bool:
    # Contractions/possessives (didn't, can't, doesn't, ...) are ordinary
    # English, never domain jargon -- excluded regardless of digits or length.
    if "'" in token:
        return False
    # A short *pure-alphabetic* token is almost always noise (a, an, ok, ...);
    # a short token that also contains a digit (r4, o2) can be a genuine
    # domain term, so length alone no longer disqualifies it.
    if token.isalpha() and len(token) < MIN_TOKEN_LENGTH:
        return False
    has_digit = any(character.isdigit() for character in token)
    return has_digit or token not in STOPWORDS


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _domain_terms_snippet(candidates: Sequence["TermCandidate"]) -> str:
    """TOML lines for `candidates` alone, meant to be pasted *inside* the
    existing `domain_terms = [...]` list in `rules.toml` -- never a full
    replacement list, so a bad paste can't clobber the current terms (issue
    #105). Each line carries the candidate's corpus document frequency as a
    trailing comment, alongside the lore/noise counts printed next to it."""
    if not candidates:
        return "# (no candidate additions)"
    return "\n".join(
        f"    {_toml_string(candidate.token)},  # corpus_df={candidate.corpus_df:.4f}"
        for candidate in candidates
    )


def _tokenize_worker_chunk(items: list[list[str]]) -> Counter[str]:
    """`items`: for each exchange in this chunk, the list of its messages'
    content strings (tokenizing needs no model or rules, so unlike the other
    two workers in this issue, this one needs no `ChunkPool` initializer).
    Returns document-frequency counts local to this chunk alone -- each
    exchange contributes each of its tokens at most once, exactly like the
    single-process version's per-exchange token set did."""
    counter: Counter[str] = Counter()
    for contents in items:
        tokens: set[str] = set()
        for content in contents:
            for token in TOKEN.findall(content.lower()):
                tokens.add(token.rstrip(TRAILING_PUNCTUATION))
        counter.update(tokens)
    return counter


def _chunked_ids(ids: Sequence[int], size: int = SQLITE_MAX_VARIABLES) -> list[Sequence[int]]:
    return [ids[start : start + size] for start in range(0, len(ids), size)]


def _contents_by_exchange(
    conn: sqlite3.Connection, exchange_ids: Sequence[int]
) -> dict[int, list[str]]:
    """`{exchange_id: [message content, ...]}` for each of `exchange_ids`
    (assumed to have no duplicates), chunking the `IN (...)` list to stay
    under SQLite's bound-parameter limit like `infovore.db.batch` does."""
    contents: dict[int, list[str]] = {exchange_id: [] for exchange_id in exchange_ids}
    for chunk in _chunked_ids(exchange_ids):
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT em.exchange_id AS exchange_id, m.content AS content"
            " FROM exchange_messages em JOIN messages m ON m.id = em.message_id"
            f" WHERE em.exchange_id IN ({placeholders}) ORDER BY em.exchange_id",
            chunk,
        )
        for row in rows:
            contents[row["exchange_id"]].append(row["content"])
    return contents


def corpus_document_frequencies(
    conn: sqlite3.Connection,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = BATCH_SIZE,
    workers: int = 1,
) -> tuple[Counter[str], int]:
    """Corpus-wide document frequency for every token the tokenizer
    (`infovore.triage.bayes.TOKEN`/`TRAILING_PUNCTUATION`, the same one
    `infovore.triage.bayes.features` uses) produces: for each token, how
    many *exchanges* -- every exchange in the database, labeled or not --
    contain it at least once (issue #105). Ordinary English words are
    common everywhere regardless of label; jargon is rare corpus-wide, so
    this is what tells the two apart when a labeled-only count can't (a
    word that happens to appear in most of a handful of long lore exchanges
    and none of the short noise ones looks like strong lore evidence by
    label alone, even though it's common in the corpus as a whole).

    A batched pass, `progress_every` exchanges at a time (in `exchanges.id`
    order): each batch's messages are loaded in the main process
    (`_contents_by_exchange`, one set-based query per
    `SQLITE_MAX_VARIABLES`-sized sub-chunk of ids) and handed to
    `ChunkPool` as plain content strings -- tokenizing (issue #113's
    CPU-bound part of this scan) runs in `workers` worker processes when
    `workers > 1`, or directly in-process when it's `1` (the default).
    Memory scales with one `progress_every`-sized batch's messages at a
    time plus the running vocabulary, never the whole corpus -- on the real
    DB (~204k exchanges / ~1.4M messages) this is a handful of sequential,
    chunked table scans rather than one row-at-a-time cursor, but the same
    total amount of work. `progress`, if given, is called with
    `(exchanges_scanned, total_exchanges)` every `progress_every` exchanges
    and once more after the last one, so a caller can print progress on a
    pass over the real DB that can take a while.
    """
    total_exchanges = int(conn.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0])
    exchange_ids = [row[0] for row in conn.execute("SELECT id FROM exchanges ORDER BY id")]
    document_frequency: Counter[str] = Counter()
    scanned = 0

    with ChunkPool(_tokenize_worker_chunk, workers) as pool:
        for start in range(0, len(exchange_ids), progress_every):
            batch_ids = exchange_ids[start : start + progress_every]
            contents = _contents_by_exchange(conn, batch_ids)
            items = [contents[exchange_id] for exchange_id in batch_ids]
            for counter in pool.map_chunks(items):
                document_frequency.update(counter)
            scanned += len(batch_ids)
            if progress is not None and scanned % progress_every == 0:
                progress(scanned, total_exchanges)

    if progress is not None:
        progress(scanned, total_exchanges)

    return document_frequency, total_exchanges


def suggest_terms(
    conn: sqlite3.Connection,
    rules: TriageRules,
    min_support: int = MIN_SIGNAL_SUPPORT,
    max_corpus_df: float = MAX_CORPUS_DF,
    progress: Callable[[int, int], None] | None = None,
    workers: int = 1,
) -> SuggestTermsResult:
    """Candidate `domain_terms` additions (strong lore evidence in the
    trained Bayes model's per-token counts, domain-ish looking, rare enough
    corpus-wide, and not already matched by a current `domain_terms` rule)
    and drop candidates (a current *literal* domain term whose own token
    counts don't actually predict lore), plus a snippet of just the
    candidate-addition TOML lines to paste inside the existing
    `domain_terms = [...]` list.

    Reuses the already-trained Bayes model's `triage_tokens` counts for the
    label-conditional counts and probability (no rescanning of raw messages
    for those) rather than rescanning raw messages: those counts already
    are "tokens with lore evidence", exactly what this is looking for. On
    top of that, a candidate is dropped if it appears in more than
    `max_corpus_df` of *all* exchanges (`corpus_document_frequencies`, a
    single streaming pass over every exchange's messages, labeled or not,
    reported through `progress` if given) -- ordinary English words are
    common everywhere regardless of label, so the label-only evidence above
    can't tell them apart from genuine jargon on its own (issue #105).
    Raises `NoTrainedModelError` if no model has been trained yet.
    """
    loaded = load_latest_model(conn)
    if loaded is None:
        raise NoTrainedModelError
    _, model = loaded
    domain_pattern = re.compile(rf"\b(?:{'|'.join(rules.domain_terms)})\b", re.IGNORECASE)
    document_frequency, total_exchanges = corpus_document_frequencies(
        conn, progress=progress, workers=workers
    )

    def corpus_df_of(token: str) -> float:
        if (
            not total_exchanges
        ):  # pragma: no cover - unreachable: a trained model implies >=1 exchange
            return 0.0
        return document_frequency.get(token, 0) / total_exchanges

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
        corpus_df = corpus_df_of(token)
        if corpus_df > max_corpus_df:
            continue
        probability = bayes_token_probability(model, token)
        if probability < STRONG_LORE_PROBABILITY:
            continue
        additions.append(TermCandidate(token, lore_count, noise_count, probability, corpus_df))
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
        drops.append(TermCandidate(term, lore_count, noise_count, probability, corpus_df_of(term)))
    drops.sort(key=lambda candidate: candidate.probability)
    drops = drops[:MAX_SUGGESTIONS]

    snippet = _domain_terms_snippet(additions)
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
    uncertain_sampling_share: float | None
    uncertain_sampling_warning: bool


CLASS_IMBALANCE_WARNING_THRESHOLD = 0.7
UNCERTAIN_SAMPLING_WARNING_THRESHOLD = 0.7

# infovore.extract.runner.TrialSampleStrategy.UNCERTAIN.value — not imported
# directly to avoid a triage -> extract dependency for one string constant.
_UNCERTAIN_SAMPLING_ORIGIN = "uncertain"
_SOURCE_REF_RUN = re.compile(r"^run:(\d+)\b")


def _sampling_origins(conn: sqlite3.Connection, exchange_ids: Sequence[int]) -> dict[int, str]:
    """`{exchange_id: origin}` for every one of `exchange_ids` (assumed
    non-empty) whose *effective* label traces back to a trial run recorded
    with a sampling origin (`extraction_runs.sampled_by`, issue #107).

    A human label has no sampling provenance of its own and always wins over
    an LLM label for the same exchange (`infovore.db.labels.effective_labels`),
    so that exchange is left out entirely here. An LLM label whose
    `source_ref` isn't the `run:<id> ...` format `derive_labels_from_runs`
    writes (missing, or hand-written some other way), or whose run predates
    `sampled_by` (still `NULL`) or doesn't exist, is left out too — those
    exchanges have "unknown" provenance, not "known and not uncertain"."""
    placeholders = ",".join("?" * len(exchange_ids))
    rows = conn.execute(
        "SELECT exchange_id, source, source_ref FROM exchange_labels"
        f" WHERE exchange_id IN ({placeholders})",
        exchange_ids,
    ).fetchall()
    by_exchange: dict[int, dict[str, str | None]] = {}
    for row in rows:
        by_exchange.setdefault(row["exchange_id"], {})[row["source"]] = row["source_ref"]

    run_id_by_exchange: dict[int, int] = {}
    for exchange_id, by_source in by_exchange.items():
        if LabelSource.HUMAN.value in by_source:
            continue
        source_ref = by_source.get(LabelSource.LLM.value)
        if source_ref is None:
            continue
        match = _SOURCE_REF_RUN.match(source_ref)
        if match is None:
            continue
        run_id_by_exchange[exchange_id] = int(match.group(1))

    if not run_id_by_exchange:
        return {}
    run_ids = sorted(set(run_id_by_exchange.values()))
    run_placeholders = ",".join("?" * len(run_ids))
    sampled_by = {
        row["id"]: row["sampled_by"]
        for row in conn.execute(
            f"SELECT id, sampled_by FROM extraction_runs WHERE id IN ({run_placeholders})",
            run_ids,
        ).fetchall()
        if row["sampled_by"] is not None
    }
    return {
        exchange_id: sampled_by[run_id]
        for exchange_id, run_id in run_id_by_exchange.items()
        if run_id in sampled_by
    }


@dataclass(frozen=True)
class SamplingBiasWarning:
    class_imbalance: float
    class_imbalance_warning: bool
    uncertain_sampling_share: float | None
    uncertain_sampling_warning: bool


def _sampling_bias(conn: sqlite3.Connection, examples: Sequence[Example]) -> SamplingBiasWarning:
    lore_count = sum(1 for example in examples if example.label is Label.LORE)
    class_imbalance = max(lore_count, len(examples) - lore_count) / len(examples)
    class_imbalance_warning = class_imbalance > CLASS_IMBALANCE_WARNING_THRESHOLD

    exchange_ids = [example.exchange_id for example in examples]
    origins = _sampling_origins(conn, exchange_ids)
    if not origins:
        return SamplingBiasWarning(class_imbalance, class_imbalance_warning, None, False)

    uncertain_count = sum(1 for origin in origins.values() if origin == _UNCERTAIN_SAMPLING_ORIGIN)
    uncertain_share = uncertain_count / len(origins)
    return SamplingBiasWarning(
        class_imbalance,
        class_imbalance_warning,
        uncertain_share,
        uncertain_share > UNCERTAIN_SAMPLING_WARNING_THRESHOLD,
    )


def sampling_bias_warning(conn: sqlite3.Connection, rules: TriageRules) -> SamplingBiasWarning:
    """Whether the labeled exchanges `--fit-weights`/`--signal-report` train
    or report on lean too hard on one sampling origin (issue #107) — or, when
    no label traces back to a recorded sampling origin at all (hand labels,
    or labels from before `extraction_runs.sampled_by` existed), the
    class-imbalance fallback the issue names for exactly that case.

    Raises `NoLabelsError` if no exchange has an effective label yet."""
    examples = build_examples(conn, rules)
    if not examples:
        raise NoLabelsError
    return _sampling_bias(conn, examples)


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
    """Fit a logistic model on the rule signals alone (`SIG_<name>` indicators,
    no `p_lore` bins or channel features) over every labeled exchange, and
    write out a rules TOML with those coefficients substituted for the
    matching weight/penalty fields. The written file drives the stand-alone
    rule score, so its weights must not be conditioned on features the rule
    score never sees. Coefficients are scaled so the largest magnitude equals
    `TARGET_MAX_WEIGHT`, keeping the additive score near today's scale. That
    rescaling preserves the fitted model's ranking only approximately: the
    rule score is clamped to [0, 1] and `domain_terms` is weighted per
    distinct term up to `domain_term_cap`, so compare `--report` before and
    after adopting a fitted file.

    Raises `NoLabelsError` if no exchange has an effective label yet.
    """
    examples = build_examples(conn, rules)
    if not examples:
        raise NoLabelsError
    fitted = _fit_signal_weights(examples)
    scaled = _scale_signal_weights(fitted)

    changes = tuple(
        WeightChange(
            name=name, field=field, before=float(getattr(rules, field)), after=scaled[field]
        )
        for name, field in SIGNAL_WEIGHT_FIELDS
    )
    bias = _sampling_bias(conn, examples)

    return FitWeightsResult(
        rules_toml=render_rules_toml(rules, scaled),
        changes=changes,
        class_imbalance=bias.class_imbalance,
        class_imbalance_warning=bias.class_imbalance_warning,
        uncertain_sampling_share=bias.uncertain_sampling_share,
        uncertain_sampling_warning=bias.uncertain_sampling_warning,
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
    "MAX_CORPUS_DF",
    "MAX_SUGGESTIONS",
    "MIN_RECALL_FOR_REPORT_CARD",
    "MIN_SIGNAL_SUPPORT",
    "SIGNAL_WEIGHT_FIELDS",
    "STOPWORDS",
    "UNCERTAIN_SAMPLING_WARNING_THRESHOLD",
    "USELESS_LIFT_BAND",
    "FitWeightsResult",
    "NoLabelsError",
    "ReportCard",
    "SamplingBiasWarning",
    "ScoreCard",
    "SignalStats",
    "SuggestTermsResult",
    "TermCandidate",
    "WeightChange",
    "bayes_bin",
    "compute_report_card",
    "corpus_document_frequencies",
    "fit_weights",
    "render_rules_toml",
    "sampling_bias_warning",
    "signal_report",
    "suggest_terms",
]
