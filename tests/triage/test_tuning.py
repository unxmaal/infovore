import sqlite3
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import insert_exchange
from infovore.db.labels import set_label
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule, Label, LabelSource
from infovore.timing import FixedClock
from infovore.triage.rules import DEFAULT_RULES, parse_rules
from infovore.triage.runner import triage_pending
from infovore.triage.train import NoTrainedModelError, load_latest_model, score_all, train_and_store
from infovore.triage.tuning import (
    UNCERTAIN_SAMPLING_WARNING_THRESHOLD,
    NoLabelsError,
    _is_domain_ish,
    bayes_bin,
    compute_report_card,
    corpus_document_frequencies,
    fit_weights,
    render_rules_toml,
    sampling_bias_warning,
    signal_report,
    suggest_terms,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)

# Every lore exchange fires: domain_terms, irix_version, part_number, unix_path,
# code, archive_link (always informative). Half of each class is padded past
# 400 characters so "substantial" fires on half of both classes equally
# (lift ~= 1, "useless"). 12/20 lore and 2/20 noise additionally link a GIF
# (gif_links is a penalty but fires *more* on lore here: "harmful").
LORE_BASE = "sgi PROM 6.5.22 part 030-1234-001 /usr/sbin/inst `hinv` see https://bitsavers.org/x"
NOISE_BASE = "lol gg no cap totally boring chat nothing here at all"
PADDING = " blah" * 100
GIF_LINK = " https://tenor.com/view/cat-party"


def db(tmp_path: Path) -> sqlite3.Connection:
    tmp_path.mkdir(parents=True, exist_ok=True)
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def seed_exchange(conn: sqlite3.Connection, message_id: int, channel_id: int, content: str) -> int:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 9, 1, 'alice', 0, ?, ?, ?, '{}')",
        (message_id, channel_id, NOW.isoformat(), content, NOW.isoformat()),
    )
    row = ExchangeRow(
        id=None,
        channel_id=channel_id,
        thread_id=None,
        first_message_id=message_id,
        last_message_id=message_id,
        started_at=NOW,
        ended_at=NOW,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=f"hash-{message_id}",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    return insert_exchange(conn, row, [message_id])


def seed_fixture(conn: sqlite3.Connection) -> tuple[list[int], list[int]]:
    """20 lore + 20 noise labeled exchanges (ids 1..40), crafted so
    `signal_report` sees one useless signal, one harmful signal, six always-
    informative signals, and six with no support at all (never fires)."""
    lore_ids: list[int] = []
    for i in range(20):
        message_id = i + 1
        content = LORE_BASE
        if i < 10:
            content += PADDING  # substantial fires on half of lore
        if i < 12:
            content += GIF_LINK  # gif_links fires on 12/20 lore
        exchange_id = seed_exchange(conn, message_id, channel_id=1, content=content)
        set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
        lore_ids.append(exchange_id)

    noise_ids: list[int] = []
    for i in range(20):
        message_id = 100 + i
        content = NOISE_BASE
        if i < 10:
            content += PADDING  # substantial fires on half of noise too
        if i < 2:
            content += GIF_LINK  # gif_links fires on only 2/20 noise
        exchange_id = seed_exchange(conn, message_id, channel_id=2, content=content)
        set_label(conn, exchange_id, Label.NOISE, LabelSource.HUMAN, None, NOW)
        noise_ids.append(exchange_id)

    return lore_ids, noise_ids


def _ensure_prompt_version(conn: sqlite3.Connection, version: str = "v1") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO prompt_versions (version, text_sha256, created_at)"
        " VALUES (?, 'sha', ?)",
        (version, NOW.isoformat()),
    )


def seed_llm_labeled_exchange(
    conn: sqlite3.Connection,
    message_id: int,
    channel_id: int,
    content: str,
    label: Label,
    source_ref: str | None,
    sampled_by: str | None = None,
) -> int:
    """A LORE/NOISE-labeled exchange whose label came (per `source_ref`) from
    a trial extraction run, so `--fit-weights`/`--signal-report`'s sampling-
    origin warning (issue #107) can trace it back to `sampled_by`. Passing a
    `source_ref` that isn't `derive_labels_from_runs`'s own `run:<id> ...`
    format (or `None`) exercises the "provenance unknown" fallback."""
    _ensure_prompt_version(conn)
    exchange_id = seed_exchange(conn, message_id, channel_id, content)
    cursor = conn.execute(
        "INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at, mode,"
        " outcome, sampled_by) VALUES (?, 'm', 'v1', ?, 'trial', 'ok', ?)",
        (exchange_id, NOW.isoformat(), sampled_by),
    )
    run_id = cursor.lastrowid
    ref = source_ref.format(run_id=run_id) if source_ref is not None else None
    set_label(conn, exchange_id, label, LabelSource.LLM, ref, NOW)
    return exchange_id


# --- signal_report ------------------------------------------------------------


def test_signal_report_raises_without_any_labels(tmp_path: Path) -> None:
    conn = db(tmp_path)
    with pytest.raises(NoLabelsError):
        signal_report(conn, DEFAULT_RULES)


def test_signal_report_covers_every_signal_and_flags_correctly(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_fixture(conn)

    stats = signal_report(conn, DEFAULT_RULES)

    by_name = {stat.name: stat for stat in stats}
    assert set(by_name) == {
        "domain_terms",
        "irix_version",
        "part_number",
        "unix_path",
        "code",
        "archive_link",
        "pdf_attachment",
        "answered_question",
        "agreed_answer",
        "thread",
        "substantial",
        "mostly_tiny_messages",
        "gif_links",
        "laughter",
    }

    # Always-informative signals: fire on every lore exchange, never on noise.
    for name in (
        "domain_terms",
        "irix_version",
        "part_number",
        "unix_path",
        "code",
        "archive_link",
    ):
        stat = by_name[name]
        assert stat.fires_lore == 20
        assert stat.fires_noise == 0
        assert stat.precision == 1.0
        assert stat.flag == ""

    # substantial fires equally on both classes: uninformative.
    substantial = by_name["substantial"]
    assert substantial.fires_lore == 10
    assert substantial.fires_noise == 10
    assert substantial.lift == pytest.approx(1.0)
    assert substantial.flag == "useless"

    # gif_links is a penalty (current_weight < 0) but fires mostly on lore here.
    gif_links = by_name["gif_links"]
    assert gif_links.fires_lore == 12
    assert gif_links.fires_noise == 2
    assert gif_links.current_weight < 0
    assert gif_links.flag == "harmful"

    # Never fire in this fixture: single-message exchanges, no attachments,
    # reactions, threads, or pure-laughter messages.
    for name in (
        "pdf_attachment",
        "answered_question",
        "agreed_answer",
        "thread",
        "mostly_tiny_messages",
        "laughter",
    ):
        stat = by_name[name]
        assert stat.fires_lore == 0
        assert stat.fires_noise == 0
        assert stat.flag == "insufficient-data"

    # The fitted (signal-only) weight should broadly agree with the flags:
    # the always-informative signals get a positive fitted weight, and the
    # harmful gif_links signal gets a positive fitted weight too (the data
    # says it predicts lore, contradicting its negative current_weight).
    assert by_name["domain_terms"].fitted_weight > 0
    assert by_name["gif_links"].fitted_weight > 0


# --- render_rules_toml / fit_weights -------------------------------------------


def test_render_rules_toml_round_trips_through_load_rules() -> None:
    text = render_rules_toml(DEFAULT_RULES, overrides={"domain_term_weight": 0.42})

    data = tomllib.loads(text)
    rules = parse_rules(data, source="test")

    assert rules.domain_term_weight == 0.42
    assert rules.domain_terms == DEFAULT_RULES.domain_terms
    assert rules.agreement_emoji == DEFAULT_RULES.agreement_emoji
    assert rules.tiny_penalty == DEFAULT_RULES.tiny_penalty


def test_render_rules_toml_keeps_non_override_fields_unchanged() -> None:
    # `version` itself is expected to differ: agreement_emoji is stored as a
    # frozenset (order lost), and the version hash is sensitive to the raw
    # TOML list order, so any regenerated file gets a fresh version — exactly
    # as intended, since switching to it should trigger a rescore anyway.
    text = render_rules_toml(DEFAULT_RULES, overrides={})
    rules = parse_rules(tomllib.loads(text), source="test")

    for field_name in (name for name in DEFAULT_RULES.__dataclass_fields__ if name != "version"):
        assert getattr(rules, field_name) == getattr(DEFAULT_RULES, field_name), field_name


def test_fit_weights_raises_without_any_labels(tmp_path: Path) -> None:
    conn = db(tmp_path)
    with pytest.raises(NoLabelsError):
        fit_weights(conn, DEFAULT_RULES)


def test_fit_weights_writes_a_loadable_rules_file_with_all_signals_changed(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_fixture(conn)

    result = fit_weights(conn, DEFAULT_RULES)

    rules = parse_rules(tomllib.loads(result.rules_toml), source="test")
    assert rules.domain_terms == DEFAULT_RULES.domain_terms
    assert len(result.changes) == 14
    assert {change.field for change in result.changes} == {
        "domain_term_weight",
        "irix_version_weight",
        "part_number_weight",
        "unix_path_weight",
        "code_weight",
        "archive_link_weight",
        "pdf_attachment_weight",
        "answered_question_weight",
        "agreed_answer_weight",
        "thread_weight",
        "substantial_weight",
        "tiny_penalty",
        "gif_penalty",
        "laughter_penalty",
    }
    for change in result.changes:
        assert change.before == getattr(DEFAULT_RULES, change.field)
        assert change.after == getattr(rules, change.field)


def test_fit_weights_class_imbalance_is_not_flagged_for_balanced_labels(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_fixture(conn)

    result = fit_weights(conn, DEFAULT_RULES)

    assert result.class_imbalance == pytest.approx(0.5)
    assert result.class_imbalance_warning is False


def test_fit_weights_flags_class_imbalance_over_the_threshold(tmp_path: Path) -> None:
    conn = db(tmp_path)
    lore_ids: list[int] = []
    for i in range(15):
        exchange_id = seed_exchange(conn, i + 1, channel_id=1, content=LORE_BASE)
        set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
        lore_ids.append(exchange_id)
    for i in range(2):
        exchange_id = seed_exchange(conn, 100 + i, channel_id=2, content=NOISE_BASE)
        set_label(conn, exchange_id, Label.NOISE, LabelSource.HUMAN, None, NOW)

    result = fit_weights(conn, DEFAULT_RULES)

    assert result.class_imbalance > 0.7
    assert result.class_imbalance_warning is True


def test_fit_weights_fits_rule_signals_alone_not_conditioned_on_p_lore(tmp_path: Path) -> None:
    # The written rules file drives the stand-alone rule score, so its weights
    # must come from the rule signals alone: training a Bayes model and
    # scoring p_lore must not change them.
    without_model = db(tmp_path / "a")
    seed_fixture(without_model)
    baseline = fit_weights(without_model, DEFAULT_RULES)

    with_model = db(tmp_path / "b")
    seed_fixture(with_model)
    train_and_store(with_model, FixedClock(NOW))
    version, model = load_latest_model(with_model)  # type: ignore[misc]
    score_all(with_model, model, version)
    result = fit_weights(with_model, DEFAULT_RULES)

    assert len(result.changes) == 14
    assert result.changes == baseline.changes


# --- sampling_bias_warning / fit_weights's new fields (issue #107) ----------


def test_sampling_bias_warning_raises_without_any_labels(tmp_path: Path) -> None:
    conn = db(tmp_path)
    with pytest.raises(NoLabelsError):
        sampling_bias_warning(conn, DEFAULT_RULES)


def test_sampling_bias_warning_falls_back_to_class_imbalance_when_provenance_is_unknown(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_fixture(conn)  # all labels are HUMAN: no sampling provenance at all

    warning = sampling_bias_warning(conn, DEFAULT_RULES)

    assert warning.uncertain_sampling_share is None
    assert warning.uncertain_sampling_warning is False
    assert warning.class_imbalance == pytest.approx(0.5)
    assert warning.class_imbalance_warning is False


def test_sampling_bias_warning_flags_class_imbalance_when_provenance_unknown_and_skewed(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    for i in range(15):
        exchange_id = seed_exchange(conn, i + 1, channel_id=1, content=LORE_BASE)
        set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
    for i in range(2):
        exchange_id = seed_exchange(conn, 100 + i, channel_id=2, content=NOISE_BASE)
        set_label(conn, exchange_id, Label.NOISE, LabelSource.HUMAN, None, NOW)

    warning = sampling_bias_warning(conn, DEFAULT_RULES)

    assert warning.uncertain_sampling_share is None
    assert warning.class_imbalance > 0.7
    assert warning.class_imbalance_warning is True


def test_sampling_bias_warning_uses_recorded_sampling_origin_when_available(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    origins = ["uncertain", "uncertain", "uncertain", "random"]
    for i, label in enumerate((Label.LORE,) * 4):
        seed_llm_labeled_exchange(
            conn, i + 1, 1, LORE_BASE, label, "run:{run_id} model:m", origins[i]
        )
    for i, label in enumerate((Label.NOISE,) * 4):
        seed_llm_labeled_exchange(
            conn, 100 + i, 2, NOISE_BASE, label, "run:{run_id} model:m", origins[i]
        )

    warning = sampling_bias_warning(conn, DEFAULT_RULES)

    # 4 labels balanced per class (imbalance would not warn), but 6/8 labels
    # trace to `uncertain` sampling: the sampling check, not class imbalance,
    # is what should fire here.
    assert warning.class_imbalance == pytest.approx(0.5)
    assert warning.class_imbalance_warning is False
    assert warning.uncertain_sampling_share == pytest.approx(0.75)
    assert warning.uncertain_sampling_warning is True


def test_sampling_bias_warning_does_not_warn_at_or_below_the_threshold(tmp_path: Path) -> None:
    conn = db(tmp_path)
    # 7 uncertain, 3 random: share is exactly the threshold, not over it.
    origins = ["uncertain"] * 7 + ["random"] * 3
    for i, origin in enumerate(origins):
        label = Label.LORE if i % 2 == 0 else Label.NOISE
        content = LORE_BASE if label is Label.LORE else NOISE_BASE
        seed_llm_labeled_exchange(conn, i + 1, 1, content, label, "run:{run_id} model:m", origin)

    warning = sampling_bias_warning(conn, DEFAULT_RULES)

    assert warning.uncertain_sampling_share == pytest.approx(UNCERTAIN_SAMPLING_WARNING_THRESHOLD)
    assert warning.uncertain_sampling_warning is False


def test_sampling_bias_warning_ignores_llm_labels_with_no_run_reference(tmp_path: Path) -> None:
    conn = db(tmp_path)
    for i in range(6):
        seed_llm_labeled_exchange(conn, i + 1, 1, LORE_BASE, Label.LORE, None)
    for i in range(6):
        seed_llm_labeled_exchange(
            conn, 100 + i, 2, NOISE_BASE, Label.NOISE, "hand-entered, no run id"
        )

    warning = sampling_bias_warning(conn, DEFAULT_RULES)

    assert warning.uncertain_sampling_share is None
    assert warning.uncertain_sampling_warning is False


def test_sampling_bias_warning_ignores_runs_without_a_recorded_sampling_origin(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    for i in range(6):
        seed_llm_labeled_exchange(
            conn, i + 1, 1, LORE_BASE, Label.LORE, "run:{run_id} model:m", sampled_by=None
        )
    for i in range(6):
        seed_llm_labeled_exchange(
            conn, 100 + i, 2, NOISE_BASE, Label.NOISE, "run:{run_id} model:m", sampled_by=None
        )

    warning = sampling_bias_warning(conn, DEFAULT_RULES)

    assert warning.uncertain_sampling_share is None  # every run predates `sampled_by`
    assert warning.class_imbalance == pytest.approx(0.5)


def test_sampling_bias_warning_prefers_a_human_label_over_llm_provenance(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_llm_labeled_exchange(
        conn, 1, 1, LORE_BASE, Label.LORE, "run:{run_id} model:m", "uncertain"
    )
    set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)  # wins, no provenance
    for i in range(5):
        seed_llm_labeled_exchange(
            conn, 100 + i, 2, NOISE_BASE, Label.NOISE, "run:{run_id} model:m", "uncertain"
        )

    warning = sampling_bias_warning(conn, DEFAULT_RULES)

    # The human-labeled exchange is excluded from the known-provenance set
    # entirely, so its `uncertain` run doesn't count toward the share.
    assert warning.uncertain_sampling_share == pytest.approx(1.0)


def test_fit_weights_reports_the_recorded_sampling_origin_when_known(tmp_path: Path) -> None:
    conn = db(tmp_path)
    origins = ["uncertain", "uncertain", "uncertain", "random"]
    for i, label in enumerate((Label.LORE,) * 4):
        seed_llm_labeled_exchange(
            conn, i + 1, 1, LORE_BASE, label, "run:{run_id} model:m", origins[i]
        )
    for i, label in enumerate((Label.NOISE,) * 4):
        seed_llm_labeled_exchange(
            conn, 100 + i, 2, NOISE_BASE, label, "run:{run_id} model:m", origins[i]
        )

    result = fit_weights(conn, DEFAULT_RULES)

    assert result.uncertain_sampling_share == pytest.approx(0.75)
    assert result.uncertain_sampling_warning is True
    assert result.class_imbalance_warning is False


def test_fit_weights_uncertain_sampling_share_is_none_when_provenance_is_unknown(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_fixture(conn)

    result = fit_weights(conn, DEFAULT_RULES)

    assert result.uncertain_sampling_share is None
    assert result.uncertain_sampling_warning is False


# --- suggest_terms --------------------------------------------------------


SUGGEST_LORE = "sgi hinv Octane needs octane2000 diagnostics, since it works, r4 unit"
SUGGEST_NOISE = "lol gg since gio broke again, whatever, r4 too"


def seed_suggest_fixture(conn: sqlite3.Connection) -> None:
    for i in range(20):
        exchange_id = seed_exchange(conn, i + 1, channel_id=1, content=SUGGEST_LORE)
        set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
    for i in range(20):
        exchange_id = seed_exchange(conn, 100 + i, channel_id=2, content=SUGGEST_NOISE)
        set_label(conn, exchange_id, Label.NOISE, LabelSource.HUMAN, None, NOW)
    # A high-evidence lore token that appears in only one document: too little
    # support to suggest even though it looks domain-ish.
    exchange_id = seed_exchange(conn, 500, channel_id=1, content="onlyonce123 special part")
    set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
    # A domain-ish token (has a digit) with genuinely mixed evidence: present
    # in about half of each class, so it shouldn't look "strong" either way.
    for i in range(6):
        exchange_id = seed_exchange(conn, 600 + i, channel_id=1, content="model99 shared context")
        set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
    for i in range(6):
        exchange_id = seed_exchange(conn, 700 + i, channel_id=2, content="model99 shared context")
        set_label(conn, exchange_id, Label.NOISE, LabelSource.HUMAN, None, NOW)
    # An existing literal domain term ("xfs") with too little support (2 <
    # the default min_support of 5) to judge as a drop candidate either way.
    for i in range(2):
        exchange_id = seed_exchange(conn, 800 + i, channel_id=2, content="xfs mount failed, meh")
        set_label(conn, exchange_id, Label.NOISE, LabelSource.HUMAN, None, NOW)


def test_suggest_terms_raises_without_a_trained_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_suggest_fixture(conn)
    with pytest.raises(NoTrainedModelError):
        suggest_terms(conn, DEFAULT_RULES)


def test_suggest_terms_finds_domain_ish_additions_and_excludes_the_rest(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_suggest_fixture(conn)
    train_and_store(conn, FixedClock(NOW))

    # max_corpus_df=1.0 (never trips) here: this test is about the domain-ish
    # / support / rule-match filters, not the corpus-DF filter (covered below).
    result = suggest_terms(conn, DEFAULT_RULES, min_support=5, max_corpus_df=1.0)

    added = {candidate.token for candidate in result.additions}
    assert "octane2000" in added  # domain-ish (has a digit), strong lore evidence, no rule matches
    assert "sgi" not in added  # already matched by the domain_terms rule
    assert "since" not in added  # ordinary stopword, no digit
    assert "r4" not in added  # equally common in lore and noise here, not "strong" evidence
    assert "onlyonce123" not in added  # not enough support
    assert "model99" not in added  # mixed evidence, not "strong"
    assert '"octane2000",' in result.snippet
    assert "domain_terms = [" not in result.snippet  # never a full replacement list (issue #105)


def test_suggest_terms_finds_drop_candidates_for_weak_domain_terms(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_suggest_fixture(conn)
    train_and_store(conn, FixedClock(NOW))

    result = suggest_terms(conn, DEFAULT_RULES, min_support=5, max_corpus_df=1.0)

    dropped = {candidate.token for candidate in result.drops}
    assert "gio" in dropped  # existing literal domain term, only ever seen in noise here
    assert "hinv" not in dropped  # existing literal domain term with strong lore evidence
    assert "xfs" not in dropped  # existing literal domain term, but not enough support to judge


def test_suggest_terms_respects_min_support(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_suggest_fixture(conn)
    train_and_store(conn, FixedClock(NOW))

    lenient = suggest_terms(conn, DEFAULT_RULES, min_support=1, max_corpus_df=1.0)

    added = {candidate.token for candidate in lenient.additions}
    assert "onlyonce123" in added


def test_suggest_terms_snippet_says_no_candidates_when_empty(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_suggest_fixture(conn)
    train_and_store(conn, FixedClock(NOW))

    # A vanishingly small max_corpus_df drops every candidate.
    result = suggest_terms(conn, DEFAULT_RULES, min_support=5, max_corpus_df=0.0)

    assert result.additions == ()
    assert result.snippet == "# (no candidate additions)"


# --- _is_domain_ish -----------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("gcc", True),  # domain-ish: not a stopword, len >= 3
        ("since", False),  # ordinary stopword, no digit
        ("didn't", False),  # contraction/apostrophe token, excluded regardless of length/digits
        ("can't", False),
        ("that's", False),  # apostrophe token that's also a stopword -- still excluded
        ("ok", False),  # pure alphabetic, shorter than 3 chars
        ("a", False),  # pure alphabetic, shorter than 3 chars (and a stopword)
        ("r4", True),  # short, but contains a digit -- no longer disqualified by length alone
        ("o2", True),
    ],
)
def test_is_domain_ish(token: str, expected: bool) -> None:
    assert _is_domain_ish(token) is expected


# --- corpus_document_frequencies ----------------------------------------------


def test_corpus_document_frequencies_counts_each_token_once_per_exchange(tmp_path: Path) -> None:
    conn = db(tmp_path)
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, content, ingested_at, raw_json)"
        " VALUES (1, 1, 9, 1, 'a', 0, ?, 'gcc build gcc again', ?, '{}')",
        (NOW.isoformat(), NOW.isoformat()),
    )
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, content, ingested_at, raw_json)"
        " VALUES (2, 1, 9, 1, 'a', 0, ?, 'gcc again please', ?, '{}')",
        (NOW.isoformat(), NOW.isoformat()),
    )
    row = ExchangeRow(
        id=None,
        channel_id=1,
        thread_id=None,
        first_message_id=1,
        last_message_id=2,
        started_at=NOW,
        ended_at=NOW,
        message_count=2,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash="multi-message",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    insert_exchange(conn, row, [1, 2])
    seed_exchange(conn, 3, channel_id=1, content="lol nothing here")  # no "gcc"

    document_frequency, total_exchanges = corpus_document_frequencies(conn)

    assert total_exchanges == 2
    assert document_frequency["gcc"] == 1  # once per exchange, not per occurrence
    assert document_frequency["again"] == 1
    assert "lol" in document_frequency
    assert "gcc" not in {"lol", "nothing", "here"}  # sanity: exchange 2 has no overlap


def test_corpus_document_frequencies_reports_progress(tmp_path: Path) -> None:
    conn = db(tmp_path)
    for i in range(5):
        seed_exchange(conn, i + 1, channel_id=1, content=f"filler{i}")

    calls: list[tuple[int, int]] = []
    corpus_document_frequencies(
        conn, progress=lambda scanned, total: calls.append((scanned, total)), progress_every=2
    )

    assert (1, 5) not in calls  # not a multiple of progress_every=2
    assert (2, 5) in calls
    assert (4, 5) in calls
    assert calls[-1] == (5, 5)  # final call, unconditional, even off the progress_every cadence


def test_corpus_document_frequencies_without_progress_callback(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, 1, channel_id=1, content="hello world")

    document_frequency, total_exchanges = corpus_document_frequencies(conn)

    assert total_exchanges == 1
    assert document_frequency["hello"] == 1


def test_corpus_document_frequencies_handles_an_empty_database(tmp_path: Path) -> None:
    conn = db(tmp_path)

    document_frequency, total_exchanges = corpus_document_frequencies(conn)

    assert total_exchanges == 0
    assert dict(document_frequency) == {}


def test_corpus_document_frequencies_calls_progress_once_for_an_empty_database(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)

    calls: list[tuple[int, int]] = []
    corpus_document_frequencies(
        conn, progress=lambda scanned, total: calls.append((scanned, total))
    )

    assert calls == [(0, 0)]


# --- issue #113: parallel document-frequency scan ---------------------------


def test_corpus_document_frequencies_workers_two_matches_workers_one(tmp_path: Path) -> None:
    conn = db(tmp_path)
    for i in range(25):
        seed_exchange(conn, i + 1, channel_id=i % 3, content=f"gcc build {i} again octane")

    serial, serial_total = corpus_document_frequencies(conn, workers=1)
    parallel, parallel_total = corpus_document_frequencies(conn, workers=2)

    assert serial_total == parallel_total == 25
    assert dict(serial) == dict(parallel)


def test_corpus_document_frequencies_workers_two_reports_progress_per_batch(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    for i in range(9):
        seed_exchange(conn, i + 1, channel_id=1, content=f"filler{i}")

    calls: list[tuple[int, int]] = []
    corpus_document_frequencies(
        conn,
        progress=lambda scanned, total: calls.append((scanned, total)),
        progress_every=3,
        workers=2,
    )

    assert calls[-1] == (9, 9)
    assert (3, 9) in calls
    assert (6, 9) in calls


def test_corpus_document_frequencies_defaults_to_one_worker(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_exchange(conn, 1, channel_id=1, content="hello world")

    document_frequency, total_exchanges = corpus_document_frequencies(conn)

    assert total_exchanges == 1
    assert document_frequency["hello"] == 1


# --- regression: issue #105 (ordinary words suggested via length bias) -------

REAL_FAILURE_COMMON_WORDS = (
    "later",
    "maybe",
    "stuff",
    "trying",
    "unfortunately",
    "system",
    "install",
)
REAL_FAILURE_LORE_COMMON = (
    "we tried building it later and maybe it needs a system update, stuff kept"
    " failing, trying again unfortunately the install broke"
)
# gcc: real jargon, rare corpus-wide. wouldn't/ok: same rare corpus-wide evidence
# as gcc, but must still be excluded (contraction / short pure-alphabetic token).
REAL_FAILURE_JARGON = "gcc wouldn't ok"
REAL_FAILURE_NOISE = "lol gg nothing here"
REAL_FAILURE_BACKGROUND = "just chatting about random stuff here, nothing to report"

REAL_FAILURE_LORE_COUNT = 20
REAL_FAILURE_NOISE_COUNT = 10  # >= train.MIN_LABELS_PER_CLASS
# How many of the lore exchanges carry the jargon tokens (gcc/wouldn't/ok).
# `--train` holds out about a fifth of labels (infovore.triage.bayes.in_holdout,
# hashed by exchange id) from the counts `suggest_terms` reads, so this is
# picked (and pinned down by a debug run against this exact fixture, since the
# holdout split is a deterministic hash of the exchange id) high enough that
# >= MIN_SIGNAL_SUPPORT of them still survive into the trained model's counts.
REAL_FAILURE_JARGON_COUNT = 6
REAL_FAILURE_BACKGROUND_COUNT = 700
REAL_FAILURE_TOTAL_EXCHANGES = (
    REAL_FAILURE_LORE_COUNT + REAL_FAILURE_NOISE_COUNT + REAL_FAILURE_BACKGROUND_COUNT
)  # == 730


def seed_real_failure_fixture(conn: sqlite3.Connection) -> None:
    """20 long lore exchanges (common English words in all of them, plus
    `gcc`/`wouldn't`/`ok` in the first `REAL_FAILURE_JARGON_COUNT`), 10 short
    noise exchanges, and 700 unlabeled background exchanges -- 730 exchanges
    total, so a token present in all 20 lore exchanges has corpus_df =
    20/730 ~= 0.027 (over the 1% default `--max-corpus-df`), while the
    jargon tokens' surviving (post-holdout) support of 5 gives corpus_df =
    5/730 ~= 0.0068 (under it). Mirrors issue #105's real failure: long lore
    exchanges full of common words, short noise exchanges, and exactly one
    rare piece of real jargon (`gcc`)."""
    for i in range(REAL_FAILURE_LORE_COUNT):
        content = REAL_FAILURE_LORE_COMMON
        if i < REAL_FAILURE_JARGON_COUNT:
            content += " " + REAL_FAILURE_JARGON
        exchange_id = seed_exchange(conn, i + 1, channel_id=1, content=content)
        set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
    for i in range(REAL_FAILURE_NOISE_COUNT):
        exchange_id = seed_exchange(conn, 100 + i, channel_id=2, content=REAL_FAILURE_NOISE)
        set_label(conn, exchange_id, Label.NOISE, LabelSource.HUMAN, None, NOW)
    for i in range(REAL_FAILURE_BACKGROUND_COUNT):
        seed_exchange(conn, 1000 + i, channel_id=3, content=REAL_FAILURE_BACKGROUND)


def test_suggest_terms_filters_common_words_by_corpus_df_and_keeps_rare_jargon(
    tmp_path: Path,
) -> None:
    """Regression test for issue #105's real failure: on real data, ordinary
    words (later, maybe, didn't, stuff, trying, unfortunately, system,
    install) were suggested as domain terms because lore exchanges are
    longer than noise ones, so any common word is more likely to appear in
    one -- the same length bias the Bayes model itself has. The only real
    jargon was `gcc`. With the default `max_corpus_df`, the ordinary words
    (common corpus-wide, not just in lore) are filtered out; `gcc` (rare
    corpus-wide) is still suggested; `wouldn't`/`ok` have exactly the same
    rare corpus-wide evidence as `gcc` but are excluded anyway (a
    contraction, and a short pure-alphabetic token)."""
    conn = db(tmp_path)
    seed_real_failure_fixture(conn)
    train_and_store(conn, FixedClock(NOW))

    result = suggest_terms(conn, DEFAULT_RULES)  # default min_support and max_corpus_df

    added = {candidate.token for candidate in result.additions}
    assert "gcc" in added
    for word in REAL_FAILURE_COMMON_WORDS:
        assert word not in added
    assert "wouldn't" not in added
    assert "ok" not in added

    # corpus_df counts every exchange containing the token, train/holdout split
    # aside -- gcc appears in exactly REAL_FAILURE_JARGON_COUNT exchanges.
    gcc_candidate = next(candidate for candidate in result.additions if candidate.token == "gcc")
    assert gcc_candidate.corpus_df == pytest.approx(
        REAL_FAILURE_JARGON_COUNT / REAL_FAILURE_TOTAL_EXCHANGES
    )


# --- bayes_bin --------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_lore", "expected"),
    [
        (0.0, "BAYES_00"),
        (0.005, "BAYES_00"),
        (0.01, "BAYES_01"),
        (0.05, "BAYES_05"),
        (0.2, "BAYES_20"),
        (0.5, "BAYES_50"),
        (0.8, "BAYES_80"),
        (0.95, "BAYES_95"),
        (0.99, "BAYES_99"),
        (1.0, "BAYES_99"),
    ],
)
def test_bayes_bin_boundaries(p_lore: float, expected: str) -> None:
    assert bayes_bin(p_lore) == expected


# --- compute_report_card ----------------------------------------------------


def test_compute_report_card_is_none_without_any_labels(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert compute_report_card(conn, DEFAULT_RULES) is None


def test_compute_report_card_before_any_model_is_trained(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_fixture(conn)
    triage_pending(conn, rules=DEFAULT_RULES)

    card = compute_report_card(conn, DEFAULT_RULES)

    assert card is not None
    assert card.rule_score.auc is not None
    assert card.rule_score.auc > 0.9
    assert card.p_lore is None
    assert card.fitted_combination is not None
    assert card.fitted_combination.auc is not None


def test_compute_report_card_includes_p_lore_once_a_model_is_trained(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_fixture(conn)
    triage_pending(conn, rules=DEFAULT_RULES)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_model(conn)  # type: ignore[misc]
    score_all(conn, model, version)

    card = compute_report_card(conn, DEFAULT_RULES)

    assert card is not None
    assert card.p_lore is not None
    assert card.p_lore.auc is not None


def test_compute_report_card_p_lore_is_none_when_model_exists_but_hasnt_scored_labels(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_fixture(conn)
    triage_pending(conn, rules=DEFAULT_RULES)
    train_and_store(conn, FixedClock(NOW))
    # Deliberately skip score_all: no labeled exchange has p_lore yet.

    card = compute_report_card(conn, DEFAULT_RULES)

    assert card is not None
    assert card.p_lore is None


def test_compute_report_card_leaves_threshold_and_share_none_when_recall_is_unreachable(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_fixture(conn)
    triage_pending(conn, rules=DEFAULT_RULES)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_model(conn)  # type: ignore[misc]
    score_all(conn, model, version)

    card = compute_report_card(conn, DEFAULT_RULES, min_recall=1.5)

    assert card is not None
    assert card.rule_score.threshold is None
    assert card.rule_score.corpus_share is None
    assert card.p_lore is not None
    assert card.p_lore.threshold is None
    assert card.p_lore.corpus_share is None
    assert card.fitted_combination.threshold is None
    assert card.fitted_combination.corpus_share is None


def test_compute_report_card_handles_folds_with_no_held_out_examples(tmp_path: Path) -> None:
    # Exchange ids 1 and 2 both land in the same sha256-based fold (verified
    # offline), so every other fold is empty: exercises the "skip this fold,
    # nothing to evaluate" path deterministically without needing dozens of
    # exchanges to guarantee every fold is populated by chance.
    conn = db(tmp_path)
    lore_id = seed_exchange(conn, 1, channel_id=1, content=LORE_BASE)
    set_label(conn, lore_id, Label.LORE, LabelSource.HUMAN, None, NOW)
    noise_id = seed_exchange(conn, 2, channel_id=2, content=NOISE_BASE)
    set_label(conn, noise_id, Label.NOISE, LabelSource.HUMAN, None, NOW)

    card = compute_report_card(conn, DEFAULT_RULES)

    assert card is not None
    assert card.fitted_combination is not None
