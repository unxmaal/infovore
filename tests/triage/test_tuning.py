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
    NoLabelsError,
    bayes_bin,
    compute_report_card,
    fit_weights,
    render_rules_toml,
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


def test_fit_weights_uses_p_lore_bins_when_a_model_is_trained(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_fixture(conn)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_model(conn)  # type: ignore[misc]
    score_all(conn, model, version)

    result = fit_weights(conn, DEFAULT_RULES)

    assert len(result.changes) == 14


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

    result = suggest_terms(conn, DEFAULT_RULES, min_support=5)

    added = {candidate.token for candidate in result.additions}
    assert "octane2000" in added  # domain-ish (has a digit), strong lore evidence, no rule matches
    assert "sgi" not in added  # already matched by the domain_terms rule
    assert "since" not in added  # ordinary stopword, no digit
    assert "r4" not in added  # too short (MIN_TOKEN_LENGTH)
    assert "onlyonce123" not in added  # not enough support
    assert "model99" not in added  # mixed evidence, not "strong"
    assert '"octane2000",' in result.snippet
    assert result.snippet.startswith("domain_terms = [")


def test_suggest_terms_finds_drop_candidates_for_weak_domain_terms(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_suggest_fixture(conn)
    train_and_store(conn, FixedClock(NOW))

    result = suggest_terms(conn, DEFAULT_RULES, min_support=5)

    dropped = {candidate.token for candidate in result.drops}
    assert "gio" in dropped  # existing literal domain term, only ever seen in noise here
    assert "hinv" not in dropped  # existing literal domain term with strong lore evidence
    assert "xfs" not in dropped  # existing literal domain term, but not enough support to judge


def test_suggest_terms_respects_min_support(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_suggest_fixture(conn)
    train_and_store(conn, FixedClock(NOW))

    lenient = suggest_terms(conn, DEFAULT_RULES, min_support=1)

    added = {candidate.token for candidate in lenient.additions}
    assert "onlyonce123" in added


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
