import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import insert_exchange
from infovore.db.labels import set_label
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule, Label, LabelSource
from infovore.timing import FixedClock
from infovore.triage.bayes import Model, token_probability
from infovore.triage.rules import DEFAULT_RULES, parse_rules
from infovore.triage.train import (
    RECALL_TARGETS,
    InsufficientLabelsError,
    NoTrainedModelError,
    TrainReport,
    build_examples,
    load_latest_model,
    model_rules_version,
    recommend,
    recommend_table,
    score_all,
    score_stale,
    train_and_store,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)

LORE_CONTENT = "PROM 6.5.22 part 030-1234-001 /usr/sbin/inst"
NOISE_CONTENT = "lol gg no cap"

CUSTOM_RULES_DATA: dict[str, object] = {
    "domain_term_weight": 0.15,
    "domain_term_cap": 0.45,
    "irix_version_weight": 0.2,
    "part_number_weight": 0.3,
    "unix_path_weight": 0.15,
    "code_weight": 0.15,
    "archive_link_weight": 0.15,
    "pdf_attachment_weight": 0.15,
    "answered_question_weight": 0.2,
    "agreed_answer_weight": 0.05,
    "thread_weight": 0.05,
    "substantial_weight": 0.1,
    "tiny_penalty": -0.2,
    "gif_penalty": -0.1,
    "laughter_penalty": -0.1,
    "substantial_characters": 400,
    "answer_min_characters": 40,
    "tiny_message_characters": 20,
    "tiny_share_threshold": 0.7,
    "laughter_share_threshold": 0.3,
    "domain_terms": ["bananarama"],
    "archive_link_hosts": ["bitsavers"],
    "gif_hosts": ["tenor\\.com"],
    "laughter_tokens": ["lol+"],
    "agreement_emoji": ["✅"],
}
CUSTOM_RULES = parse_rules(CUSTOM_RULES_DATA, source="test")


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def seed_exchange(conn: sqlite3.Connection, index: int, channel_id: int, content: str) -> int:
    message_id = index
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, ?, ?, 'alice', 0, ?, ?, ?, '{}')",
        (message_id, channel_id, 9, message_id, NOW.isoformat(), content, NOW.isoformat()),
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
        content_hash=f"hash-{index}",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    return insert_exchange(conn, row, [message_id])


def seed_labeled_exchanges(
    conn: sqlite3.Connection, lore_count: int, noise_count: int
) -> tuple[list[int], list[int]]:
    lore_ids: list[int] = []
    for i in range(lore_count):
        index = i + 1
        exchange_id = seed_exchange(conn, index, channel_id=1, content=LORE_CONTENT)
        set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
        lore_ids.append(exchange_id)

    noise_ids: list[int] = []
    for i in range(noise_count):
        index = lore_count + i + 1
        exchange_id = seed_exchange(conn, index, channel_id=2, content=NOISE_CONTENT)
        set_label(conn, exchange_id, Label.NOISE, LabelSource.HUMAN, None, NOW)
        noise_ids.append(exchange_id)

    return lore_ids, noise_ids


def test_build_examples_returns_tokens_and_effective_label_per_labeled_exchange(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    lore_ids, noise_ids = seed_labeled_exchanges(conn, 3, 3)

    examples = build_examples(conn)

    assert {example.exchange_id for example in examples} == set(lore_ids) | set(noise_ids)
    lore_example = next(e for e in examples if e.exchange_id == lore_ids[0])
    assert lore_example.label is Label.LORE
    assert lore_example.channel_id == 1
    assert "prom" in lore_example.tokens
    assert "SIG_part_number" in lore_example.tokens


def test_build_examples_respects_a_rules_override(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1000, channel_id=1, content="bananarama forever")
    set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)

    default_examples = build_examples(conn)
    custom_examples = build_examples(conn, rules=CUSTOM_RULES)

    assert "SIG_domain_terms" not in default_examples[0].tokens
    assert "SIG_domain_terms" in custom_examples[0].tokens


def test_train_and_store_refuses_with_fewer_than_ten_labels_per_class(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 9, 20)

    with pytest.raises(InsufficientLabelsError, match="10"):
        train_and_store(conn, FixedClock(NOW))


def test_train_and_store_builds_a_report_and_persists_the_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)

    report = train_and_store(conn, FixedClock(NOW))

    assert isinstance(report, TrainReport)
    assert report.labels_used == 40
    assert report.holdout_size > 0
    assert [metric.threshold for metric in report.metrics] == [
        round(i / 10, 1) for i in range(1, 10)
    ]
    assert (
        report.confusion.tp + report.confusion.fp + report.confusion.fn + report.confusion.tn
        == (report.holdout_size)
    )
    assert 0 < len(report.top_tokens) <= 15
    top = report.top_tokens[0]
    assert abs(top.probability - 0.5) == max(abs(t.probability - 0.5) for t in report.top_tokens)

    row = conn.execute(
        "SELECT version, labels_used, holdout_size, params_json FROM triage_model"
    ).fetchone()
    assert row["version"] == report.version
    assert row["labels_used"] == 40
    assert row["holdout_size"] == report.holdout_size
    params = json.loads(row["params_json"])
    assert params["lore_documents"] > 0
    assert params["noise_documents"] > 0

    token_rows = conn.execute(
        "SELECT COUNT(*) AS n FROM triage_tokens WHERE model_version = ?", (report.version,)
    ).fetchone()
    assert token_rows["n"] > 0


def test_train_and_store_confusion_uses_the_given_threshold(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)

    lenient = train_and_store(conn, FixedClock(NOW), confusion_threshold=0.1)
    strict = train_and_store(conn, FixedClock(NOW), confusion_threshold=0.9)

    assert lenient.confusion.threshold == 0.1
    assert strict.confusion.threshold == 0.9

    off_grid = train_and_store(conn, FixedClock(NOW), confusion_threshold=0.42)
    assert off_grid.confusion.threshold == 0.42


def test_load_latest_model_reconstructs_token_probabilities(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    report = train_and_store(conn, FixedClock(NOW))

    loaded = load_latest_model(conn)
    assert loaded is not None
    version, model = loaded
    assert version == report.version
    assert isinstance(model, Model)
    assert token_probability(model, "prom") > 0.9
    assert token_probability(model, "gg") < 0.1


def test_load_latest_model_returns_none_when_untrained(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert load_latest_model(conn) is None


def test_score_all_writes_p_lore_and_p_lore_model_for_every_exchange(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_model(conn)  # type: ignore[misc]

    scored = score_all(conn, model, version)

    assert scored == 40
    rows = conn.execute("SELECT id, p_lore, p_lore_model FROM exchanges").fetchall()
    assert len(rows) == 40
    for row in rows:
        assert row["p_lore"] is not None
        assert row["p_lore_model"] == version

    lore_row = conn.execute(
        "SELECT p_lore FROM exchanges WHERE id = (SELECT MIN(id) FROM exchanges)"
    ).fetchone()
    noise_row = conn.execute(
        "SELECT p_lore FROM exchanges WHERE id = (SELECT MAX(id) FROM exchanges)"
    ).fetchone()
    assert lore_row["p_lore"] > noise_row["p_lore"]


def test_score_stale_only_rescopes_exchanges_behind_the_latest_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_model(conn)  # type: ignore[misc]

    first_pass = score_stale(conn, model, version)
    assert first_pass == 40

    second_pass = score_stale(conn, model, version)
    assert second_pass == 0

    conn.execute("UPDATE exchanges SET p_lore_model = NULL WHERE id = 1")
    third_pass = score_stale(conn, model, version)
    assert third_pass == 1


def test_recommend_raises_without_a_trained_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    with pytest.raises(NoTrainedModelError):
        recommend(conn, min_recall=0.5)


def test_recommend_returns_threshold_and_expected_share(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_model(conn)  # type: ignore[misc]
    score_all(conn, model, version)

    result = recommend(conn, min_recall=0.5)

    assert result is not None
    metric, share = result
    assert 0.0 <= share <= 1.0
    assert metric.recall >= 0.5


def test_recommend_returns_none_when_no_threshold_meets_recall(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW))

    assert recommend(conn, min_recall=1.01) is None


def test_recommend_evaluates_holdout_scores_not_a_fixed_grid(tmp_path: Path) -> None:
    # With identical content per class, Fisher-combined p_lore saturates the
    # holdout scores near 0 and 1 (well outside a fixed 0.1..0.9 grid).
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_model(conn)  # type: ignore[misc]
    score_all(conn, model, version)

    result = recommend(conn, min_recall=1.0)

    assert result is not None
    metric, share = result
    assert metric.recall == pytest.approx(1.0)
    assert metric.threshold > 0.9
    assert f"{metric.threshold:.1f}" not in {f"{i / 10:.1f}" for i in range(1, 10)}
    assert 0.0 <= share <= 1.0


def test_recommend_share_is_zero_when_no_exchange_has_been_p_lore_scored(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW))
    # Deliberately skip score_all: no exchange has p_lore yet.

    result = recommend(conn, min_recall=0.5)

    assert result is not None
    _, share = result
    assert share == 0.0


def test_recommend_table_raises_without_a_trained_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    with pytest.raises(NoTrainedModelError):
        recommend_table(conn)


def test_recommend_table_covers_the_standard_recall_targets_with_monotonic_thresholds(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_model(conn)  # type: ignore[misc]
    score_all(conn, model, version)

    rows = recommend_table(conn)

    assert [row.min_recall for row in rows] == list(RECALL_TARGETS)
    thresholds = [row.metric.threshold for row in rows if row.metric is not None]
    assert thresholds == sorted(thresholds)
    for row in rows:
        if row.metric is None:
            assert row.share is None
            assert row.formatted_threshold is None
        else:
            assert row.metric.recall >= row.min_recall
            assert row.share is not None
            assert 0.0 <= row.share <= 1.0
            assert row.formatted_threshold is not None
            assert 0.0 < float(row.formatted_threshold) <= 1.0


def test_recommend_table_accepts_custom_recall_targets(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_model(conn)  # type: ignore[misc]
    score_all(conn, model, version)

    rows = recommend_table(conn, min_recalls=(0.5,))

    assert [row.min_recall for row in rows] == [0.5]


def test_recommend_table_share_is_measured_at_the_printed_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_model(conn)  # type: ignore[misc]
    score_all(conn, model, version)
    raw_share = recommend_table(conn, min_recalls=(0.5,))[0].share
    assert raw_share is not None and raw_share < 1.0

    monkeypatch.setattr("infovore.triage.train.format_threshold", lambda threshold, scored: "0")
    row = recommend_table(conn, min_recalls=(0.5,))[0]

    assert row.formatted_threshold == "0"
    assert row.share == 1.0


# --- issue #95 PR A: rules threaded consistently, staleness recorded -------


def test_train_and_store_records_the_rules_version_it_trained_under(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)

    default_report = train_and_store(conn, FixedClock(NOW))
    custom_report = train_and_store(conn, FixedClock(NOW), rules=CUSTOM_RULES)

    default_params = json.loads(
        conn.execute(
            "SELECT params_json FROM triage_model WHERE version = ?", (default_report.version,)
        ).fetchone()["params_json"]
    )
    custom_params = json.loads(
        conn.execute(
            "SELECT params_json FROM triage_model WHERE version = ?", (custom_report.version,)
        ).fetchone()["params_json"]
    )
    assert default_params["rules_version"] == DEFAULT_RULES.version
    assert custom_params["rules_version"] == CUSTOM_RULES.version


def test_train_and_store_uses_the_given_rules_to_build_features(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1000, channel_id=1, content="bananarama forever")
    set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
    seed_labeled_exchanges(conn, 19, 20)

    report = train_and_store(conn, FixedClock(NOW), rules=CUSTOM_RULES)

    counts = conn.execute(
        "SELECT lore_count, noise_count FROM triage_tokens"
        " WHERE model_version = ? AND token = 'SIG_domain_terms'",
        (report.version,),
    ).fetchone()
    assert counts is not None
    assert counts["lore_count"] >= 1


def test_model_rules_version_is_none_without_a_trained_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert model_rules_version(conn) is None


def test_model_rules_version_reads_the_latest_models_recorded_version(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW))

    assert model_rules_version(conn) == DEFAULT_RULES.version

    train_and_store(conn, FixedClock(NOW), rules=CUSTOM_RULES)

    assert model_rules_version(conn) == CUSTOM_RULES.version


def test_model_rules_version_is_empty_string_for_a_model_trained_before_this_was_recorded(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    conn.execute(
        "INSERT INTO triage_model (trained_at, labels_used, holdout_size, params_json)"
        " VALUES (?, 40, 8, ?)",
        (NOW.isoformat(), json.dumps({"lore_documents": 20, "noise_documents": 20})),
    )

    assert model_rules_version(conn) == ""


def test_score_all_uses_the_given_rules_consistently_with_training(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1000, channel_id=1, content="bananarama forever")
    set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
    seed_labeled_exchanges(conn, 19, 20)
    report = train_and_store(conn, FixedClock(NOW), rules=CUSTOM_RULES)
    _, model = load_latest_model(conn)  # type: ignore[misc]

    score_all(conn, model, report.version, rules=CUSTOM_RULES)

    row = conn.execute("SELECT p_lore FROM exchanges WHERE id = ?", (exchange_id,)).fetchone()
    assert row["p_lore"] > 0.5


def test_score_stale_uses_the_given_rules(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1000, channel_id=1, content="bananarama forever")
    set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
    seed_labeled_exchanges(conn, 19, 20)
    report = train_and_store(conn, FixedClock(NOW), rules=CUSTOM_RULES)
    _, model = load_latest_model(conn)  # type: ignore[misc]

    scored = score_stale(conn, model, report.version, rules=CUSTOM_RULES)

    assert scored > 0
    row = conn.execute("SELECT p_lore FROM exchanges WHERE id = ?", (exchange_id,)).fetchone()
    assert row["p_lore"] > 0.5


def test_recommend_accepts_a_rules_override(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW), rules=CUSTOM_RULES)
    _, model = load_latest_model(conn)  # type: ignore[misc]
    score_all(conn, model, load_latest_model(conn)[0], rules=CUSTOM_RULES)  # type: ignore[index]

    result = recommend(conn, min_recall=0.5, rules=CUSTOM_RULES)

    assert result is not None


def test_recommend_table_accepts_a_rules_override(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled_exchanges(conn, 20, 20)
    train_and_store(conn, FixedClock(NOW), rules=CUSTOM_RULES)
    version, model = load_latest_model(conn)  # type: ignore[misc]
    score_all(conn, model, version, rules=CUSTOM_RULES)

    rows = recommend_table(conn, min_recalls=(0.5,), rules=CUSTOM_RULES)

    assert rows[0].min_recall == 0.5
