import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.message_labels import set_message_label
from infovore.rows import MessageLabel, MessageLabelSource
from infovore.sift.features import DEFAULT_FEATURE_SET, FEATURE_SET_VERSION, FeatureSet
from infovore.sift.train import (
    DEFAULT_MIN_HUMAN_LABELS_PER_CLASS,
    DISCARD_THRESHOLDS,
    Ensemble,
    FeatureSetAblation,
    FeatureSetMismatchError,
    InsufficientLabelsError,
    MessageTrainReport,
    build_message_examples,
    load_latest_message_model,
    score_all,
    score_stale,
    train_and_store,
)
from infovore.timing import FixedClock
from infovore.triage.bayes import Model, in_holdout, token_probability

NOW = datetime(2026, 1, 1, tzinfo=UTC)
NOW_TEXT = NOW.isoformat()

KEEP_CONTENT = "PROM 6.5.22 install guide with part 030-1234-001 and manual details"
TRASH_CONTENT = "lol gg no cap bro same energy again"

GROUP_SIZE = 40
HUMAN_GROUP_SIZE = DEFAULT_MIN_HUMAN_LABELS_PER_CLASS + 5


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def seed_channels(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO channels (id, guild_id, name, kind) VALUES (1, 1, 'general', 'text')")
    conn.execute("INSERT INTO channels (id, guild_id, name, kind) VALUES (2, 1, 'food', 'text')")


def seed_message_with_exchange(
    conn: sqlite3.Connection, message_id: int, channel_id: int, content: str
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 1, 1, 'alice', ?, ?, ?, '{}')",
        (message_id, channel_id, NOW_TEXT, content, NOW_TEXT),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, ?, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
        (message_id, channel_id, message_id, message_id, NOW_TEXT, NOW_TEXT, f"h{message_id}"),
    )
    conn.execute(
        "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
        (message_id, message_id),
    )


def seed_labeled(
    conn: sqlite3.Connection,
    count: int,
    label: MessageLabel,
    source: MessageLabelSource,
    channel_id: int,
    start_id: int,
    content: str,
) -> list[int]:
    ids = []
    for i in range(count):
        message_id = start_id + i
        seed_message_with_exchange(conn, message_id, channel_id, f"{content} token{i}")
        set_message_label(conn, message_id, label, source, None, NOW)
        ids.append(message_id)
    return ids


def seed_citation_only_corpus(conn: sqlite3.Connection) -> None:
    """Citation labels only, enough to train the citation model but nowhere
    near `DEFAULT_MIN_HUMAN_LABELS_PER_CLASS` human labels of either class
    (i.e. no human labels at all) -- the fallback path."""
    seed_channels(conn)
    seed_labeled(
        conn, GROUP_SIZE, MessageLabel.KEEP, MessageLabelSource.CITATION, 1, 2000, KEEP_CONTENT
    )
    seed_labeled(
        conn, GROUP_SIZE, MessageLabel.TRASH, MessageLabelSource.CITATION, 2, 3000, TRASH_CONTENT
    )


def seed_full_corpus(conn: sqlite3.Connection) -> None:
    """Enough of both citation and human labels (>= the default 30/class
    minimum) to exercise the full two-model ensemble, with no overlap between
    the human-labeled and citation-labeled message ids (issue #135's honest
    counts: a message never counts toward both models)."""
    seed_channels(conn)
    seed_labeled(
        conn, HUMAN_GROUP_SIZE, MessageLabel.KEEP, MessageLabelSource.HUMAN, 1, 1, KEEP_CONTENT
    )
    seed_labeled(
        conn, HUMAN_GROUP_SIZE, MessageLabel.TRASH, MessageLabelSource.HUMAN, 1, 1000, TRASH_CONTENT
    )
    seed_labeled(
        conn, GROUP_SIZE, MessageLabel.KEEP, MessageLabelSource.CITATION, 2, 2000, KEEP_CONTENT
    )
    seed_labeled(
        conn, GROUP_SIZE, MessageLabel.TRASH, MessageLabelSource.CITATION, 2, 3000, TRASH_CONTENT
    )


def non_holdout_ids(start: int, count: int) -> list[int]:
    ids: list[int] = []
    candidate = start
    while len(ids) < count:
        if not in_holdout(candidate):
            ids.append(candidate)
        candidate += 1
    return ids


# --- build_message_examples -------------------------------------------------


def test_build_message_examples_returns_tokens_label_and_source(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    examples = build_message_examples(conn)

    assert len(examples) == HUMAN_GROUP_SIZE * 2 + GROUP_SIZE * 2
    keep_example = next(e for e in examples if e.message_id == 1)
    assert keep_example.label is MessageLabel.KEEP
    assert keep_example.source is MessageLabelSource.HUMAN
    assert keep_example.channel_id == 1
    assert "prom" in keep_example.context_tokens
    assert "prom" in keep_example.plain_tokens
    assert "prom" in keep_example.structural_tokens

    citation_example = next(e for e in examples if e.message_id == 2000)
    assert citation_example.source is MessageLabelSource.CITATION
    assert citation_example.channel_id == 2


def test_build_message_examples_is_empty_with_no_labels(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_channels(conn)
    assert build_message_examples(conn) == []


# --- train_and_store: gating ---------------------------------------------------


def test_train_and_store_refuses_with_fewer_than_ten_citation_labels_per_class(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_channels(conn)
    seed_labeled(conn, 9, MessageLabel.KEEP, MessageLabelSource.CITATION, 1, 1, KEEP_CONTENT)
    seed_labeled(conn, 20, MessageLabel.TRASH, MessageLabelSource.CITATION, 1, 100, TRASH_CONTENT)

    with pytest.raises(InsufficientLabelsError, match="10"):
        train_and_store(conn, FixedClock(NOW))


def test_train_and_store_citation_gate_ignores_messages_that_also_have_a_human_label(
    tmp_path: Path,
) -> None:
    """A message with both a citation label and a human label counts as
    human-only for training (human beats citation at read time) -- it must
    not count toward the citation model's minimum either."""
    conn = db(tmp_path)
    seed_channels(conn)
    ids = seed_labeled(conn, 20, MessageLabel.KEEP, MessageLabelSource.CITATION, 1, 1, KEEP_CONTENT)
    seed_labeled(conn, 20, MessageLabel.TRASH, MessageLabelSource.CITATION, 1, 100, TRASH_CONTENT)
    # Relabel every "citation keep" message with a human label too, leaving
    # zero citation-only keep examples.
    for message_id in ids:
        set_message_label(conn, message_id, MessageLabel.KEEP, MessageLabelSource.HUMAN, None, NOW)

    with pytest.raises(InsufficientLabelsError, match="10"):
        train_and_store(conn, FixedClock(NOW))


# --- train_and_store: fallback (issue #135 design point 4) ---------------------


def test_train_and_store_falls_back_to_citation_only_below_the_human_minimum(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_citation_only_corpus(conn)

    report = train_and_store(conn, FixedClock(NOW))

    assert isinstance(report, MessageTrainReport)
    assert report.fallback is True
    assert report.human_labels_used == 0
    assert report.human_auc is None
    assert report.combined_auc is None
    assert all(row.keep_lost is None and row.trash_caught is None for row in report.discard_pile)
    assert report.human_top_tokens == ()
    assert len(report.citation_top_tokens) > 0

    loaded = load_latest_message_model(conn)
    assert loaded is not None
    assert loaded.fallback is True
    assert loaded.human_model is None
    assert loaded.human_version is None
    assert loaded.combiner is None


def test_train_and_store_fallback_still_reports_citation_auc_against_human_labels(
    tmp_path: Path,
) -> None:
    """Below-minimum human labels still fall back, but any human labels that
    do exist are still an honest (out-of-sample) check on the citation
    model, since the citation model never trains on human-labeled messages."""
    conn = db(tmp_path)
    seed_citation_only_corpus(conn)
    # A handful of human labels -- fewer than DEFAULT_MIN_HUMAN_LABELS_PER_CLASS.
    seed_labeled(conn, 5, MessageLabel.KEEP, MessageLabelSource.HUMAN, 1, 9000, KEEP_CONTENT)
    seed_labeled(conn, 5, MessageLabel.TRASH, MessageLabelSource.HUMAN, 1, 9100, TRASH_CONTENT)

    report = train_and_store(conn, FixedClock(NOW))

    assert report.fallback is True
    assert report.human_labels_used == 10
    assert report.citation_auc is not None


def test_min_human_labels_per_class_is_configurable(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)  # HUMAN_GROUP_SIZE (35) of each human label

    report = train_and_store(conn, FixedClock(NOW), min_human_labels_per_class=1000)

    assert report.fallback is True
    assert report.min_human_labels_per_class == 1000


def test_min_human_labels_per_class_zero_fits_a_combiner_with_no_human_labels_at_all(
    tmp_path: Path,
) -> None:
    """An edge case of the configurable minimum: `--min-human-labels 0` means
    even zero human labels of a class clears the bar, so the non-fallback
    path runs with an empty human example set -- every human-only figure
    (AUC, discard pile shares) comes back `None` rather than a division by
    zero or a crash."""
    conn = db(tmp_path)
    seed_citation_only_corpus(conn)

    report = train_and_store(conn, FixedClock(NOW), min_human_labels_per_class=0)

    assert report.fallback is False
    assert report.human_labels_used == 0
    assert report.human_auc is None
    assert report.combined_auc is None
    assert all(row.keep_lost is None and row.trash_caught is None for row in report.discard_pile)

    loaded = load_latest_message_model(conn)
    assert loaded is not None
    assert loaded.fallback is False
    assert loaded.human_model is not None
    assert loaded.combiner is not None


# --- train_and_store: full ensemble ---------------------------------------------


def test_train_and_store_builds_a_report_and_persists_both_models_and_the_combiner(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    report = train_and_store(conn, FixedClock(NOW))

    assert isinstance(report, MessageTrainReport)
    assert report.fallback is False
    assert report.citation_labels_used == GROUP_SIZE * 2
    assert report.human_labels_used == HUMAN_GROUP_SIZE * 2
    assert report.citation_auc is not None
    assert report.human_auc is not None
    assert report.combined_auc is not None
    assert 0.0 <= report.citation_auc <= 1.0
    assert 0.0 <= report.human_auc <= 1.0
    assert 0.0 <= report.combined_auc <= 1.0
    assert len(report.discard_pile) == len(DISCARD_THRESHOLDS)
    for row in report.discard_pile:
        assert row.keep_lost is not None
        assert row.trash_caught is not None
        assert 0.0 <= row.keep_lost <= 1.0
        assert 0.0 <= row.trash_caught <= 1.0
    assert 0 < len(report.citation_top_tokens) <= 15
    assert 0 < len(report.human_top_tokens) <= 15
    assert set(report.channel_stats) <= {"general", "food"}

    combiner_row = conn.execute(
        "SELECT version, citation_model_version, human_model_version, fallback, params_json"
        " FROM message_combiner"
    ).fetchone()
    assert combiner_row["version"] == report.version
    assert combiner_row["fallback"] == 0
    assert combiner_row["human_model_version"] is not None

    citation_model_row = conn.execute(
        "SELECT kind FROM message_model WHERE version = ?",
        (combiner_row["citation_model_version"],),
    ).fetchone()
    assert citation_model_row["kind"] == "citation"
    human_model_row = conn.execute(
        "SELECT kind FROM message_model WHERE version = ?", (combiner_row["human_model_version"],)
    ).fetchone()
    assert human_model_row["kind"] == "human"

    combiner_params = json.loads(combiner_row["params_json"])
    assert "intercept" in combiner_params
    assert "weights" in combiner_params

    token_rows = conn.execute(
        "SELECT COUNT(*) AS n FROM message_tokens WHERE model_version = ?",
        (combiner_row["citation_model_version"],),
    ).fetchone()
    assert token_rows["n"] > 0


def test_train_and_store_citation_holdout_secondary_section(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    report = train_and_store(conn, FixedClock(NOW))

    assert [m.threshold for m in report.citation_holdout_metrics] == [
        round(i / 10, 1) for i in range(1, 10)
    ]
    total = (
        report.citation_holdout_confusion.tp
        + report.citation_holdout_confusion.fp
        + report.citation_holdout_confusion.fn
        + report.citation_holdout_confusion.tn
    )
    assert total >= 0


def test_train_and_store_confusion_uses_the_given_threshold(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    lenient = train_and_store(conn, FixedClock(NOW), confusion_threshold=0.1)
    strict = train_and_store(conn, FixedClock(NOW), confusion_threshold=0.9)
    off_grid = train_and_store(conn, FixedClock(NOW), confusion_threshold=0.42)

    assert lenient.citation_holdout_confusion.threshold == 0.1
    assert strict.citation_holdout_confusion.threshold == 0.9
    assert off_grid.citation_holdout_confusion.threshold == 0.42


def test_train_and_store_is_deterministic(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    first = train_and_store(conn, FixedClock(NOW))
    second = train_and_store(conn, FixedClock(NOW))

    assert first.combined_auc == second.combined_auc
    assert first.human_auc == second.human_auc
    assert first.citation_auc == second.citation_auc
    assert first.discard_pile == second.discard_pile


# --- load_latest_message_model ------------------------------------------------


def test_load_latest_message_model_reconstructs_both_models(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)
    report = train_and_store(conn, FixedClock(NOW))

    loaded = load_latest_message_model(conn)
    assert loaded is not None
    assert isinstance(loaded, Ensemble)
    assert loaded.combiner_version == report.version
    assert isinstance(loaded.citation_model, Model)
    assert loaded.human_model is not None
    assert isinstance(loaded.human_model, Model)
    assert token_probability(loaded.human_model, "lol") > 0.9
    assert token_probability(loaded.citation_model, "lol") > 0.9
    assert loaded.combiner is not None
    assert loaded.fallback is False


def test_load_latest_message_model_returns_none_when_untrained(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert load_latest_message_model(conn) is None


def test_load_latest_message_model_returns_the_most_recently_trained_version(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)
    train_and_store(conn, FixedClock(NOW))
    second = train_and_store(conn, FixedClock(NOW))

    loaded = load_latest_message_model(conn)
    assert loaded is not None
    assert loaded.combiner_version == second.version


# --- issue #135's regression fixture -------------------------------------------


def test_a_single_human_trashed_message_does_not_dominate_the_citation_model(
    tmp_path: Path,
) -> None:
    """The bug that #135 fixes: with plain duplication (the old
    `--human-weight`), a single human-trashed message's distinctive words
    could dominate the model and send an unrelated, keep-worthy message
    containing one of those words to a high p_trash. With honest per-model
    counts and a fitted combiner, one human example is just one example."""
    conn = db(tmp_path)
    seed_channels(conn)
    conn.execute(
        "INSERT INTO channels (id, guild_id, name, kind) VALUES (3, 1, 'off-topic', 'text')"
    )

    # Plenty of ordinary citation-labeled keep/trash so the citation model is
    # well-formed and the human minimum-per-class gate is met independently.
    # Channels 2/3 (never used by a human-labeled message below) keep the
    # citation model's CHAN_<id> tokens from confounding with channel 1,
    # where every human example lives -- otherwise CHAN_1 alone would look
    # like strong citation evidence for every human example regardless of
    # its true label, just because human labeling happened to be scoped to
    # one channel, which has nothing to do with this fixture's actual bug.
    seed_labeled(
        conn, GROUP_SIZE, MessageLabel.KEEP, MessageLabelSource.CITATION, 2, 2000, KEEP_CONTENT
    )
    seed_labeled(
        conn, GROUP_SIZE, MessageLabel.TRASH, MessageLabelSource.CITATION, 3, 3000, TRASH_CONTENT
    )
    seed_labeled(
        conn,
        HUMAN_GROUP_SIZE,
        MessageLabel.KEEP,
        MessageLabelSource.HUMAN,
        1,
        5000,
        "ordinary discord chatter about nothing in particular",
    )
    seed_labeled(
        conn,
        HUMAN_GROUP_SIZE - 1,
        MessageLabel.TRASH,
        MessageLabelSource.HUMAN,
        1,
        6000,
        "lol gg no cap bro same energy again",
    )
    # The one human-trashed message with distinctive words (issue #135's SGI
    # "Judge" graphics example): one document, honestly counted once.
    distinctive_trash_id = 6000 + HUMAN_GROUP_SIZE - 1
    seed_message_with_exchange(conn, distinctive_trash_id, 1, "diesel judge corolla mcdonalds")
    set_message_label(
        conn, distinctive_trash_id, MessageLabel.TRASH, MessageLabelSource.HUMAN, None, NOW
    )

    train_and_store(conn, FixedClock(NOW))
    loaded = load_latest_message_model(conn)
    assert loaded is not None
    assert loaded.fallback is False

    from infovore.rows import MessageRow
    from infovore.sift.features import message_features
    from infovore.sift.train import p_trash_for_tokens

    unrelated_keep_message = MessageRow(
        id=999999,
        channel_id=1,
        guild_id=1,
        author_id=1,
        author_name_at_time="alice",
        author_is_bot=False,
        created_at=NOW,
        content=(
            "PROM 6.5.22 install guide with part 030-1234-001 and manual details, also"
            " ran into a diesel generator part number while reading it"
        ),
        edited_at=None,
        reply_to_id=None,
        thread_id=None,
        deleted_at=None,
        ingested_at=NOW,
        raw_json="{}",
    )
    tokens = message_features(unrelated_keep_message, unrelated_keep_message.channel_id)
    assert "diesel" in tokens

    p_trash = p_trash_for_tokens(loaded, tokens)
    assert p_trash < 0.5


# --- scoring -----------------------------------------------------------------


def test_score_all_writes_p_trash_only_for_messages_in_exchanges(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)
    train_and_store(conn, FixedClock(NOW))
    ensemble = load_latest_message_model(conn)
    assert ensemble is not None

    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (99999, 1, 1, 1, 'alice', ?, 'not in any exchange', ?, '{}')",
        (NOW_TEXT, NOW_TEXT),
    )

    scored = score_all(conn, ensemble)

    assert scored == HUMAN_GROUP_SIZE * 2 + GROUP_SIZE * 2
    orphan = conn.execute("SELECT p_trash FROM messages WHERE id = 99999").fetchone()
    assert orphan["p_trash"] is None

    rows = conn.execute(
        "SELECT m.p_trash AS p_trash, m.p_trash_model AS p_trash_model FROM messages m"
        " JOIN exchange_messages em ON em.message_id = m.id"
    ).fetchall()
    assert len(rows) == HUMAN_GROUP_SIZE * 2 + GROUP_SIZE * 2
    for row in rows:
        assert row["p_trash"] is not None
        assert row["p_trash_model"] == ensemble.combiner_version

    keep_row = conn.execute("SELECT p_trash FROM messages WHERE id = 1").fetchone()
    trash_row = conn.execute("SELECT p_trash FROM messages WHERE id = 1000").fetchone()
    assert keep_row["p_trash"] < trash_row["p_trash"]


def test_score_all_falls_back_to_citation_only_scoring(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_citation_only_corpus(conn)
    train_and_store(conn, FixedClock(NOW))
    ensemble = load_latest_message_model(conn)
    assert ensemble is not None
    assert ensemble.fallback is True

    scored = score_all(conn, ensemble)
    assert scored == GROUP_SIZE * 2
    rows = conn.execute("SELECT p_trash FROM messages WHERE p_trash IS NOT NULL").fetchall()
    assert all(row["p_trash"] is not None for row in rows)


def test_score_stale_only_rescores_messages_behind_the_latest_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)
    train_and_store(conn, FixedClock(NOW))
    ensemble = load_latest_message_model(conn)
    assert ensemble is not None

    first_pass = score_stale(conn, ensemble)
    assert first_pass == HUMAN_GROUP_SIZE * 2 + GROUP_SIZE * 2

    second_pass = score_stale(conn, ensemble)
    assert second_pass == 0

    conn.execute("UPDATE messages SET p_trash_model = NULL WHERE id = 1")
    third_pass = score_stale(conn, ensemble)
    assert third_pass == 1


def test_score_all_workers_two_matches_workers_one(tmp_path: Path) -> None:
    conn_serial = open_database(tmp_path / "serial.db")
    migrate(conn_serial)
    seed_full_corpus(conn_serial)
    train_and_store(conn_serial, FixedClock(NOW))
    ensemble_serial = load_latest_message_model(conn_serial)
    assert ensemble_serial is not None

    conn_parallel = open_database(tmp_path / "parallel.db")
    migrate(conn_parallel)
    seed_full_corpus(conn_parallel)
    train_and_store(conn_parallel, FixedClock(NOW))
    ensemble_parallel = load_latest_message_model(conn_parallel)
    assert ensemble_parallel is not None

    scored_serial = score_all(conn_serial, ensemble_serial, workers=1)
    scored_parallel = score_all(conn_parallel, ensemble_parallel, workers=2)

    assert scored_serial == scored_parallel == HUMAN_GROUP_SIZE * 2 + GROUP_SIZE * 2
    serial_rows = conn_serial.execute(
        "SELECT id, p_trash, p_trash_model FROM messages WHERE p_trash_model IS NOT NULL"
        " ORDER BY id"
    ).fetchall()
    parallel_rows = conn_parallel.execute(
        "SELECT id, p_trash, p_trash_model FROM messages WHERE p_trash_model IS NOT NULL"
        " ORDER BY id"
    ).fetchall()
    assert [(r["id"], r["p_trash"]) for r in serial_rows] == [
        (r["id"], r["p_trash"]) for r in parallel_rows
    ]


# --- issue #141: conversation-context features --------------------------------


def seed_message(
    conn: sqlite3.Connection, message_id: int, channel_id: int, author_id: int, content: str
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 1, ?, 'someone', ?, ?, ?, '{}')",
        (message_id, channel_id, author_id, NOW_TEXT, content, NOW_TEXT),
    )


def seed_exchange(
    conn: sqlite3.Connection, exchange_id: int, channel_id: int, message_ids: list[int]
) -> None:
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, 'quiet_gap', ?)",
        (
            exchange_id,
            channel_id,
            message_ids[0],
            message_ids[-1],
            NOW_TEXT,
            NOW_TEXT,
            len(message_ids),
            f"h{exchange_id}",
        ),
    )
    for position, message_id in enumerate(message_ids):
        conn.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, ?)",
            (exchange_id, message_id, position),
        )


def test_build_message_examples_tokens_include_context_but_plain_tokens_do_not(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_channels(conn)
    seed_message(conn, 400, 1, 1, "run pip install widget")
    seed_message(conn, 500, 1, 2, "that works")
    seed_exchange(conn, 600, 1, [400, 500])
    set_message_label(conn, 500, MessageLabel.KEEP, MessageLabelSource.CITATION, None, NOW)

    examples = build_message_examples(conn)
    focus = next(e for e in examples if e.message_id == 500)

    assert "PREV_run" in focus.context_tokens
    assert "PREV_run" not in focus.plain_tokens
    assert "PREV_run" not in focus.structural_tokens
    assert "that" in focus.context_tokens
    assert "that" in focus.plain_tokens
    assert "that" in focus.structural_tokens


def test_build_message_examples_structural_tokens_keep_shape_but_not_neighbour_words(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_channels(conn)
    seed_message(conn, 400, 1, 1, "see the manual at /etc/config/app.conf")
    seed_message(conn, 500, 1, 2, "that works")
    seed_exchange(conn, 600, 1, [400, 500])
    set_message_label(conn, 500, MessageLabel.KEEP, MessageLabelSource.CITATION, None, NOW)

    examples = build_message_examples(conn)
    focus = next(e for e in examples if e.message_id == 500)

    assert "PREV_FACT_path" in focus.structural_tokens
    assert "POS_last" in focus.structural_tokens
    assert "EXSIZE_1-2" in focus.structural_tokens
    assert not any(
        token.startswith("PREV_") and "_FACT_" not in token for token in focus.structural_tokens
    )
    assert "PREV_FACT_path" in focus.context_tokens
    assert "PREV_manual" in focus.context_tokens
    assert "PREV_manual" not in focus.structural_tokens


def test_build_message_examples_context_excludes_opted_out_neighbour(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_channels(conn)
    seed_message(conn, 400, 1, 42, "secret private words")
    seed_message(conn, 500, 1, 2, "that works")
    seed_exchange(conn, 600, 1, [400, 500])
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (42, ?)", (NOW_TEXT,))
    set_message_label(conn, 500, MessageLabel.KEEP, MessageLabelSource.CITATION, None, NOW)

    examples = build_message_examples(conn)
    focus = next(e for e in examples if e.message_id == 500)

    assert not any(token.startswith("PREV_") for token in focus.context_tokens)


def test_train_and_store_persists_the_current_feature_set_version(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    train_and_store(conn, FixedClock(NOW))

    row = conn.execute(
        "SELECT feature_set_version, feature_set_name FROM message_combiner"
    ).fetchone()
    assert row["feature_set_version"] == FEATURE_SET_VERSION
    assert row["feature_set_name"] == FeatureSet.PLAIN.value

    loaded = load_latest_message_model(conn)
    assert loaded is not None
    assert loaded.feature_set_version == FEATURE_SET_VERSION
    assert loaded.feature_set is FeatureSet.PLAIN


def test_train_and_store_default_feature_set_is_plain(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    report = train_and_store(conn, FixedClock(NOW))

    assert report.feature_set is FeatureSet.PLAIN
    assert DEFAULT_FEATURE_SET is FeatureSet.PLAIN


def test_train_and_store_persists_an_explicitly_chosen_feature_set(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    report = train_and_store(conn, FixedClock(NOW), feature_set=FeatureSet.STRUCTURAL)

    assert report.feature_set is FeatureSet.STRUCTURAL
    row = conn.execute("SELECT feature_set_name FROM message_combiner").fetchone()
    assert row["feature_set_name"] == "structural"
    loaded = load_latest_message_model(conn)
    assert loaded is not None
    assert loaded.feature_set is FeatureSet.STRUCTURAL


def test_train_and_store_report_includes_ablations_for_all_three_feature_sets(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    report = train_and_store(conn, FixedClock(NOW))

    assert len(report.ablations) == 3
    assert {a.feature_set for a in report.ablations} == {
        FeatureSet.PLAIN,
        FeatureSet.STRUCTURAL,
        FeatureSet.CONTEXT,
    }
    for ablation in report.ablations:
        assert isinstance(ablation, FeatureSetAblation)
        assert ablation.combined_auc is not None
        assert 0.0 <= ablation.combined_auc <= 1.0
        assert len(ablation.discard_pile) == len(DISCARD_THRESHOLDS)
        for row in ablation.discard_pile:
            assert row.keep_lost is not None
            assert row.trash_caught is not None

    # The persisted (default: plain) row's own figures match the chosen
    # ablation entry exactly -- same fit, not a re-derived approximation.
    plain_row = next(a for a in report.ablations if a.feature_set is FeatureSet.PLAIN)
    assert report.combined_auc == plain_row.combined_auc
    assert report.citation_auc == plain_row.citation_auc
    assert report.human_auc == plain_row.human_auc
    assert report.discard_pile == plain_row.discard_pile


def test_train_and_store_ablations_fall_back_the_same_way_as_the_primary_fit(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_citation_only_corpus(conn)

    report = train_and_store(conn, FixedClock(NOW))

    assert report.fallback is True
    assert len(report.ablations) == 3
    for ablation in report.ablations:
        assert ablation.human_auc is None
        assert ablation.combined_auc is None
        assert all(
            row.keep_lost is None and row.trash_caught is None for row in ablation.discard_pile
        )


def seed_decisive_context_pair(
    conn: sqlite3.Connection,
    exchange_id: int,
    prev_id: int,
    focus_id: int,
    prev_content: str,
    label: MessageLabel,
) -> None:
    seed_message(conn, prev_id, 1, 1, prev_content)
    seed_message(conn, focus_id, 1, 2, "that works")
    seed_exchange(conn, exchange_id, 1, [prev_id, focus_id])
    set_message_label(conn, focus_id, label, MessageLabelSource.HUMAN, None, NOW)


def seed_decisive_context_corpus(conn: sqlite3.Connection) -> None:
    """The issue's own motivating fixture: "that works" is identical junk
    or a meaningful confirmation depending on what came before it. Every
    `keep` example is literally the message "that works" after a how-to;
    every `trash` example is the identical message after unrelated chatter
    -- so a model without conversation-context features cannot possibly
    separate them (their own tokens are identical), while a model with
    context features can, via the previous message's words."""
    seed_channels(conn)
    seed_labeled(
        conn, GROUP_SIZE, MessageLabel.KEEP, MessageLabelSource.CITATION, 2, 2000, KEEP_CONTENT
    )
    seed_labeled(
        conn, GROUP_SIZE, MessageLabel.TRASH, MessageLabelSource.CITATION, 2, 3000, TRASH_CONTENT
    )
    for i in range(HUMAN_GROUP_SIZE):
        seed_decisive_context_pair(
            conn,
            exchange_id=40000 + i,
            prev_id=41000 + i,
            focus_id=42000 + i,
            prev_content=f"run pip install widget{i} then restart the service",
            label=MessageLabel.KEEP,
        )
    for i in range(HUMAN_GROUP_SIZE):
        seed_decisive_context_pair(
            conn,
            exchange_id=50000 + i,
            prev_id=51000 + i,
            focus_id=52000 + i,
            prev_content=f"lol remember that time we did the thing haha {i}",
            label=MessageLabel.TRASH,
        )


def test_ablation_compares_all_three_feature_sets_on_the_decisive_fixture(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_decisive_context_corpus(conn)

    report = train_and_store(conn, FixedClock(NOW))  # default persisted: plain

    assert report.fallback is False
    by_set = {a.feature_set: a for a in report.ablations}
    plain_auc = by_set[FeatureSet.PLAIN].combined_auc
    structural_auc = by_set[FeatureSet.STRUCTURAL].combined_auc
    context_auc = by_set[FeatureSet.CONTEXT].combined_auc
    assert plain_auc is not None
    assert structural_auc is not None
    assert context_auc is not None
    # Without any neighbour word tokens, every "that works" message has
    # identical own-message tokens regardless of label (plain), and no
    # differing fact-shape/position signal either (structural) -- both are
    # close to chance.
    assert plain_auc < 0.65
    assert structural_auc < 0.7
    # With full context, the previous message's own words separate them
    # cleanly.
    assert context_auc > 0.9
    assert context_auc - plain_auc > 0.3
    # The persisted model (default: plain) matches the plain ablation row.
    assert report.feature_set is FeatureSet.PLAIN
    assert report.combined_auc == plain_auc


def test_score_all_raises_when_the_stored_feature_set_version_is_stale(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_citation_only_corpus(conn)
    train_and_store(conn, FixedClock(NOW))
    ensemble = load_latest_message_model(conn)
    assert ensemble is not None
    stale = replace(ensemble, feature_set_version=ensemble.feature_set_version - 1)

    with pytest.raises(FeatureSetMismatchError, match="infovore sift train"):
        score_all(conn, stale)


def test_score_stale_raises_when_the_stored_feature_set_version_is_stale(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_citation_only_corpus(conn)
    train_and_store(conn, FixedClock(NOW))
    ensemble = load_latest_message_model(conn)
    assert ensemble is not None
    stale = replace(ensemble, feature_set_version=ensemble.feature_set_version - 1)

    with pytest.raises(FeatureSetMismatchError):
        score_stale(conn, stale)


# --- issue #144: scoring builds exactly the persisted feature set -------------


def _seed_context_sensitive_pair(conn: sqlite3.Connection) -> None:
    """Two unlabeled, never-trained-on messages -- own tokens ("that
    works") identical in both -- whose previous message differs (how-to vs
    chatter), each in its own exchange, so `p_trash` differs between them
    only if scoring actually consults conversation context."""
    seed_message(conn, 90000, 1, 1, "run pip install widget then restart")
    seed_message(conn, 90001, 1, 2, "that works")
    seed_exchange(conn, 90100, 1, [90000, 90001])
    seed_message(conn, 90002, 1, 1, "lol remember that time we did the thing")
    seed_message(conn, 90003, 1, 2, "that works")
    seed_exchange(conn, 90101, 1, [90002, 90003])


def test_score_all_builds_context_tokens_when_the_persisted_feature_set_is_context(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_decisive_context_corpus(conn)
    train_and_store(conn, FixedClock(NOW), feature_set=FeatureSet.CONTEXT)
    ensemble = load_latest_message_model(conn)
    assert ensemble is not None
    assert ensemble.feature_set is FeatureSet.CONTEXT

    _seed_context_sensitive_pair(conn)
    score_all(conn, ensemble)

    keep_like = conn.execute("SELECT p_trash FROM messages WHERE id = 90001").fetchone()["p_trash"]
    trash_like = conn.execute("SELECT p_trash FROM messages WHERE id = 90003").fetchone()["p_trash"]
    assert keep_like != trash_like


def test_score_all_builds_only_plain_tokens_when_the_persisted_feature_set_is_plain(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_decisive_context_corpus(conn)
    train_and_store(conn, FixedClock(NOW))  # default: plain
    ensemble = load_latest_message_model(conn)
    assert ensemble is not None
    assert ensemble.feature_set is FeatureSet.PLAIN

    _seed_context_sensitive_pair(conn)
    score_all(conn, ensemble)

    keep_like = conn.execute("SELECT p_trash FROM messages WHERE id = 90001").fetchone()["p_trash"]
    trash_like = conn.execute("SELECT p_trash FROM messages WHERE id = 90003").fetchone()["p_trash"]
    assert keep_like == trash_like


def test_score_all_builds_only_structural_tokens_when_the_persisted_feature_set_is_structural(
    tmp_path: Path,
) -> None:
    """Structural sees the surrounding conversation's *shape* but never a
    neighbour's own words, so two messages whose neighbours differ only in
    wording (no differing fact-shape/position signal) still score
    identically."""
    conn = db(tmp_path)
    seed_decisive_context_corpus(conn)
    train_and_store(conn, FixedClock(NOW), feature_set=FeatureSet.STRUCTURAL)
    ensemble = load_latest_message_model(conn)
    assert ensemble is not None
    assert ensemble.feature_set is FeatureSet.STRUCTURAL

    _seed_context_sensitive_pair(conn)
    score_all(conn, ensemble)

    keep_like = conn.execute("SELECT p_trash FROM messages WHERE id = 90001").fetchone()["p_trash"]
    trash_like = conn.execute("SELECT p_trash FROM messages WHERE id = 90003").fetchone()["p_trash"]
    assert keep_like == trash_like


# --- issue #144: backward compatibility with a pre-#144 stored model ----------


def _insert_legacy_message_combiner_row(conn: sqlite3.Connection, feature_set_version: int) -> None:
    """Simulates a `message_combiner` row written before issue #144 --
    before the `feature_set_name` column existed, only `feature_set_version`
    was ever recorded (issue #141's migration 0013)."""
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json, kind)"
        " VALUES (?, 1, 0, '{\"trash_documents\": 1, \"keep_documents\": 1}', 'citation')",
        (NOW_TEXT,),
    )
    citation_version = conn.execute("SELECT version FROM message_model").fetchone()["version"]
    conn.execute(
        "INSERT INTO message_combiner (trained_at, citation_model_version, human_model_version,"
        " fallback, params_json, feature_set_version)"
        " VALUES (?, ?, NULL, 1, '{}', ?)",
        (NOW_TEXT, citation_version, feature_set_version),
    )


def test_load_latest_message_model_resolves_a_legacy_version_2_row_to_context(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    _insert_legacy_message_combiner_row(conn, feature_set_version=2)

    loaded = load_latest_message_model(conn)

    assert loaded is not None
    assert loaded.feature_set_version == 2
    assert loaded.feature_set is FeatureSet.CONTEXT


def test_load_latest_message_model_resolves_a_legacy_version_1_row_to_plain(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    _insert_legacy_message_combiner_row(conn, feature_set_version=1)

    loaded = load_latest_message_model(conn)

    assert loaded is not None
    assert loaded.feature_set_version == 1
    assert loaded.feature_set is FeatureSet.PLAIN
