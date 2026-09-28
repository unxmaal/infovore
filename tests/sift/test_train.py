import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.message_labels import set_message_label
from infovore.rows import MessageLabel, MessageLabelSource
from infovore.sift.train import (
    DEFAULT_HUMAN_WEIGHT,
    DISCARD_THRESHOLDS,
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


def seed_full_corpus(conn: sqlite3.Connection) -> None:
    seed_channels(conn)
    seed_labeled(conn, GROUP_SIZE, MessageLabel.KEEP, MessageLabelSource.HUMAN, 1, 1, KEEP_CONTENT)
    seed_labeled(
        conn, GROUP_SIZE, MessageLabel.TRASH, MessageLabelSource.HUMAN, 1, 1000, TRASH_CONTENT
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

    assert len(examples) == GROUP_SIZE * 4
    keep_example = next(e for e in examples if e.message_id == 1)
    assert keep_example.label is MessageLabel.KEEP
    assert keep_example.source is MessageLabelSource.HUMAN
    assert keep_example.channel_id == 1
    assert "prom" in keep_example.tokens

    citation_example = next(e for e in examples if e.message_id == 2000)
    assert citation_example.source is MessageLabelSource.CITATION
    assert citation_example.channel_id == 2


def test_build_message_examples_is_empty_with_no_labels(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_channels(conn)
    assert build_message_examples(conn) == []


# --- train_and_store ---------------------------------------------------------


def test_train_and_store_refuses_with_fewer_than_ten_labels_per_class(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_channels(conn)
    seed_labeled(conn, 9, MessageLabel.KEEP, MessageLabelSource.HUMAN, 1, 1, KEEP_CONTENT)
    seed_labeled(conn, 20, MessageLabel.TRASH, MessageLabelSource.HUMAN, 1, 100, TRASH_CONTENT)

    with pytest.raises(InsufficientLabelsError, match="10"):
        train_and_store(conn, FixedClock(NOW))


def test_train_and_store_builds_a_report_and_persists_the_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    report = train_and_store(conn, FixedClock(NOW))

    assert isinstance(report, MessageTrainReport)
    assert report.labels_used == GROUP_SIZE * 4
    assert report.holdout_size > 0
    assert report.human_weight == DEFAULT_HUMAN_WEIGHT
    assert [m.threshold for m in report.overall_metrics] == [round(i / 10, 1) for i in range(1, 10)]
    assert (
        report.confusion.tp + report.confusion.fp + report.confusion.fn + report.confusion.tn
        == report.holdout_size
    )
    assert 0 < len(report.top_tokens) <= 15
    assert len(report.discard_pile) == len(DISCARD_THRESHOLDS)
    for row in report.discard_pile:
        if row.human_keep_discarded is not None:
            assert 0.0 <= row.human_keep_discarded <= 1.0
        if row.citation_keep_discarded is not None:
            assert 0.0 <= row.citation_keep_discarded <= 1.0
    assert set(report.channel_stats) <= {1, 2}

    row = conn.execute(
        "SELECT version, labels_used, holdout_size, params_json FROM message_model"
    ).fetchone()
    assert row["version"] == report.version
    assert row["labels_used"] == GROUP_SIZE * 4
    params = json.loads(row["params_json"])
    assert params["trash_documents"] > 0
    assert params["keep_documents"] > 0
    assert params["human_weight"] == DEFAULT_HUMAN_WEIGHT

    token_rows = conn.execute(
        "SELECT COUNT(*) AS n FROM message_tokens WHERE model_version = ?", (report.version,)
    ).fetchone()
    assert token_rows["n"] > 0


def test_train_and_store_confusion_uses_the_given_threshold(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    lenient = train_and_store(conn, FixedClock(NOW), confusion_threshold=0.1)
    strict = train_and_store(conn, FixedClock(NOW), confusion_threshold=0.9)
    off_grid = train_and_store(conn, FixedClock(NOW), confusion_threshold=0.42)

    assert lenient.confusion.threshold == 0.1
    assert strict.confusion.threshold == 0.9
    assert off_grid.confusion.threshold == 0.42


def test_train_and_store_reports_human_and_citation_metrics_separately(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)

    report = train_and_store(conn, FixedClock(NOW))

    assert [m.threshold for m in report.human_metrics] == [round(i / 10, 1) for i in range(1, 10)]
    assert [m.threshold for m in report.citation_metrics] == [
        round(i / 10, 1) for i in range(1, 10)
    ]
    human_total = report.human_metrics[0].tp + report.human_metrics[0].fp
    human_total += report.human_metrics[0].fn + report.human_metrics[0].tn
    citation_total = report.citation_metrics[0].tp + report.citation_metrics[0].fp
    citation_total += report.citation_metrics[0].fn + report.citation_metrics[0].tn
    assert human_total + citation_total == report.holdout_size


def test_load_latest_message_model_reconstructs_token_probabilities(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)
    report = train_and_store(conn, FixedClock(NOW))

    loaded = load_latest_message_model(conn)
    assert loaded is not None
    version, model = loaded
    assert version == report.version
    assert isinstance(model, Model)
    assert token_probability(model, "lol") > 0.9
    assert token_probability(model, "prom") < 0.1


def test_load_latest_message_model_returns_none_when_untrained(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert load_latest_message_model(conn) is None


def test_human_weight_lets_one_human_example_outweigh_many_citation_examples(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    seed_channels(conn)

    # filler labels so the >=10-per-class floor is met regardless of holdout split.
    seed_labeled(conn, 12, MessageLabel.KEEP, MessageLabelSource.CITATION, 1, 5000, "keep filler")
    seed_labeled(conn, 12, MessageLabel.TRASH, MessageLabelSource.CITATION, 1, 6000, "trash filler")

    citation_trash_ids = non_holdout_ids(7000, 20)
    for message_id in citation_trash_ids:
        seed_message_with_exchange(conn, message_id, 1, "distinctiveword trash chatter")
        set_message_label(
            conn, message_id, MessageLabel.TRASH, MessageLabelSource.CITATION, None, NOW
        )

    human_keep_id = non_holdout_ids(8000, 1)[0]
    seed_message_with_exchange(conn, human_keep_id, 1, "distinctiveword keep content")
    set_message_label(conn, human_keep_id, MessageLabel.KEEP, MessageLabelSource.HUMAN, None, NOW)

    train_and_store(conn, FixedClock(NOW), human_weight=1)
    _, model_light = load_latest_message_model(conn)  # type: ignore[misc]
    p_light = token_probability(model_light, "distinctiveword")

    train_and_store(conn, FixedClock(NOW), human_weight=25)
    _, model_heavy = load_latest_message_model(conn)  # type: ignore[misc]
    p_heavy = token_probability(model_heavy, "distinctiveword")

    assert p_light > 0.5
    assert p_heavy < 0.5
    assert p_heavy < p_light


# --- scoring -----------------------------------------------------------------


def test_score_all_writes_p_trash_only_for_messages_in_exchanges(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_message_model(conn)  # type: ignore[misc]

    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (99999, 1, 1, 1, 'alice', ?, 'not in any exchange', ?, '{}')",
        (NOW_TEXT, NOW_TEXT),
    )

    scored = score_all(conn, model, version)

    assert scored == GROUP_SIZE * 4
    orphan = conn.execute("SELECT p_trash FROM messages WHERE id = 99999").fetchone()
    assert orphan["p_trash"] is None

    rows = conn.execute(
        "SELECT m.p_trash AS p_trash, m.p_trash_model AS p_trash_model FROM messages m"
        " JOIN exchange_messages em ON em.message_id = m.id"
    ).fetchall()
    assert len(rows) == GROUP_SIZE * 4
    for row in rows:
        assert row["p_trash"] is not None
        assert row["p_trash_model"] == version

    keep_row = conn.execute("SELECT p_trash FROM messages WHERE id = 1").fetchone()
    trash_row = conn.execute("SELECT p_trash FROM messages WHERE id = 1000").fetchone()
    assert keep_row["p_trash"] < trash_row["p_trash"]


def test_score_stale_only_rescores_messages_behind_the_latest_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_full_corpus(conn)
    train_and_store(conn, FixedClock(NOW))
    version, model = load_latest_message_model(conn)  # type: ignore[misc]

    first_pass = score_stale(conn, model, version)
    assert first_pass == GROUP_SIZE * 4

    second_pass = score_stale(conn, model, version)
    assert second_pass == 0

    conn.execute("UPDATE messages SET p_trash_model = NULL WHERE id = 1")
    third_pass = score_stale(conn, model, version)
    assert third_pass == 1


def test_score_all_workers_two_matches_workers_one(tmp_path: Path) -> None:
    conn_serial = open_database(tmp_path / "serial.db")
    migrate(conn_serial)
    seed_full_corpus(conn_serial)
    train_and_store(conn_serial, FixedClock(NOW))
    version_serial, model_serial = load_latest_message_model(conn_serial)  # type: ignore[misc]

    conn_parallel = open_database(tmp_path / "parallel.db")
    migrate(conn_parallel)
    seed_full_corpus(conn_parallel)
    train_and_store(conn_parallel, FixedClock(NOW))
    version_parallel, model_parallel = load_latest_message_model(conn_parallel)  # type: ignore[misc]

    scored_serial = score_all(conn_serial, model_serial, version_serial, workers=1)
    scored_parallel = score_all(conn_parallel, model_parallel, version_parallel, workers=2)

    assert scored_serial == scored_parallel == GROUP_SIZE * 4
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
