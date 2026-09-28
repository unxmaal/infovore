import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.rows import MessageLabel, MessageLabelSource
from infovore.sift.export import MANIFEST_NAME, export_batch
from infovore.sift.importer import MissingManifestError
from infovore.sift.rules import BulkRule, RuleType
from infovore.sift.sampling import NoScoredMessagesError, SiftStrategy
from infovore.sift.serve import (
    ServeApp,
    UnknownMessageError,
    build_serve_app,
    load_batch_messages,
    resolve_batch,
)
from infovore.timing import FixedClock

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _channel(conn: sqlite3.Connection, channel_id: int, name: str) -> None:
    conn.execute(
        "INSERT INTO channels (id, guild_id, name, kind) VALUES (?, 1, ?, 'text')",
        (channel_id, name),
    )


def _message_with_exchange(
    conn: sqlite3.Connection,
    message_id: int,
    channel_id: int,
    author_id: int = 1,
    author_name: str = "alice",
    content: str = "hello there",
    created_at: str = NOW.isoformat(),
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 1, ?, ?, ?, ?, ?, '{}')",
        (message_id, channel_id, author_id, author_name, created_at, content, NOW.isoformat()),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, ?, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
        (
            message_id,
            channel_id,
            message_id,
            message_id,
            created_at,
            created_at,
            f"h{message_id}",
        ),
    )
    conn.execute(
        "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
        (message_id, message_id),
    )


def seeded(tmp_path: Path, name: str = "x.db") -> sqlite3.Connection:
    conn = open_database(tmp_path / name)
    migrate(conn)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    for i in range(1, 4):
        _message_with_exchange(conn, i, 1, content=f"general chatter {i}")
    for i in range(4, 7):
        _message_with_exchange(conn, i, 2, content=f"food talk {i}")
    return conn


# --- load_batch_messages ----------------------------------------------------


def test_load_batch_messages_returns_rows_in_order(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    messages = load_batch_messages(conn, [3, 1, 2])
    assert [m.id for m in messages] == [1, 2, 3]


def test_load_batch_messages_excludes_opted_out_authors_defensively(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1, author_id=1, content="from a")
    _message_with_exchange(conn, 2, 1, author_id=2, content="from b")
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (1, ?)", (NOW.isoformat(),))

    messages = load_batch_messages(conn, [1, 2])

    assert [m.id for m in messages] == [2]


def test_load_batch_messages_empty(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    assert load_batch_messages(conn, []) == []


# --- resolve_batch -----------------------------------------------------------


def test_resolve_batch_reads_an_existing_manifest(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    out_dir = tmp_path / "batch"
    export_batch(
        conn, size=3, strategy=SiftStrategy.RANDOM, seed=0, mix=0.5, out_dir=out_dir, now=NOW
    )

    batch = resolve_batch(
        conn,
        dir_=out_dir,
        new=False,
        size=0,
        strategy=SiftStrategy.RANDOM,
        seed=0,
        mix=0.5,
        out_dir=None,
        now=NOW,
    )

    manifest = json.loads((out_dir / MANIFEST_NAME).read_text())
    assert batch.dir == out_dir
    assert batch.message_ids == manifest["message_ids"]


def test_resolve_batch_missing_manifest_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    conn = seeded(tmp_path)
    with pytest.raises(MissingManifestError):
        resolve_batch(
            conn,
            dir_=empty,
            new=False,
            size=0,
            strategy=SiftStrategy.RANDOM,
            seed=0,
            mix=0.5,
            out_dir=None,
            now=NOW,
        )


def test_resolve_batch_new_samples_and_writes_a_manifest(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    out_dir = tmp_path / "fresh"

    batch = resolve_batch(
        conn,
        dir_=None,
        new=True,
        size=2,
        strategy=SiftStrategy.RANDOM,
        seed=1,
        mix=0.5,
        out_dir=out_dir,
        now=NOW,
    )

    assert batch.dir == out_dir
    assert len(batch.message_ids) == 2
    assert (out_dir / MANIFEST_NAME).exists()


def test_resolve_batch_new_propagates_no_scored_messages_error(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    out_dir = tmp_path / "fresh"
    with pytest.raises(NoScoredMessagesError):
        resolve_batch(
            conn,
            dir_=None,
            new=True,
            size=2,
            strategy=SiftStrategy.UNCERTAIN,
            seed=1,
            mix=0.5,
            out_dir=out_dir,
            now=NOW,
        )


# --- build_serve_app ---------------------------------------------------------


def test_build_serve_app_from_existing_dir(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    out_dir = tmp_path / "batch1"
    export_batch(
        conn, size=6, strategy=SiftStrategy.RANDOM, seed=0, mix=0.5, out_dir=out_dir, now=NOW
    )

    app = build_serve_app(
        conn,
        dir_=out_dir,
        new=False,
        size=0,
        strategy=SiftStrategy.RANDOM,
        seed=0,
        mix=0.5,
        out_dir=None,
        scratch_dir=tmp_path / "scratch",
        clock=FixedClock(NOW),
    )

    assert app.batch_name == "batch1"
    assert len(app.messages()) == 6


# --- ServeApp: state / progress ----------------------------------------------


def _app(conn: sqlite3.Connection, tmp_path: Path, message_ids: list[int]) -> ServeApp:
    messages = load_batch_messages(conn, message_ids)
    return ServeApp(conn, messages, "batch1", tmp_path / "scratch", FixedClock(NOW))


def test_progress_starts_unlabeled(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1, 2, 4])
    progress = app.progress()
    assert progress.total == 3
    assert progress.labeled == 0
    assert progress.keep == 0
    assert progress.trash == 0
    assert progress.remaining_by_channel == {"general": 2, "food": 1}


def test_state_reports_null_label_before_any_action(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1])
    state = app.state()
    assert state["batch"] == "batch1"
    messages = state["messages"]
    assert isinstance(messages, list)
    assert messages[0]["id"] == "1"
    assert messages[0]["label"] is None


def test_source_ref_includes_batch_name(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1])
    assert app.source_ref == "sift-serve:batch1"


# --- ServeApp: label / undo ---------------------------------------------------


def test_label_writes_a_human_row_with_the_batch_source_ref(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1, 2])

    app.label(1, MessageLabel.TRASH)

    row = conn.execute(
        "SELECT label, source, source_ref FROM message_labels WHERE message_id = 1"
    ).fetchone()
    assert row["label"] == "trash"
    assert row["source"] == "human"
    assert row["source_ref"] == "sift-serve:batch1"
    assert app.progress().labeled == 1
    assert app.progress().trash == 1


def test_label_unknown_message_raises(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1])
    with pytest.raises(UnknownMessageError):
        app.label(999, MessageLabel.KEEP)


def test_reloading_resumes_already_labeled_messages(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1, 2])
    app.label(1, MessageLabel.KEEP)

    reloaded = _app(conn, tmp_path, [1, 2])
    state = reloaded.state()
    state_messages = state["messages"]
    assert isinstance(state_messages, list)
    messages = {m["id"]: m["label"] for m in state_messages}
    assert messages["1"] == "keep"
    assert messages["2"] is None
    assert reloaded.progress().labeled == 1


def test_undo_reverts_a_label_that_had_no_prior_state(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1])
    app.label(1, MessageLabel.TRASH)

    undone = app.undo()

    assert undone is True
    row = conn.execute("SELECT * FROM message_labels WHERE message_id = 1").fetchone()
    assert row is None
    assert app.progress().labeled == 0


def test_undo_restores_a_prior_human_label_rather_than_deleting(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1])
    app.label(1, MessageLabel.KEEP)
    app.label(1, MessageLabel.TRASH)

    assert app.undo() is True

    row = conn.execute("SELECT label FROM message_labels WHERE message_id = 1").fetchone()
    assert row["label"] == "keep"


def test_undo_is_lifo_across_several_actions(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1, 2])
    app.label(1, MessageLabel.KEEP)
    app.label(2, MessageLabel.TRASH)

    assert app.undo() is True
    row2 = conn.execute("SELECT * FROM message_labels WHERE message_id = 2").fetchone()
    assert row2 is None
    row1 = conn.execute("SELECT label FROM message_labels WHERE message_id = 1").fetchone()
    assert row1["label"] == "keep"

    assert app.undo() is True
    row1_after = conn.execute("SELECT * FROM message_labels WHERE message_id = 1").fetchone()
    assert row1_after is None


def test_undo_with_nothing_to_undo_returns_false(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1])
    assert app.undo() is False


# --- ServeApp: trash_rest_of_channel -------------------------------------------


def test_trash_rest_of_channel_only_trashes_unlabeled_messages_in_that_channel(
    tmp_path: Path,
) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1, 2, 3, 4])
    app.label(2, MessageLabel.KEEP)

    trashed = app.trash_rest_of_channel("general")

    assert trashed == 2  # messages 1 and 3; message 2 already kept, message 4 is #food
    labels = {
        row["message_id"]: row["label"]
        for row in conn.execute("SELECT message_id, label FROM message_labels")
    }
    assert labels == {1: "trash", 2: "keep", 3: "trash"}


def test_trash_rest_of_channel_with_nothing_left_returns_zero(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1])
    app.label(1, MessageLabel.TRASH)
    assert app.trash_rest_of_channel("general") == 0


def test_undo_reverts_the_whole_trash_rest_of_channel_batch(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1, 2, 3])
    app.trash_rest_of_channel("general")

    assert app.undo() is True

    remaining = conn.execute("SELECT COUNT(*) AS n FROM message_labels").fetchone()["n"]
    assert remaining == 0


# --- ServeApp: bulk rules -------------------------------------------------------


def test_preview_rule_reflects_current_batch_labels(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1, 2, 3])
    app.label(2, MessageLabel.KEEP)
    rule = BulkRule(type=RuleType.CONTAINS, value="chatter")

    preview = app.preview_rule(rule)

    assert preview.matched_count == 3
    assert preview.conflicts == (2,)


def test_apply_rule_trashes_matches_overriding_conflicts_and_saves_the_rule(
    tmp_path: Path,
) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1, 2, 3])
    app.label(2, MessageLabel.KEEP)
    rule = BulkRule(type=RuleType.CONTAINS, value="chatter")

    preview, saved_path = app.apply_rule(rule, "general-chatter")

    assert preview.matched_count == 3
    assert preview.conflict_count == 1
    labels = {
        row["message_id"]: row["label"]
        for row in conn.execute("SELECT message_id, label FROM message_labels")
    }
    assert labels == {1: "trash", 2: "trash", 3: "trash"}
    assert saved_path.exists()
    saved = json.loads(saved_path.read_text())
    assert saved["type"] == "contains"
    assert saved["value"] == "chatter"


def test_undo_reverts_a_bulk_rule_apply_restoring_conflicting_keep(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1, 2, 3])
    app.label(2, MessageLabel.KEEP)
    rule = BulkRule(type=RuleType.CONTAINS, value="chatter")
    app.apply_rule(rule, "general-chatter")

    assert app.undo() is True

    labels = {
        row["message_id"]: row["label"]
        for row in conn.execute(
            "SELECT message_id, label FROM message_labels WHERE source = ?",
            (MessageLabelSource.HUMAN.value,),
        )
    }
    assert labels == {2: "keep"}


def test_apply_rule_with_no_matches_does_not_record_an_undo_action(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = _app(conn, tmp_path, [1])
    rule = BulkRule(type=RuleType.CONTAINS, value="nonexistent-phrase")

    preview, _ = app.apply_rule(rule, "noop")

    assert preview.matched_count == 0
    assert app.undo() is False


def test_progress_and_state_with_an_empty_batch(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    app = ServeApp(conn, [], "empty-batch", tmp_path / "scratch", FixedClock(NOW))

    progress = app.progress()

    assert progress.total == 0
    assert progress.labeled == 0
    assert progress.remaining_by_channel == {}
    assert app.state()["messages"] == []
