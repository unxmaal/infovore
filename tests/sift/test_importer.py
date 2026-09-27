import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.message_labels import effective_message_labels
from infovore.rows import MessageLabel
from infovore.sift.export import export_batch
from infovore.sift.importer import (
    KEPT_CSV_NAME,
    TRASH_REGEXES_CSV_NAME,
    ChannelCounts,
    MissingManifestError,
    NoSiftResultsFoundError,
    import_batch,
    save_trash_rules,
)
from infovore.sift.sampling import SiftStrategy

NOW = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
IMPORTED_AT = datetime(2026, 1, 3, tzinfo=UTC)


def _channel(conn: sqlite3.Connection, channel_id: int, name: str) -> None:
    conn.execute(
        "INSERT INTO channels (id, guild_id, name, kind) VALUES (?, 1, ?, 'text')",
        (channel_id, name),
    )


def _message_with_exchange(
    conn: sqlite3.Connection,
    message_id: int,
    channel_id: int,
    content: str,
    author_name: str = "alice",
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 1, 1, ?, ?, ?, ?, '{}')",
        (message_id, channel_id, author_name, NOW, content, NOW),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, ?, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
        (message_id, channel_id, message_id, message_id, NOW, NOW, f"h{message_id}"),
    )
    conn.execute(
        "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
        (message_id, message_id),
    )


def seeded_batch(tmp_path: Path) -> tuple[sqlite3.Connection, Path]:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    _message_with_exchange(conn, 1, 1, "lol so true every time")
    _message_with_exchange(conn, 2, 1, "does anyone remember the old rules")
    _message_with_exchange(conn, 3, 2, "here is my sourdough starter recipe")
    out_dir = tmp_path / "batch"
    export_batch(
        conn,
        size=3,
        strategy=SiftStrategy.RANDOM,
        seed=0,
        mix=0.5,
        out_dir=out_dir,
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )
    return conn, out_dir


def test_import_missing_manifest_raises(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(MissingManifestError):
        import_batch(conn, empty_dir, IMPORTED_AT)


def test_import_with_no_sift_result_files_raises(tmp_path: Path) -> None:
    conn, out_dir = seeded_batch(tmp_path)
    with pytest.raises(NoSiftResultsFoundError):
        import_batch(conn, out_dir, IMPORTED_AT)


def test_import_kept_csv_marks_missing_ids_as_trash(tmp_path: Path) -> None:
    conn, out_dir = seeded_batch(tmp_path)
    (out_dir / KEPT_CSV_NAME).write_text("msg\n3\n")

    report = import_batch(conn, out_dir, IMPORTED_AT)

    assert report.path == "a"
    assert report.keep == 1
    assert report.trash == 2
    assert report.source_ref == "sift:2026-01-02T00:00:00+00:00:a"
    labels = effective_message_labels(conn)
    assert labels == {1: MessageLabel.TRASH, 2: MessageLabel.TRASH, 3: MessageLabel.KEEP}


def test_import_kept_csv_channel_counts(tmp_path: Path) -> None:
    conn, out_dir = seeded_batch(tmp_path)
    (out_dir / KEPT_CSV_NAME).write_text("msg\n3\n")

    report = import_batch(conn, out_dir, IMPORTED_AT)

    assert report.by_channel == {
        "general": ChannelCounts(keep=0, trash=2),
        "food": ChannelCounts(keep=1, trash=0),
    }


def test_import_trash_regexes_matches_against_batch_log_lines(tmp_path: Path) -> None:
    conn, out_dir = seeded_batch(tmp_path)
    (out_dir / TRASH_REGEXES_CSV_NAME).write_text("pattern\nlol\n")

    report = import_batch(conn, out_dir, IMPORTED_AT)

    assert report.path == "b"
    assert report.source_ref == "sift:2026-01-02T00:00:00+00:00:b"
    labels = effective_message_labels(conn)
    assert labels[1] == MessageLabel.TRASH
    assert labels[2] == MessageLabel.KEEP
    assert labels[3] == MessageLabel.KEEP
    assert report.patterns == ("lol",)


def test_import_trash_regexes_with_multiple_patterns(tmp_path: Path) -> None:
    conn, out_dir = seeded_batch(tmp_path)
    (out_dir / TRASH_REGEXES_CSV_NAME).write_text("pattern\nlol\nsourdough\n")

    report = import_batch(conn, out_dir, IMPORTED_AT)

    labels = effective_message_labels(conn)
    assert labels == {1: MessageLabel.TRASH, 2: MessageLabel.KEEP, 3: MessageLabel.TRASH}
    assert report.keep == 1
    assert report.trash == 2


def test_import_prefers_kept_csv_when_both_files_exist(tmp_path: Path) -> None:
    conn, out_dir = seeded_batch(tmp_path)
    (out_dir / KEPT_CSV_NAME).write_text("msg\n1\n2\n3\n")
    (out_dir / TRASH_REGEXES_CSV_NAME).write_text("pattern\nlol\n")

    report = import_batch(conn, out_dir, IMPORTED_AT)

    assert report.path == "a"
    labels = effective_message_labels(conn)
    assert labels == {1: MessageLabel.KEEP, 2: MessageLabel.KEEP, 3: MessageLabel.KEEP}


def test_reimport_replaces_prior_human_labels(tmp_path: Path) -> None:
    conn, out_dir = seeded_batch(tmp_path)
    (out_dir / KEPT_CSV_NAME).write_text("msg\n3\n")
    import_batch(conn, out_dir, IMPORTED_AT)

    (out_dir / KEPT_CSV_NAME).write_text("msg\n1\n2\n3\n")
    later = datetime(2026, 1, 4, tzinfo=UTC)
    report = import_batch(conn, out_dir, later)

    assert report.keep == 3
    assert report.trash == 0
    labels = effective_message_labels(conn)
    assert labels == {1: MessageLabel.KEEP, 2: MessageLabel.KEEP, 3: MessageLabel.KEEP}


def test_import_with_an_empty_kept_csv_marks_everything_trash(tmp_path: Path) -> None:
    conn, out_dir = seeded_batch(tmp_path)
    (out_dir / KEPT_CSV_NAME).write_text("msg\n")

    report = import_batch(conn, out_dir, IMPORTED_AT)

    assert report.keep == 0
    assert report.trash == 3


def test_import_ignores_batch_log_lines_without_a_msg_tag(tmp_path: Path) -> None:
    conn, out_dir = seeded_batch(tmp_path)
    with (out_dir / "batch.log").open("a") as handle:
        handle.write("not a real sift line at all\n")
    (out_dir / TRASH_REGEXES_CSV_NAME).write_text("pattern\nlol\n")

    report = import_batch(conn, out_dir, IMPORTED_AT)

    assert report.keep == 2
    assert report.trash == 1


def test_import_an_empty_batch_has_no_channel_counts(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    out_dir = tmp_path / "empty_batch"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(
        '{"message_ids": [], "strategy": "random", "seed": 0,'
        ' "created_at": "2026-01-02T00:00:00+00:00"}'
    )
    (out_dir / KEPT_CSV_NAME).write_text("msg\n")

    report = import_batch(conn, out_dir, IMPORTED_AT)

    assert report.keep == 0
    assert report.trash == 0
    assert report.by_channel == {}


def test_save_trash_rules_writes_a_json_file(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    at = datetime(2026, 1, 5, tzinfo=UTC)

    path = save_trash_rules(scratch, "food-chatter", ("lol", "sourdough"), at)

    assert path == scratch / "sift_trash_rules" / "food-chatter.json"
    import json

    data = json.loads(path.read_text())
    assert data["name"] == "food-chatter"
    assert data["patterns"] == ["lol", "sourdough"]
    assert data["saved_at"] == "2026-01-05T00:00:00+00:00"
