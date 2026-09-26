import sqlite3
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.snapshot import snapshot


def make_db(path: Path) -> sqlite3.Connection:
    conn = open_database(path)
    migrate(conn)
    return conn


def insert_message(conn: sqlite3.Connection, moment: str) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json) VALUES (1, 1, 9, 1, 'a', ?, 'x', ?, '{}')",
        (moment, moment),
    )


def test_snapshot_creates_a_consistent_copy(tmp_path: Path) -> None:
    conn = make_db(tmp_path / "infovore.db")
    dest = tmp_path / "out" / "snap.db"
    report = snapshot(conn, dest)
    assert report.dest == dest
    assert dest.exists()
    assert report.size_bytes == dest.stat().st_size
    assert report.size_bytes > 0
    assert report.user_version == conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()


def test_snapshot_refuses_to_overwrite_existing_dest(tmp_path: Path) -> None:
    conn = make_db(tmp_path / "infovore.db")
    dest = tmp_path / "snap.db"
    snapshot(conn, dest)
    with pytest.raises(FileExistsError):
        snapshot(conn, dest)
    conn.close()


def test_snapshot_force_overwrites_existing_dest(tmp_path: Path) -> None:
    conn = make_db(tmp_path / "infovore.db")
    dest = tmp_path / "snap.db"
    snapshot(conn, dest)
    insert_message(conn, "2026-01-01T00:00:00+00:00")
    report = snapshot(conn, dest, force=True)
    assert report.dest == dest
    check = sqlite3.connect(dest)
    assert check.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    check.close()
    conn.close()


def test_snapshot_leaves_no_temp_files_behind(tmp_path: Path) -> None:
    conn = make_db(tmp_path / "infovore.db")
    dest = tmp_path / "out" / "snap.db"
    snapshot(conn, dest)
    assert list(dest.parent.iterdir()) == [dest]
    conn.close()


def test_snapshot_cleans_up_temp_file_on_failure(tmp_path: Path) -> None:
    conn = make_db(tmp_path / "infovore.db")
    dest = tmp_path / "snap.db"
    conn.close()
    with pytest.raises(sqlite3.ProgrammingError):
        snapshot(conn, dest)
    assert not dest.exists()
    assert not any(entry.name.startswith(".snap.db") for entry in tmp_path.iterdir())


def test_snapshot_is_consistent_while_a_writer_has_an_uncommitted_transaction(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "infovore.db"
    writer = make_db(db_path)
    writer.execute("BEGIN IMMEDIATE")
    insert_message(writer, "2026-01-01T00:00:00+00:00")

    reader = open_database(db_path)
    dest = tmp_path / "snap.db"
    snapshot(reader, dest)

    before_commit = sqlite3.connect(dest)
    assert before_commit.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert before_commit.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    before_commit.close()

    writer.execute("COMMIT")

    dest_after = tmp_path / "snap-after.db"
    snapshot(reader, dest_after)
    after_commit = sqlite3.connect(dest_after)
    assert after_commit.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    assert after_commit.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    after_commit.close()

    writer.close()
    reader.close()
