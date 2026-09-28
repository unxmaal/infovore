import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.connection import migrate, open_database
from infovore.sift.export import (
    BATCH_TEXT_LIMIT,
    SiftBatchMessage,
    export_batch,
    fetch_batch_messages,
    render_batch_line,
)
from infovore.sift.lnav_format import LNAV_FORMAT_JSON
from infovore.sift.sampling import SiftStrategy

NOW = datetime(2026, 1, 1, tzinfo=UTC).isoformat()


def _channel(conn: sqlite3.Connection, channel_id: int, name: str) -> None:
    conn.execute(
        "INSERT INTO channels (id, guild_id, name, kind) VALUES (?, 1, ?, 'text')",
        (channel_id, name),
    )


def _message_with_exchange(
    conn: sqlite3.Connection,
    message_id: int,
    channel_id: int,
    author_name: str = "alice",
    content: str = "hello",
    p_trash: float | None = None,
    created_at: str = NOW,
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json, p_trash)"
        " VALUES (?, ?, 1, 1, ?, ?, ?, ?, '{}', ?)",
        (message_id, channel_id, author_name, created_at, content, NOW, p_trash),
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


def seeded(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def test_render_batch_line_format_with_a_score() -> None:
    row = SiftBatchMessage(
        id=101,
        exchange_id=11,
        channel_name="general",
        author_name="alice",
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        content="hello world",
        p_trash=0.5,
    )
    line = render_batch_line(row)
    assert line == "2026-01-01T12:00:00+00:00 #general alice [msg:101 ex:11 p:0.50] hello world"


def test_render_batch_line_format_without_a_score() -> None:
    row = SiftBatchMessage(
        id=102,
        exchange_id=11,
        channel_name="general",
        author_name="bob dodd",
        created_at=datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC),
        content="hi",
        p_trash=None,
    )
    line = render_batch_line(row)
    assert line == "2026-01-01T12:00:05+00:00 #general bob dodd [msg:102 ex:11 p:-] hi"


def test_render_batch_line_collapses_newlines_and_whitespace() -> None:
    row = SiftBatchMessage(
        id=1,
        exchange_id=1,
        channel_name="general",
        author_name="a",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        content="line one\nline   two\t\tline three",
        p_trash=None,
    )
    line = render_batch_line(row)
    assert "\n" not in line
    assert "line one line two line three" in line


def test_render_batch_line_truncates_long_content_with_an_ellipsis() -> None:
    row = SiftBatchMessage(
        id=1,
        exchange_id=1,
        channel_name="general",
        author_name="a",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        content="x" * 500,
        p_trash=None,
    )
    line = render_batch_line(row)
    body = line.split("] ", 1)[1]
    assert len(body) <= BATCH_TEXT_LIMIT
    assert body.endswith("...")


def test_render_batch_line_leaves_short_content_untouched() -> None:
    row = SiftBatchMessage(
        id=1,
        exchange_id=1,
        channel_name="general",
        author_name="a",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        content="short message",
        p_trash=None,
    )
    line = render_batch_line(row)
    assert line.endswith("short message")


def test_fetch_batch_messages_orders_chronologically(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _message_with_exchange(
        conn, 2, 1, created_at=datetime(2026, 1, 1, 12, 5, tzinfo=UTC).isoformat()
    )
    _message_with_exchange(
        conn, 1, 1, created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC).isoformat()
    )
    batch = fetch_batch_messages(conn, [1, 2])
    assert [row.id for row in batch] == [1, 2]


def test_fetch_batch_messages_empty_ids_returns_empty(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    assert fetch_batch_messages(conn, []) == []


def test_export_batch_writes_batch_log_format_and_manifest(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    for i in range(1, 4):
        _message_with_exchange(conn, i, 1, content=f"general message {i}")
    for i in range(4, 7):
        _message_with_exchange(conn, i, 2, content=f"food message {i}")

    out_dir = tmp_path / "out"
    report = export_batch(
        conn,
        size=4,
        strategy=SiftStrategy.RANDOM,
        seed=0,
        mix=0.5,
        out_dir=out_dir,
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert report.count == 4
    assert report.batch_log_path == out_dir / "batch.log"
    assert report.format_path == out_dir / "infovore-sift.json"
    assert report.manifest_path == out_dir / "manifest.json"

    lines = report.batch_log_path.read_text().splitlines()
    assert len(lines) == 4
    for line in lines:
        assert re.match(
            r"^\S+ #\S+ .*? \[msg:\d+ ex:\d+ p:(-|[\d.]+)\] .*$",
            line,
        )

    format_text = report.format_path.read_text()
    assert format_text == LNAV_FORMAT_JSON
    json.loads(format_text)

    manifest = json.loads(report.manifest_path.read_text())
    assert manifest["strategy"] == "random"
    assert manifest["seed"] == 0
    assert manifest["size"] == 4
    assert manifest["created_at"] == "2026-01-02T00:00:00+00:00"
    assert len(manifest["message_ids"]) == 4
    assert set(manifest["message_ids"]) <= set(range(1, 7))


def test_export_batch_creates_out_dir(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1)
    out_dir = tmp_path / "nested" / "out"
    report = export_batch(
        conn,
        size=1,
        strategy=SiftStrategy.RANDOM,
        seed=0,
        mix=0.5,
        out_dir=out_dir,
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert report.count == 1
    assert out_dir.is_dir()


def test_export_batch_respects_exclude_channels(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    _message_with_exchange(conn, 1, 1, content="general message")
    _message_with_exchange(conn, 2, 2, content="food message")

    out_dir = tmp_path / "out"
    report = export_batch(
        conn,
        size=10,
        strategy=SiftStrategy.RANDOM,
        seed=0,
        mix=0.5,
        out_dir=out_dir,
        now=datetime(2026, 1, 2, tzinfo=UTC),
        exclude_channels=frozenset({"food"}),
    )

    assert report.count == 1
    manifest = json.loads(report.manifest_path.read_text())
    assert manifest["message_ids"] == [1]


def test_export_batch_respects_include_channels(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    _message_with_exchange(conn, 1, 1, content="general message")
    _message_with_exchange(conn, 2, 2, content="food message")

    out_dir = tmp_path / "out"
    report = export_batch(
        conn,
        size=10,
        strategy=SiftStrategy.RANDOM,
        seed=0,
        mix=0.5,
        out_dir=out_dir,
        now=datetime(2026, 1, 2, tzinfo=UTC),
        include_channels=frozenset({"general"}),
    )

    assert report.count == 1
    manifest = json.loads(report.manifest_path.read_text())
    assert manifest["message_ids"] == [1]


def test_lnav_format_json_is_valid_and_matches_the_batch_log_pattern() -> None:
    parsed = json.loads(LNAV_FORMAT_JSON)
    fmt = parsed["infovore_sift"]
    assert fmt["value"]["channel"]["identifier"] is True
    assert fmt["value"]["author"]["identifier"] is True
    assert re.search(fmt["file-pattern"], "some/dir/batch.log")
