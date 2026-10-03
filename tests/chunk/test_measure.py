import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from infovore.chunk.measure import (
    BUCKET_LABELS,
    MeasureRule,
    bucket_index,
    measure,
    render,
)
from infovore.cli import ExitCode
from infovore.db.connection import migrate, open_database
from tests.chunk.test_command import environment, run


def add(conn: sqlite3.Connection, channel: int, rows: list[tuple[int, int]], bot: int = 0) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO channels (id, guild_id, name, kind) VALUES (?, 9, ?, 'text')",
        (channel, f"chan{channel}"),
    )
    for message_id, minute in rows:
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json, author_is_bot)"
            " VALUES (?, ?, 9, 1, 'a', datetime('2026-01-01', ?), 'x', '2026-01-01', '{}',"
            " ?)",
            (message_id, channel, f"+{minute} minutes", bot),
        )


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "m.db")
    migrate(connection)
    add(connection, 1, [(1, 0), (2, 1), (3, 100), (4, 101), (5, 102)])
    add(connection, 2, [(10, 0), (11, 200), (12, 400)])
    add(connection, 3, [(20, 0)], bot=1)
    return connection


def test_bucket_index_maps_sizes_to_the_labelled_buckets() -> None:
    assert BUCKET_LABELS == ("1", "2", "3-5", "6-15", "16-49", "cap")
    sizes = [1, 2, 3, 5, 6, 15, 16, 49, 50, 80]
    assert [bucket_index(n, 50) for n in sizes] == [0, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_a_fixed_gap_rule_distribution_per_channel_and_overall(conn: sqlite3.Connection) -> None:
    results = measure(conn, [MeasureRule("gap 30", gap=timedelta(minutes=30))], 50, False)
    (result,) = results
    assert result.overall == (3, 1, 1, 0, 0, 0)
    assert result.by_channel["chan1"] == (0, 1, 1, 0, 0, 0)
    assert result.by_channel["chan2"] == (3, 0, 0, 0, 0, 0)
    assert "chan3" not in result.by_channel


def test_a_wider_gap_merges_fragments(conn: sqlite3.Connection) -> None:
    (result,) = measure(conn, [MeasureRule("gap 240", gap=timedelta(minutes=240))], 50, False)
    assert result.overall == (0, 0, 2, 0, 0, 0)


def test_an_adaptive_rule_derives_each_channels_own_gap(conn: sqlite3.Connection) -> None:
    rule = MeasureRule(
        "adaptive p90",
        percentile=90,
        floor=timedelta(minutes=30),
        ceiling=timedelta(hours=6),
    )
    (result,) = measure(conn, [rule], 50, False)
    assert result.gaps["chan1"] == timedelta(minutes=99)
    assert result.gaps["chan2"] == timedelta(minutes=200)
    assert result.overall == (0, 0, 2, 0, 0, 0)


def test_folding_attaches_a_lone_message_to_a_neighbour_within_the_wider_gap(
    conn: sqlite3.Connection,
) -> None:
    add(conn, 4, [(30, 0), (31, 1), (32, 50)])
    rule = MeasureRule("gap 30 fold x2", gap=timedelta(minutes=30), fold_factor=2)
    (result,) = measure(conn, [rule], 50, False)
    assert result.by_channel["chan4"] == (0, 0, 1, 0, 0, 0)
    plain = MeasureRule("gap 30", gap=timedelta(minutes=30))
    assert measure(conn, [plain], 50, False)[0].by_channel["chan4"] == (1, 1, 0, 0, 0, 0)


def test_bots_count_only_when_asked(conn: sqlite3.Connection) -> None:
    rule = MeasureRule("gap 30", gap=timedelta(minutes=30))
    (result,) = measure(conn, [rule], 50, True)
    assert result.by_channel["chan3"] == (1, 0, 0, 0, 0, 0)


def test_render_shows_overall_and_top_channels(conn: sqlite3.Connection) -> None:
    rules = [
        MeasureRule("gap 30", gap=timedelta(minutes=30)),
        MeasureRule(
            "adaptive p90", percentile=90, floor=timedelta(minutes=30), ceiling=timedelta(hours=6)
        ),
    ]
    text = render(measure(conn, rules, 50, False), top_channels=1)
    assert "overall" in text
    assert "gap 30" in text
    assert "adaptive p90" in text
    assert "1-2 msgs" in text
    assert "chan1" in text
    assert "chan2" not in text
    assert "gap=99m" in text


def test_the_command_measures_read_only(tmp_path: Path) -> None:
    env = environment(tmp_path)
    connection = open_database(env["INFOVORE_DB_PATH"])
    migrate(connection)
    add(connection, 1, [(1, 0), (2, 1), (3, 100)])
    connection.close()
    code, out, _ = run(
        ["chunk", "--measure", "--gap", "30", "--gap", "120", "--adaptive", "--fold", "2"],
        env,
    )
    assert code == ExitCode.OK
    assert "gap 30" in out
    assert "gap 120" in out
    assert "adaptive p90" in out
    check = open_database(env["INFOVORE_DB_PATH"])
    assert check.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0] == 0


def test_the_command_defaults_to_the_standard_gaps(tmp_path: Path) -> None:
    env = environment(tmp_path)
    connection = open_database(env["INFOVORE_DB_PATH"])
    migrate(connection)
    connection.close()
    code, out, _ = run(["chunk", "--measure"], env)
    assert code == ExitCode.OK
    for minutes in (30, 60, 120, 240):
        assert f"gap {minutes}" in out
