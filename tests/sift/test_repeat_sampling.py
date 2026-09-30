import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.connection import migrate, open_database
from infovore.db.message_labels import set_message_label
from infovore.rows import MessageLabel, MessageLabelSource
from infovore.sift.sampling import (
    SiftStrategy,
    repeat_message_pool,
    select_sift_sample,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
NOW_TEXT = NOW.isoformat()


def _channel(conn: sqlite3.Connection, channel_id: int, name: str) -> None:
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind) VALUES (?, 1, NULL, ?, 'text')",
        (channel_id, name),
    )


def _message(conn: sqlite3.Connection, message_id: int, channel_id: int = 1) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 1, 1, 'author', ?, 'hello', ?, '{}')",
        (message_id, channel_id, NOW_TEXT, NOW_TEXT),
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


def _judged(conn: sqlite3.Connection, message_id: int) -> None:
    set_message_label(
        conn,
        message_id,
        MessageLabel.KEEP,
        MessageLabelSource.HUMAN,
        "round:0",
        NOW,
    )


def _corpus(tmp_path: Path, fresh: int, judged: int) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    _channel(conn, 1, "general")
    for message_id in range(1, fresh + 1):
        _message(conn, message_id)
    for message_id in range(100, 100 + judged):
        _message(conn, message_id)
        _judged(conn, message_id)
    return conn


def test_repeat_pool_is_only_messages_already_judged(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, fresh=3, judged=2)

    pool = repeat_message_pool(conn)

    assert {row.id for row in pool} == {100, 101}


def test_no_repeats_by_default(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, fresh=5, judged=3)

    picked = select_sift_sample(conn, n=4, seed=1, strategy=SiftStrategy.RANDOM)

    assert len(picked) == 4
    assert not set(picked) & {100, 101, 102}


def test_repeat_seeds_exactly_that_many_already_judged_messages(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, fresh=10, judged=5)

    picked = select_sift_sample(conn, n=6, seed=1, strategy=SiftStrategy.RANDOM, repeat=2)

    judged = {message_id for message_id in picked if message_id >= 100}
    assert len(picked) == 6
    assert len(judged) == 2


def test_repeats_are_not_identifiable_by_position(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, fresh=10, judged=5)

    picked = select_sift_sample(conn, n=6, seed=1, strategy=SiftStrategy.RANDOM, repeat=2)

    assert picked == sorted(picked)


def test_repeat_draw_is_reproducible_for_a_seed(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, fresh=10, judged=5)

    first = select_sift_sample(conn, n=6, seed=7, strategy=SiftStrategy.RANDOM, repeat=2)
    second = select_sift_sample(conn, n=6, seed=7, strategy=SiftStrategy.RANDOM, repeat=2)

    assert first == second


def test_a_different_seed_draws_different_repeats(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, fresh=40, judged=20)

    first = select_sift_sample(conn, n=10, seed=1, strategy=SiftStrategy.RANDOM, repeat=4)
    second = select_sift_sample(conn, n=10, seed=2, strategy=SiftStrategy.RANDOM, repeat=4)

    assert first != second


def test_repeat_larger_than_the_judged_pool_takes_what_exists(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, fresh=10, judged=2)

    picked = select_sift_sample(conn, n=6, seed=1, strategy=SiftStrategy.RANDOM, repeat=5)

    judged = {message_id for message_id in picked if message_id >= 100}
    assert judged == {100, 101}
    assert len(picked) == 6


def test_repeat_never_exceeds_the_batch_size(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, fresh=10, judged=10)

    picked = select_sift_sample(conn, n=3, seed=1, strategy=SiftStrategy.RANDOM, repeat=5)

    assert len(picked) == 3


def test_repeats_work_with_the_uncertain_strategy(tmp_path: Path) -> None:
    conn = _corpus(tmp_path, fresh=10, judged=5)
    conn.execute("UPDATE messages SET p_trash = 0.5 WHERE id <= 10")

    picked = select_sift_sample(conn, n=6, seed=1, strategy=SiftStrategy.UNCERTAIN, repeat=2)

    judged = {message_id for message_id in picked if message_id >= 100}
    assert len(judged) == 2
    assert len(picked) == 6
