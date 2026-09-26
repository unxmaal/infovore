import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import get_exchange, insert_exchange
from infovore.db.labels import set_label
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule, Label, LabelSource
from infovore.timing import FixedClock
from infovore.triage.gate import passes_gate
from infovore.triage.train import load_latest_model, recommend, score_all, train_and_store

NOW = datetime(2026, 1, 1, tzinfo=UTC)

LORE_CONTENTS = [
    "PROM 6.5.22 part 030-1234-001 /usr/sbin/inst hinv nvram",
    "Octane2 jumper on the R12000 board fixes the boot hang, see /usr/var/log",
    "Indigo2 IRIX 6.5.19 install from /stand/cdrom worked after swmgr update",
    "XFS repair on the Onyx needed xfs_repair /dev/dsk/first, IP30 hinv confirmed",
]
NOISE_CONTENTS = [
    "lol gg no cap",
    "haha same here, anyone up for a call later",
    "lmao that meme again, xd",
    "brb grabbing coffee, back in a bit",
]

MIN_SCORE = 0.3
MIN_P_LORE = 0.5


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def seed_exchange(conn: sqlite3.Connection, index: int, channel_id: int, content: str) -> int:
    message_id = index
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 9, ?, 'alice', 0, ?, ?, ?, '{}')",
        (message_id, channel_id, message_id, NOW.isoformat(), content, NOW.isoformat()),
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


def test_bayes_training_loop_separates_lore_from_noise_and_gates_on_p_lore(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)

    lore_ids: list[int] = []
    noise_ids: list[int] = []
    index = 1
    for repeat in range(5):
        for channel_id, content in enumerate(LORE_CONTENTS, start=1):
            exchange_id = seed_exchange(conn, index, channel_id, f"{content} take {repeat}")
            set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
            lore_ids.append(exchange_id)
            index += 1
        for channel_id, content in enumerate(NOISE_CONTENTS, start=1):
            exchange_id = seed_exchange(conn, index, channel_id + 10, f"{content} take {repeat}")
            set_label(conn, exchange_id, Label.NOISE, LabelSource.HUMAN, None, NOW)
            noise_ids.append(exchange_id)
            index += 1

    report = train_and_store(conn, FixedClock(NOW), confusion_threshold=MIN_P_LORE)
    assert report.labels_used == len(lore_ids) + len(noise_ids)
    assert report.holdout_size > 0

    loaded = load_latest_model(conn)
    assert loaded is not None
    version, model = loaded
    scored = score_all(conn, model, version)
    assert scored == len(lore_ids) + len(noise_ids)

    def p_lore_of(exchange_id: int) -> float:
        exchange = get_exchange(conn, exchange_id)
        assert exchange is not None
        assert exchange.p_lore is not None
        return exchange.p_lore

    lore_p_lore = [p_lore_of(eid) for eid in lore_ids]
    noise_p_lore = [p_lore_of(eid) for eid in noise_ids]
    assert min(lore_p_lore) > max(noise_p_lore)

    a_lore_exchange = get_exchange(conn, lore_ids[0])
    a_noise_exchange = get_exchange(conn, noise_ids[0])
    assert a_lore_exchange is not None
    assert a_noise_exchange is not None
    assert passes_gate(a_lore_exchange, min_score=MIN_SCORE, min_p_lore=MIN_P_LORE) is True
    assert passes_gate(a_noise_exchange, min_score=MIN_SCORE, min_p_lore=MIN_P_LORE) is False

    result = recommend(conn, min_recall=0.9)
    assert result is not None
    metric, share = result
    assert metric.recall >= 0.9
    assert 0.0 <= share <= 1.0
