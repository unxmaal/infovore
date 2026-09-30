import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.label_events import self_consistency
from infovore.db.message_labels import set_message_label
from infovore.rows import MessageLabel, MessageLabelSource
from infovore.sift.export import SiftBatchMessage
from infovore.sift.serve import ServeApp
from infovore.timing import FixedClock

EARLIER = datetime(2026, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 2, 1, tzinfo=UTC)


def _message(conn: sqlite3.Connection, message_id: int) -> SiftBatchMessage:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, 1, 1, 1, 'author', ?, 'hello', ?, '{}')",
        (message_id, EARLIER.isoformat(), EARLIER.isoformat()),
    )
    return SiftBatchMessage(
        id=message_id,
        exchange_id=message_id,
        channel_name="general",
        author_name="author",
        created_at=EARLIER,
        content="hello",
        p_trash=None,
    )


@pytest.fixture
def app(tmp_path: Path) -> ServeApp:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    fresh = _message(conn, 10)
    repeat = _message(conn, 11)
    # the repeat already carries a human label from an earlier round
    set_message_label(
        conn, 11, MessageLabel.KEEP, MessageLabelSource.HUMAN, "sift-serve:batch-004", EARLIER
    )
    return ServeApp(conn, [fresh, repeat], "batch-005", tmp_path, FixedClock(NOW))


def test_a_label_from_an_earlier_round_does_not_count_as_done(app: ServeApp) -> None:
    progress = app.progress()

    assert progress.total == 2
    assert progress.labeled == 0


def test_a_repeat_is_offered_for_judgment_again(app: ServeApp) -> None:
    rows = cast(list[dict[str, object]], app.state()["messages"])

    unlabeled = [row["id"] for row in rows if row["label"] is None]
    assert unlabeled == ["10", "11"]


def test_re_judging_a_repeat_counts_toward_this_round(app: ServeApp) -> None:
    app.label(11, MessageLabel.TRASH)

    progress = app.progress()
    assert progress.labeled == 1
    assert progress.trash == 1


def test_re_judging_a_repeat_makes_self_consistency_measurable(app: ServeApp) -> None:
    app.label(11, MessageLabel.TRASH)

    report = self_consistency(app._conn)
    assert report.repeated == 1
    assert report.agreed == 0


def test_agreeing_with_the_earlier_round_scores_as_agreement(app: ServeApp) -> None:
    app.label(11, MessageLabel.KEEP)

    report = self_consistency(app._conn)
    assert report.repeated == 1
    assert report.agreed == 1


def test_a_label_from_this_round_does_count_as_done(app: ServeApp) -> None:
    app.label(10, MessageLabel.KEEP)

    progress = app.progress()
    assert progress.labeled == 1
    assert progress.keep == 1
