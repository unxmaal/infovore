import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.eval.judge import (
    BAD_GROUPING,
    IRRELEVANT,
    JUDGE_INTERFACE_VERSION,
    JUDGE_SCORER,
    LIKELY_IRRELEVANT,
    MIN_TEXT_MESSAGES,
    RELEVANT,
    UNCERTAIN,
    InvalidLabelError,
    NotInExchangeError,
    QueueItem,
    UnknownQueueItemError,
    c1_queue,
    exchange_view,
    first_unjudged,
    frozen_queue,
    label_counts,
    labels_needed,
    likely_irrelevant_queue,
    progress,
    queue_stats,
    self_agreement,
    slice_progress,
    submit,
    uncertain_judged,
    uncertain_queue,
)
from infovore.eval.slices import BUILD, GOLD, GOLD_REPEATS

AT = datetime(2026, 10, 2, tzinfo=UTC)


def _exchange(
    conn: sqlite3.Connection, exchange_id: int, message_ids: list[int], parent: int | None = None
) -> None:
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash, parent_exchange_id)"
        " VALUES (?, 1, ?, ?, ?, ?, ?, 'quiet_gap', ?, ?)",
        (
            exchange_id,
            message_ids[0],
            message_ids[-1],
            AT.isoformat(),
            AT.isoformat(),
            len(message_ids),
            f"h{exchange_id}",
            parent,
        ),
    )
    for position, message_id in enumerate(message_ids, start=1):
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 9, 5, 'hal', ?, ?, ?, '{}')",
            (
                message_id,
                (AT + timedelta(minutes=message_id)).isoformat(),
                f"msg {message_id}",
                AT.isoformat(),
            ),
        )
        conn.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, ?)",
            (exchange_id, message_id, position),
        )


def _slice(conn: sqlite3.Connection, name: str, ids: list[int]) -> None:
    for position, exchange_id in enumerate(ids, start=1):
        conn.execute(
            "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
            " VALUES (?, ?, ?, 'test', 0, ?)",
            (name, exchange_id, position, AT.isoformat()),
        )


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 9, NULL, 'hardware', 'text')"
    )
    _exchange(connection, 1, [11, 12, 13])
    _exchange(connection, 2, [21, 22], parent=1)
    _exchange(connection, 3, [31])
    _exchange(connection, 4, [41])
    _exchange(connection, 5, [51])
    _slice(connection, GOLD, [1, 2, 3])
    _slice(connection, GOLD_REPEATS, [1])
    _slice(connection, BUILD, [3, 4, 5])
    return connection


def _ids(queue: list[QueueItem]) -> list[tuple[str, int, int]]:
    return [(i.slice_name, i.position, i.exchange_id) for i in queue]


def _label(conn: sqlite3.Connection, queue: list[QueueItem], index: int, label: str) -> None:
    submit(conn, queue, index, queue[index].exchange_id, label, AT)


def test_the_queue_is_gold_then_repeats_then_the_rest_of_s1(conn: sqlite3.Connection) -> None:
    assert _ids(frozen_queue(conn)) == [
        (GOLD, 1, 1),
        (GOLD, 2, 2),
        (GOLD, 3, 3),
        (GOLD_REPEATS, 1, 1),
        (BUILD, 1, 4),
        (BUILD, 2, 5),
    ]


def test_a_repeat_arrives_unmarked(conn: sqlite3.Connection) -> None:
    """The second showing must not be pre-labelled, or it measures memory of
    the page rather than the judgment."""
    queue = frozen_queue(conn)
    _label(conn, queue, 0, RELEVANT)

    assert exchange_view(conn, queue, 0).label == RELEVANT
    assert exchange_view(conn, queue, 3).label is None


def test_submitting_records_one_exchange_annotation(conn: sqlite3.Connection) -> None:
    queue = frozen_queue(conn)
    _label(conn, queue, 1, BAD_GROUPING)

    row = conn.execute("SELECT * FROM annotations").fetchone()
    assert (row["subject_kind"], row["subject_id"], row["scorer"]) == ("exchange", 2, JUDGE_SCORER)
    assert (row["label"], row["scorer_version"], row["reproducibility"]) == (
        BAD_GROUPING,
        JUDGE_INTERFACE_VERSION,
        "recorded",
    )
    assert row["source_ref"] == "judge:gold:2"
    assert row["created_at"].startswith("2026-10-02")


def test_an_unjudged_exchange_has_no_row(conn: sqlite3.Connection) -> None:
    queue = frozen_queue(conn)
    _label(conn, queue, 0, RELEVANT)

    assert progress(conn, queue) == (1, 6)
    assert first_unjudged(conn, queue) == 1
    assert exchange_view(conn, queue, 1).label is None


def test_judging_again_appends_and_the_newest_shows(conn: sqlite3.Connection) -> None:
    queue = frozen_queue(conn)
    _label(conn, queue, 0, RELEVANT)
    _label(conn, queue, 0, IRRELEVANT)

    assert exchange_view(conn, queue, 0).label == IRRELEVANT
    assert conn.execute("SELECT COUNT(*) AS n FROM annotations").fetchone()["n"] == 2


def test_nothing_left_means_no_resume_point(conn: sqlite3.Connection) -> None:
    queue = frozen_queue(conn)
    for index in range(len(queue)):
        _label(conn, queue, index, IRRELEVANT)

    assert first_unjudged(conn, queue) is None


def test_bad_submissions_are_refused_and_write_nothing(conn: sqlite3.Connection) -> None:
    queue = frozen_queue(conn)
    with pytest.raises(InvalidLabelError):
        submit(conn, queue, 0, 1, "fact", AT)
    with pytest.raises(NotInExchangeError, match="not exchange"):
        submit(conn, queue, 0, 2, RELEVANT, AT)
    with pytest.raises(UnknownQueueItemError):
        submit(conn, queue, -1, 1, RELEVANT, AT)
    with pytest.raises(UnknownQueueItemError):
        exchange_view(conn, queue, 6)
    assert conn.execute("SELECT COUNT(*) AS n FROM annotations").fetchone()["n"] == 0


def test_context_from_the_parent_is_shown_but_marked_as_context(conn: sqlite3.Connection) -> None:
    view = exchange_view(conn, frozen_queue(conn), 1)

    assert [(m.id, m.is_context) for m in view.messages] == [
        (11, True),
        (12, True),
        (13, True),
        (21, False),
        (22, False),
    ]


def test_the_view_carries_channel_link_and_position(conn: sqlite3.Connection) -> None:
    view = exchange_view(conn, frozen_queue(conn), 2)

    assert (view.index, view.total, view.exchange_id, view.channel) == (2, 6, 3, "hardware")
    assert view.messages[0].link == "https://discord.com/channels/9/1/31"
    assert view.messages[0].author == "hal"


def test_label_counts_use_the_newest_judgment_and_skip_repeats(conn: sqlite3.Connection) -> None:
    queue = frozen_queue(conn)
    _label(conn, queue, 0, RELEVANT)
    _label(conn, queue, 3, IRRELEVANT)
    _label(conn, queue, 1, IRRELEVANT)
    _label(conn, queue, 1, BAD_GROUPING)

    assert label_counts(conn) == {RELEVANT: 1, IRRELEVANT: 0, BAD_GROUPING: 1}


def test_labels_needed_counts_down_to_the_bayes_minimum() -> None:
    assert labels_needed({RELEVANT: 150, IRRELEVANT: 260, BAD_GROUPING: 9}) == {
        RELEVANT: 50,
        IRRELEVANT: 0,
    }


def test_slice_progress_counts_judged_per_slice(conn: sqlite3.Connection) -> None:
    queue = frozen_queue(conn)
    _label(conn, queue, 0, RELEVANT)
    _label(conn, queue, 4, RELEVANT)

    assert [(p.name, p.done, p.total) for p in slice_progress(conn)] == [
        (GOLD, 1, 3),
        (GOLD_REPEATS, 0, 1),
        (BUILD, 1, 2),
    ]


def test_self_agreement_compares_the_two_showings(conn: sqlite3.Connection) -> None:
    queue = frozen_queue(conn)
    _label(conn, queue, 0, RELEVANT)
    assert self_agreement(conn).rate is None

    _label(conn, queue, 3, RELEVANT)
    assert (self_agreement(conn).exchanges, self_agreement(conn).rate) == (1, 1.0)

    _label(conn, queue, 3, IRRELEVANT)
    assert self_agreement(conn).rate == 0.0


def _texty(conn: sqlite3.Connection, *exchange_ids: int) -> None:
    for eid in exchange_ids:
        have = conn.execute(
            "SELECT COUNT(*) FROM exchange_messages WHERE exchange_id = ?", (eid,)
        ).fetchone()[0]
        for extra in range(have, MIN_TEXT_MESSAGES):
            mid = eid * 1000 + extra
            conn.execute(
                "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
                " created_at, content, ingested_at, raw_json)"
                " VALUES (?, 1, 9, 5, 'hal', ?, 'more text', ?, '{}')",
                (mid, AT.isoformat(), AT.isoformat()),
            )
            conn.execute(
                "INSERT INTO exchange_messages (exchange_id, message_id, position)"
                " VALUES (?, ?, ?)",
                (eid, mid, extra + 1),
            )


def test_the_uncertain_queue_is_closest_to_a_coin_flip_first(conn: sqlite3.Connection) -> None:
    _texty(conn, 2, 3, 5)
    _gate(conn, {1: 0.9, 2: 0.45, 3: 0.52, 4: None, 5: 0.1})
    queue = uncertain_queue(3, frozenset())(conn)

    assert [i.exchange_id for i in queue] == [3, 2, 1, 5]
    assert [i.position for i in queue] == [3, 2, 1, 5]


def _gate(conn: sqlite3.Connection, scores: dict[int, float | None]) -> None:
    _derived(conn, "relevance_bayes", 1, {i: s for i, s in scores.items() if s is not None})


def _derived(conn: sqlite3.Connection, scorer: str, version: int, scores: dict[int, float]) -> None:
    for exchange_id, score in scores.items():
        conn.execute(
            "INSERT INTO annotations (subject_kind, subject_id, scorer, scorer_version,"
            " reproducibility, score, recipe_json, created_at)"
            " VALUES ('exchange', ?, ?, ?, 'derived', ?, '{}', ?)",
            (exchange_id, scorer, version, score, AT.isoformat()),
        )


def test_the_uncertain_queue_ranks_by_the_named_scorers_latest_version(
    conn: sqlite3.Connection,
) -> None:
    _texty(conn, 2, 3)
    _gate(conn, {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5})
    _derived(conn, "local-model", 1, {1: 0.5, 2: 0.9, 3: 0.1})
    _derived(conn, "local-model", 2, {1: 0.95, 2: 0.6, 3: 0.45})
    _derived(conn, "other", 1, {4: 0.5})
    queue = uncertain_queue(3, frozenset(), "local-model")(conn)

    assert [i.exchange_id for i in queue] == [3, 2, 1]


def test_the_uncertain_queue_defaults_to_the_cascade_bayes_score(conn: sqlite3.Connection) -> None:
    _texty(conn, 2)
    conn.execute("UPDATE exchanges SET p_lore = 0.5 WHERE id = 1")
    _gate(conn, {1: 0.9, 2: 0.5})
    _derived(conn, "local-model", 1, {1: 0.5, 2: 0.9})

    assert [i.exchange_id for i in uncertain_queue(3, frozenset())(conn)] == [2, 1]


def test_the_uncertain_queue_drops_judged_exchanges_and_excluded_channels(
    conn: sqlite3.Connection,
) -> None:
    _texty(conn, 2, 3)
    _gate(conn, {1: 0.5, 2: 0.51, 3: 0.52})
    build = uncertain_queue(3, frozenset())
    first = build(conn)
    submit(conn, first, 0, first[0].exchange_id, RELEVANT, AT)
    conn.execute("UPDATE exchanges SET extraction_status = 'done' WHERE id = 3")

    assert [i.exchange_id for i in build(conn)] == [2]
    assert uncertain_judged(conn) == 1
    assert uncertain_queue(3, frozenset({"hardware"}))(conn) == []


def test_judging_in_the_uncertain_queue_never_marks_a_different_exchange_judged(
    conn: sqlite3.Connection,
) -> None:
    _texty(conn, 2, 3)
    _gate(conn, {1: 0.5, 2: 0.51, 3: 0.52})
    build = uncertain_queue(3, frozenset())
    first = build(conn)
    submit(conn, first, 0, first[0].exchange_id, RELEVANT, AT)

    after = build(conn)
    assert [i.exchange_id for i in after] == [2, 3]
    assert progress(conn, after) == (0, 2)
    assert first_unjudged(conn, after) == 0


def test_a_missing_channel_falls_back_to_its_id(tmp_path: Path) -> None:
    connection = open_database(tmp_path / "y.db")
    migrate(connection)
    connection.execute("PRAGMA foreign_keys = OFF")
    _exchange(connection, 1, [11])
    _slice(connection, GOLD, [1])

    assert exchange_view(connection, frozen_queue(connection), 0).channel == "1"


def test_an_exchange_with_no_messages_is_refused(tmp_path: Path) -> None:
    connection = open_database(tmp_path / "z.db")
    migrate(connection)
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (1, 1, 1, 1, ?, ?, 1, 'quiet_gap', 'h')",
        (AT.isoformat(), AT.isoformat()),
    )
    _slice(connection, GOLD, [1])

    with pytest.raises(NotInExchangeError, match="no messages"):
        submit(connection, frozen_queue(connection), 0, 1, RELEVANT, AT)


def _label_with(conn: sqlite3.Connection, exchange_id: int) -> None:
    conn.execute(
        "INSERT INTO annotations (subject_kind, subject_id, scorer, scorer_version,"
        " reproducibility, label, created_at)"
        " VALUES ('exchange', ?, ?, 2, 'recorded', 'relevant', ?)",
        (exchange_id, JUDGE_SCORER, AT.isoformat()),
    )


def test_the_c1_queue_is_the_control_exchanges_nobody_has_judged(
    conn: sqlite3.Connection,
) -> None:
    _slice(conn, "c1", [2, 4, 5])
    _texty(conn, 2, 4, 5)
    _label_with(conn, 4)

    assert _ids(c1_queue(frozenset())(conn)) == [("c1", 1, 2), ("c1", 3, 5)]


def _content(conn: sqlite3.Connection, exchange_id: int, text: str) -> None:
    conn.execute(
        "UPDATE messages SET content = ? WHERE id IN"
        " (SELECT message_id FROM exchange_messages WHERE exchange_id = ?)",
        (text, exchange_id),
    )


def test_likely_irrelevant_ranks_by_lexicon_share_then_bayes_score(
    conn: sqlite3.Connection,
) -> None:
    for eid in (6, 7, 8, 9):
        _exchange(conn, eid, [eid * 10 + 1])
    _texty(conn, 3, 4, 5, 6, 7, 8, 9)
    _slice(conn, "s2", [8])
    texts = {4: "great food", 5: "kernel panic", 6: "nothing", 9: "scsi disk", 7: "x", 8: "y"}
    for eid, text in texts.items():
        _content(conn, eid, text)
    _gate(conn, {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.2, 5: 0.1, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0})
    _label_with(conn, 7)

    queue = likely_irrelevant_queue(frozenset())(conn)

    assert [(i.slice_name, i.exchange_id) for i in queue] == [
        (LIKELY_IRRELEVANT, 6),
        (LIKELY_IRRELEVANT, 4),
        (LIKELY_IRRELEVANT, 9),
        (LIKELY_IRRELEVANT, 5),
    ]
    assert progress(conn, queue) == (0, 4)


def test_the_page_states_the_labelling_definition() -> None:
    from importlib import resources

    page = resources.files("infovore.eval").joinpath("templates/judge.html").read_text()
    page = " ".join(page.split())

    assert "tech, computers, SGI, IRIX or retrocomputing" in page
    assert "reusable SGI/IRIX" not in page
    assert JUDGE_INTERFACE_VERSION == 3


def test_usable_label_counts_skip_excluded_channels(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (2, 9, NULL, 'food', 'text')"
    )
    conn.execute("UPDATE exchanges SET channel_id = 2 WHERE id = 4")
    queue = frozen_queue(conn)
    _label(conn, queue, 0, RELEVANT)
    _label(conn, queue, next(i for i, q in enumerate(queue) if q.exchange_id == 4), IRRELEVANT)

    assert label_counts(conn)[IRRELEVANT] == 1
    assert label_counts(conn, frozenset({"food"}))[IRRELEVANT] == 0
    assert label_counts(conn, frozenset({"food"}))[RELEVANT] == 1


def test_queue_stats_count_judged_from_the_queue_across_rebuilds(
    conn: sqlite3.Connection,
) -> None:
    _texty(conn, 2, 3)
    _gate(conn, {1: 0.5, 2: 0.51, 3: 0.52})
    build = uncertain_queue(3, frozenset())
    first = build(conn)
    submit(conn, first, 0, first[0].exchange_id, IRRELEVANT, AT)

    after = build(conn)

    assert queue_stats(conn, UNCERTAIN, after) == {
        "queue": UNCERTAIN,
        "judged": 1,
        "queued": 2,
        "relevant_needed": 200,
        "irrelevant_needed": 200,
        "target": 200,
    }
    assert queue_stats(conn, "c1", after)["judged"] == 0


def test_c1_and_likely_irrelevant_queues_skip_excluded_channels(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (2, 9, NULL, 'Food', 'text')"
    )
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (3, 9, 2, 'pizza', 'thread')"
    )
    conn.execute("UPDATE exchanges SET channel_id = 2 WHERE id = 4")
    conn.execute("UPDATE exchanges SET channel_id = 3 WHERE id = 5")
    _exchange(conn, 6, [61])
    _texty(conn, 2, 4, 5, 6)
    _gate(conn, {i: 0.1 for i in range(1, 7)})
    _slice(conn, "c1", [2, 4, 5])
    excluded = frozenset({"food"})

    assert [i.exchange_id for i in c1_queue(excluded)(conn)] == [2]
    assert {i.exchange_id for i in likely_irrelevant_queue(excluded)(conn)} == {6}
    assert {i.exchange_id for i in likely_irrelevant_queue(frozenset())(conn)} == {4, 5, 6}


def _blank(conn: sqlite3.Connection, message_id: int, content: str) -> None:
    conn.execute("UPDATE messages SET content = ? WHERE id = ?", (content, message_id))


def test_dynamic_queues_skip_exchanges_with_fewer_than_three_text_messages(
    conn: sqlite3.Connection,
) -> None:
    assert MIN_TEXT_MESSAGES == 3
    _exchange(conn, 6, [61, 62, 63])
    _exchange(conn, 7, [71, 72, 73])
    _exchange(conn, 8, [81, 82, 83])
    _exchange(conn, 9, [91, 92, 93])
    _blank(conn, 62, "  \n\t ")
    _blank(conn, 72, "[redacted]")
    conn.execute("UPDATE messages SET deleted_at = ? WHERE id = 82", (AT.isoformat(),))
    _gate(conn, {i: 0.1 for i in range(1, 10)})
    conn.execute("UPDATE exchanges SET extraction_status = 'pending'")
    _slice(conn, "c1", [1, 2, 3, 6, 7, 8, 9])

    assert [i.exchange_id for i in c1_queue(frozenset())(conn)] == [1, 9]
    assert {i.exchange_id for i in likely_irrelevant_queue(frozenset())(conn)} == {9}
    assert [i.exchange_id for i in uncertain_queue(3, frozenset())(conn)] == [1, 9]


def test_the_likely_irrelevant_pool_is_drawn_after_the_fragment_filter(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("infovore.eval.judge.LIKELY_IRRELEVANT_POOL", 2)
    _exchange(conn, 6, [61, 62, 63])
    _gate(conn, {6: 0.5, 4: 0.0, 5: 0.0})

    assert [i.exchange_id for i in likely_irrelevant_queue(frozenset())(conn)] == [6]
