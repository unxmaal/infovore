import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.chunk.grouper import group_pending
from infovore.chunk.rechunk import UnknownRecipeError, apply_rechunk, plan_rechunk, render_plan
from infovore.chunk.recipe import ChunkRecipe
from infovore.cli import ExitCode
from infovore.db.annotations import Annotation, record_annotation
from infovore.db.batch import exchange_inputs_for_ids
from infovore.db.chunk_recipes import register_recipe
from infovore.db.connection import migrate, open_database
from infovore.eval.slices import slice_ids, slice_names, slice_summary
from infovore.timing import FixedClock
from infovore.triage.human import held_out_ids, training_labels
from tests.chunk.test_command import environment, run

BASE = datetime(2026, 1, 1, tzinfo=UTC)
NOW = BASE + timedelta(days=30)
GAP = timedelta(minutes=30)


def msg(
    conn: sqlite3.Connection,
    id: int,
    minute: int,
    channel: int = 1,
    reply_to: int | None = None,
    content: str = "hello there",
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO channels (id, guild_id, name, kind) VALUES (?, 9, ?, 'text')",
        (channel, f"chan{channel}"),
    )
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json, reply_to_id)"
        " VALUES (?, ?, 9, 1, 'a', ?, ?, ?, '{}', ?)",
        (
            id,
            channel,
            (BASE + timedelta(minutes=minute)).isoformat(),
            content,
            BASE.isoformat(),
            reply_to,
        ),
    )


def exchange_of(conn: sqlite3.Connection, message_id: int) -> int:
    row = conn.execute(
        "SELECT exchange_id FROM all_exchange_messages WHERE message_id = ?"
        " ORDER BY exchange_id LIMIT 1",
        (message_id,),
    ).fetchone()
    return int(row["exchange_id"])


def current_exchange_of(conn: sqlite3.Connection, message_id: int) -> int:
    row = conn.execute(
        "SELECT exchange_id FROM exchange_messages WHERE message_id = ?", (message_id,)
    ).fetchone()
    return int(row["exchange_id"])


def label(
    conn: sqlite3.Connection, exchange_id: int, value: str, source: str | None = None
) -> None:
    record_annotation(
        conn,
        Annotation(
            subject_kind="exchange",
            subject_id=exchange_id,
            scorer="human_exchange",
            scorer_version=1,
            reproducibility="recorded",
            label=value,
            source_ref=source,
        ),
        BASE,
    )


def fixed(conn: sqlite3.Connection, minutes: int, fold: float = 0) -> int:
    recipe = ChunkRecipe(
        quiet_gap=timedelta(minutes=minutes),
        max_messages=50,
        overlap=3,
        fold_factor=fold,
        fold_size=2 if fold else 0,
    )
    return register_recipe(conn, recipe, BASE)


def freeze(conn: sqlite3.Connection, name: str, ids: list[int]) -> None:
    for position, exchange_id in enumerate(ids, start=1):
        conn.execute(
            "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
            " VALUES (?, ?, ?, 'queue', 190, ?)",
            (name, exchange_id, position, BASE.isoformat()),
        )


def count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "r.db")
    migrate(connection)
    for id, minute in [(1, 0), (2, 1), (3, 2), (4, 50), (5, 51), (6, 300)]:
        msg(connection, id, minute)
    for id, minute in [(7, 500), (8, 501), (9, 502), (10, 503), (11, 550)]:
        msg(connection, id, minute)
    group_pending(connection, FixedClock(NOW), quiet_gap=GAP)
    return connection


def test_a_rechunk_plans_reuse_merges_and_label_conflicts(conn: sqlite3.Connection) -> None:
    e1, e2, e3, e4, e5 = (exchange_of(conn, m) for m in (1, 4, 6, 7, 11))
    label(conn, e1, "relevant")
    label(conn, e2, "irrelevant")
    label(conn, e3, "relevant")
    label(conn, e4, "relevant")
    label(conn, e5, "relevant")
    freeze(conn, "s1", [e1, e2, e3, e4, e5])
    version = fixed(conn, 60, fold=2)

    plan = plan_rechunk(conn, version, False)

    assert [len(t.message_ids) for t in plan.targets] == [5, 1, 5]
    assert plan.reused == 1
    assert plan.label_kinds["conflict"] == 2
    assert plan.label_kinds["one_to_one"] == 1
    assert plan.label_kinds["merged"] == 2
    assert plan.labels_before == (4, 1)
    assert plan.labels_after == (2, 0)
    assert [r.kind for r in plan.remaps] == ["merged", "merged", "one_to_one", "merged", "merged"]
    assert plan.slice_before == {f"s1@{version}": 5}
    assert len(plan.slices[f"s1@{version}"]) == 3


def test_a_dry_run_changes_nothing(conn: sqlite3.Connection) -> None:
    version = fixed(conn, 60, fold=2)
    tables = ("exchanges", "exchange_messages", "exchange_remap", "annotations")
    before = [count(conn, t) for t in tables]
    text = render_plan(plan_rechunk(conn, version, False), dry_run=True)
    assert before == [count(conn, t) for t in tables]
    assert "dry run" in text
    assert "reused unchanged: 1" in text


def test_applying_supersedes_old_rows_and_keeps_them_reachable(conn: sqlite3.Connection) -> None:
    e1, e3, e4 = (exchange_of(conn, m) for m in (1, 6, 7))
    conn.execute("UPDATE exchanges SET extraction_status = 'done' WHERE id = ?", (e4,))
    version = fixed(conn, 60, fold=2)
    plan = plan_rechunk(conn, version, False)

    apply_rechunk(conn, plan, NOW)

    assert plan.done_superseded == 1
    old = conn.execute("SELECT * FROM exchanges WHERE id = ?", (e1,)).fetchone()
    assert old["superseded_by_recipe"] == version
    assert old["extraction_status"] == "skipped"
    done = conn.execute("SELECT extraction_status FROM exchanges WHERE id = ?", (e4,)).fetchone()
    assert done[0] == "done"
    kept = conn.execute("SELECT * FROM exchanges WHERE id = ?", (e3,)).fetchone()
    assert kept["superseded_by_recipe"] is None
    assert kept["chunk_recipe"] == version
    live = conn.execute("SELECT COUNT(*) FROM exchange_messages WHERE exchange_id = ?", (e1,))
    assert live.fetchone()[0] == 0
    kept_rows = conn.execute(
        "SELECT COUNT(*) FROM all_exchange_messages WHERE exchange_id = ?", (e1,)
    )
    assert kept_rows.fetchone()[0] == 3
    new = current_exchange_of(conn, 1)
    assert new != e1
    size = conn.execute("SELECT message_count FROM exchanges WHERE id = ?", (new,)).fetchone()
    assert size[0] == 5
    assert current_exchange_of(conn, 4) == new
    remap = conn.execute("SELECT * FROM exchange_remap WHERE old_exchange_id = ?", (e1,)).fetchone()
    assert (remap["new_exchange_id"], remap["kind"], remap["shared_messages"]) == (new, "merged", 3)
    assert conn.execute("SELECT gap_seconds FROM channel_chunk_gaps").fetchone()[0] == 3600
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE exchange_remap SET kind = 'split'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM exchange_remap")


def test_applying_maps_labels_and_slices_without_touching_recorded_rows(
    conn: sqlite3.Connection,
) -> None:
    e1, e2, e3, e4, e5 = (exchange_of(conn, m) for m in (1, 4, 6, 7, 11))
    label(conn, e1, "relevant", "judge:s1:1")
    label(conn, e2, "irrelevant")
    label(conn, e3, "relevant")
    label(conn, e4, "relevant")
    label(conn, e5, "relevant")
    freeze(conn, "s1", [e1, e2, e3, e4, e5])
    freeze(conn, "s2", [e3])
    version = fixed(conn, 60, fold=2)
    apply_rechunk(conn, plan_rechunk(conn, version, False), NOW)

    new_ab, new_d = current_exchange_of(conn, 1), current_exchange_of(conn, 7)
    copied = conn.execute(
        "SELECT subject_id, label, source_ref FROM annotations WHERE source_ref LIKE 'rechunk:%'"
        " ORDER BY id"
    ).fetchall()
    assert [(r["subject_id"], r["label"]) for r in copied] == [(new_d, "relevant")] * 2
    assert copied[0]["source_ref"] == f"rechunk:{version}:{e4}"
    assert count(conn, "annotations") == 7
    labels, _ = training_labels(conn)
    assert {eid: label.value for eid, label in labels.items()} == {e3: "lore", new_d: "lore"}
    assert new_ab not in labels
    assert slice_ids(conn, "s1") == [new_ab, e3, new_d]
    assert slice_ids(conn, "s2") == [e3]
    frozen = conn.execute("SELECT COUNT(*) FROM eval_slices WHERE name = 's1'").fetchone()
    assert frozen[0] == 5
    assert slice_names(conn) == ["s1", "s2"]
    assert len(exchange_inputs_for_ids(conn, [e1])[e1].messages) == 3
    assert sum(b.exchanges for b in slice_summary(conn, "s1")) == 3
    assert held_out_ids(conn) == frozenset({e3})


def test_the_second_rechunk_has_nothing_left_to_do(conn: sqlite3.Connection) -> None:
    freeze(conn, "s1", [exchange_of(conn, 1)])
    version = fixed(conn, 60, fold=2)
    apply_rechunk(conn, plan_rechunk(conn, version, False), NOW)
    again = plan_rechunk(conn, version, False)
    assert again.remaps == []
    assert again.targets == []
    assert again.slices == {}
    apply_rechunk(conn, again, NOW)


def test_a_smaller_gap_splits_by_majority_and_flags_ambiguity(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "s.db")
    migrate(conn)
    for id, minute in [(30, 0), (31, 1), (32, 2), (33, 20)]:
        msg(conn, id, minute)
    for id, minute in [(40, 0), (41, 1), (42, 20), (43, 21)]:
        msg(conn, id, minute, channel=2)
    group_pending(conn, FixedClock(NOW), quiet_gap=GAP)
    split, ambiguous = exchange_of(conn, 30), exchange_of(conn, 40)
    label(conn, split, "irrelevant")
    label(conn, ambiguous, "relevant")
    freeze(conn, "s1", [split, ambiguous])
    version = fixed(conn, 5)

    plan = plan_rechunk(conn, version, False)
    apply_rechunk(conn, plan, NOW)

    assert [r.kind for r in plan.remaps] == ["split", "ambiguous"]
    assert plan.label_kinds["split"] == 1
    assert plan.label_kinds["ambiguous"] == 1
    assert plan.labels_after == (0, 1)
    assert slice_ids(conn, "s1") == [current_exchange_of(conn, 30)]
    row = conn.execute(
        "SELECT new_exchange_id FROM exchange_remap WHERE old_exchange_id = ?", (ambiguous,)
    ).fetchone()
    assert row["new_exchange_id"] is None
    text = render_plan(plan, dry_run=False)
    assert "applied" in text
    assert "slice s1@" in text


def test_a_skip_is_not_a_label_but_its_row_is_still_carried(conn: sqlite3.Connection) -> None:
    e1 = exchange_of(conn, 1)
    label(conn, e1, "relevant")
    label(conn, e1, "bad_grouping")
    version = fixed(conn, 60, fold=2)
    plan = plan_rechunk(conn, version, False)
    assert plan.labels_before == (0, 0)
    assert plan.label_kinds["merged"] == 1
    apply_rechunk(conn, plan, NOW)
    carried = conn.execute("SELECT COUNT(*) FROM annotations WHERE source_ref IS NOT NULL")
    assert carried.fetchone()[0] == 2


def test_an_unknown_recipe_is_refused(conn: sqlite3.Connection) -> None:
    with pytest.raises(UnknownRecipeError):
        plan_rechunk(conn, 99, False)


def test_the_migrated_adaptive_recipe_rechunks_with_a_derived_gap(conn: sqlite3.Connection) -> None:
    plan = plan_rechunk(conn, 2, False)
    assert plan.gaps[1] >= GAP
    apply_rechunk(conn, plan, NOW)
    recorded = conn.execute("SELECT gap_seconds FROM channel_chunk_gaps WHERE recipe = 2")
    assert recorded.fetchone()[0] == int(plan.gaps[1].total_seconds())


def command_env(tmp_path: Path) -> dict[str, str]:
    env = environment(tmp_path)
    connection = open_database(env["INFOVORE_DB_PATH"])
    migrate(connection)
    for id, minute in [(1, 0), (2, 1), (3, 100)]:
        msg(connection, id, minute)
    group_pending(connection, FixedClock(NOW), quiet_gap=GAP)
    connection.close()
    return env


def test_the_command_dry_runs_then_applies(tmp_path: Path) -> None:
    env = command_env(tmp_path)
    code, out, _ = run(["chunk", "--rechunk", "--recipe", "2", "--dry-run"], env)
    assert code == ExitCode.OK
    assert "dry run" in out
    check = open_database(env["INFOVORE_DB_PATH"])
    assert count(check, "exchange_remap") == 0
    code, out, _ = run(["chunk", "--rechunk", "--recipe", "2"], env)
    assert code == ExitCode.OK
    assert "applied" in out
    assert count(check, "exchange_remap") == 2


def test_the_command_needs_a_recipe(tmp_path: Path) -> None:
    code, _, err = run(["chunk", "--rechunk"], environment(tmp_path))
    assert code == ExitCode.CONFIG
    assert "--recipe" in err


def test_the_command_refuses_an_unknown_recipe(tmp_path: Path) -> None:
    code, _, err = run(["chunk", "--rechunk", "--recipe", "77"], environment(tmp_path))
    assert code == ExitCode.CONFIG
    assert "77" in err
