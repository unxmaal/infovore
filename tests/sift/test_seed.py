import io
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.cli import main
from infovore.db.connection import migrate, open_database
from infovore.db.message_labels import (
    effective_message_labels,
    effective_message_labels_with_source,
    set_message_label,
)
from infovore.rows import LabelRegime, MessageLabel, MessageLabelSource
from infovore.sift.sampling import SiftStrategy
from infovore.sift.seed import (
    BUCKET_EDGES,
    REPORT_FLOORS,
    SEED_FLOOR,
    SEED_SOURCE_REF_PREFIX,
    SEED_TOP,
    SeedRow,
    format_rows,
    percentile,
    render_report,
    seed_rows,
    top_rows,
    write_seed_queue,
)
from infovore.sift.serve import ServeApp, build_serve_app
from infovore.timing import FixedClock
from infovore.triage.human import load_gazetteer
from infovore.triage.lexicon import Lexicon

NOW = datetime(2026, 1, 1, tzinfo=UTC)
STAMP = NOW.isoformat()
LEXICON = Lexicon("t", frozenset({"irix", "mips"}), load_gazetteer(), {})
FOOD = frozenset({"food"})

MESSAGES = [
    (10, 1, 1, "irix mips box runs well", 0, None),
    (11, 1, 1, "irix mips box runs well", 1, None),
    (12, 1, 1, "irix mips box runs well", 0, STAMP),
    (13, 1, 2, "irix mips box runs well", 0, None),
    (14, 2, 1, "irix mips box runs well", 0, None),
    (15, 3, 1, "irix mips box runs well", 0, None),
    (16, 4, 1, "irix box runs well", 0, None),
    (18, 1, 1, "irix mips fast disk", 0, None),
    (19, 1, 1, "just some plain words", 0, None),
    (20, 1, 1, "mips irix hot disk", 0, None),
    (23, 1, 1, "irix mips one two three four five six seven eight nine ten", 0, None),
    (24, 1, 1, "irix mips", 0, None),
]


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "secret-token",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1,2",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_SCRATCH_DIR": str(tmp_path / "scratch"),
        "INFOVORE_EXCLUDE_CHANNELS": "food",
    }


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def build(
    path: str | Path, messages: list[tuple[int, int, int, str, int, str | None]]
) -> sqlite3.Connection:
    conn = open_database(str(path))
    migrate(conn)
    for channel_id, name, kind, parent in [
        (1, "general", "text", None),
        (2, "food", "text", None),
        (3, "pizza", "thread", 2),
        (4, "chat", "thread", 1),
    ]:
        conn.execute(
            "INSERT INTO channels (id, guild_id, name, kind, parent_id) VALUES (?, 1, ?, ?, ?)",
            (channel_id, name, kind, parent),
        )
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (2, ?)", (STAMP,))
    for message_id, channel_id, author_id, content, bot, deleted in messages:
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json, author_is_bot, deleted_at)"
            " VALUES (?, ?, 1, ?, 'alice', ?, ?, ?, '{}', ?, ?)",
            (message_id, channel_id, author_id, STAMP, content, STAMP, bot, deleted),
        )
        conn.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
            " started_at, ended_at, message_count, grouping_rule, content_hash)"
            " VALUES (?, ?, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
            (message_id, channel_id, message_id, message_id, STAMP, STAMP, f"h{message_id}"),
        )
        conn.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
            (message_id, message_id),
        )
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (17, 1, 1, 1, 'alice', ?, 'irix mips box runs well', ?, '{}')",
        (STAMP, STAMP),
    )
    conn.commit()
    return conn


def test_seed_rows_exclusions_and_ranking(tmp_path: Path) -> None:
    conn = build(tmp_path / "t.db", MESSAGES)
    rows = seed_rows(conn, LEXICON, FOOD, 3)
    assert [r.id for r in rows] == [18, 20, 10, 23, 16, 19]
    assert [(r.words, r.hits) for r in rows] == [(4, 3), (4, 3), (5, 3), (12, 3), (4, 1), (4, 0)]
    assert rows[0].score == 75.0
    assert rows[0].channel == "general"
    assert rows[-1].score == 0.0


def test_score_tie_breaks_on_hits_then_id(tmp_path: Path) -> None:
    conn = build(tmp_path / "t.db", MESSAGES)
    by_score = [r for r in seed_rows(conn, LEXICON, FOOD, 3) if r.score == 25.0]
    assert [r.id for r in by_score] == [23, 16]


def test_floor_filters_by_word_count(tmp_path: Path) -> None:
    conn = build(tmp_path / "t.db", MESSAGES)
    assert [r.id for r in seed_rows(conn, LEXICON, FOOD, 5)] == [10, 23]
    assert 24 in [r.id for r in seed_rows(conn, LEXICON, FOOD, 2)]


def test_without_denylist_food_threads_return(tmp_path: Path) -> None:
    conn = build(tmp_path / "t.db", MESSAGES)
    ids = {r.id for r in seed_rows(conn, LEXICON, frozenset(), 3)}
    assert {14, 15} <= ids
    assert not ids & {11, 12, 13, 17}


def test_top_rows_and_percentile() -> None:
    rows = [SeedRow(i, "c", 5, 1, 20.0, "t") for i in range(5)]
    assert top_rows(rows, 2) == rows[:2]
    assert percentile([], 50) == 0
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 99) == 4


def test_render_report(tmp_path: Path) -> None:
    conn = build(tmp_path / "t.db", MESSAGES)
    text = render_report(conn, LEXICON, FOOD, SEED_FLOOR, 2)
    assert "candidates: 7" in text
    assert "p50:" in text and "p99:" in text
    for edge in BUCKET_EDGES:
        assert f"<= {edge}:" in text
    assert f"> {BUCKET_EDGES[-1]}: 0" in text
    for floor in REPORT_FLOORS:
        assert f"\n{floor:>5} " in text
    assert f"floor used: {SEED_FLOOR}" in text


def test_render_report_with_no_candidates(tmp_path: Path) -> None:
    conn = build(tmp_path / "t.db", [])
    text = render_report(conn, LEXICON, FOOD, SEED_FLOOR, 50)
    assert "candidates: 0" in text
    assert "p50: 0" in text
    assert "0.0" in text


def test_write_seed_queue_and_format_rows(tmp_path: Path) -> None:
    rows = [SeedRow(7, "general", 5, 2, 40.0, "line one\nline two " + "x" * 100)]
    ref = write_seed_queue(rows, tmp_path / "q", NOW)
    assert ref == SEED_SOURCE_REF_PREFIX + "2026-01-01"
    manifest = json.loads((tmp_path / "q" / "manifest.json").read_text())
    assert manifest["message_ids"] == [7]
    assert manifest["strategy"] == "seed"
    assert manifest["size"] == 1
    assert manifest["source_ref"] == ref
    header, line = format_rows(rows).splitlines()
    assert header == "id\tchannel\twords\thits\tscore\ttext"
    fields = line.split("\t")
    assert fields[:5] == ["7", "general", "5", "2", "40.0"]
    assert fields[5].startswith("line one line two ") and len(fields[5]) == 80
    assert SEED_TOP == 50


def test_cli_ranked_listing_report_and_floor_validation(tmp_path: Path) -> None:
    build(tmp_path / "infovore.db", MESSAGES).close()
    env = environment(tmp_path)
    code, out, _ = run(["sift", "seed", "--floor", "3", "--top", "2"], env)
    assert code == 0
    assert out.splitlines()[0].startswith("id\tchannel")
    assert len(out.splitlines()) == 3
    code, out, _ = run(["sift", "seed", "--report"], env)
    assert code == 0 and "candidates:" in out
    code, _, err = run(["sift", "seed", "--floor", "0"], env)
    assert code != 0 and "at least 1" in err


def test_cli_out_then_serve_labels_carry_seed_source_ref(tmp_path: Path) -> None:
    build(tmp_path / "infovore.db", MESSAGES).close()
    env = environment(tmp_path)
    queue = tmp_path / "queue"
    code, out, _ = run(["sift", "seed", "--floor", "3", "--top", "3", "--out", str(queue)], env)
    assert code == 0
    assert f"serve with: infovore sift serve {queue}" in out
    manifest = json.loads((queue / "manifest.json").read_text())
    assert len(manifest["message_ids"]) == 3

    conn = open_database(env["INFOVORE_DB_PATH"])
    clock = FixedClock(NOW)
    app = build_serve_app(
        conn,
        dir_=queue,
        new=False,
        size=50,
        strategy=SiftStrategy.RANDOM,
        seed=0,
        mix=0.5,
        out_dir=None,
        scratch_dir=tmp_path / "scratch",
        clock=clock,
    )
    assert app.source_ref == manifest["source_ref"]
    assert {m.id for m in app.messages()} == set(manifest["message_ids"])
    app.label(manifest["message_ids"][0], MessageLabel.KEEP)
    ref = conn.execute("SELECT source_ref FROM message_labels").fetchone()[0]
    assert ref.startswith(SEED_SOURCE_REF_PREFIX)
    assert app.progress().labeled == 1


def test_plain_serve_app_keeps_default_source_ref(tmp_path: Path) -> None:
    conn = build(tmp_path / "t.db", [])
    app = ServeApp(conn, [], "batch9", tmp_path / "scratch", FixedClock(NOW))
    assert app.source_ref == "sift-serve:batch9"


def test_seed_rows_skip_human_labeled(tmp_path: Path) -> None:
    conn = build(tmp_path / "infovore.db", MESSAGES)
    set_message_label(conn, 10, MessageLabel.KEEP, MessageLabelSource.HUMAN, "x", NOW)
    assert 10 not in [r.id for r in seed_rows(conn, LEXICON, FOOD, 3)]


def test_value_regime_labels_stay_out_of_training(tmp_path: Path) -> None:
    build(tmp_path / "infovore.db", MESSAGES).close()
    env = environment(tmp_path)
    queue = tmp_path / "q"
    run(["sift", "seed", "--floor", "3", "--top", "3", "--out", str(queue)], env)
    manifest = json.loads((queue / "manifest.json").read_text())
    assert manifest["regime"] == "value"
    conn = open_database(env["INFOVORE_DB_PATH"])
    app = build_serve_app(
        conn,
        dir_=queue,
        new=False,
        size=50,
        strategy=SiftStrategy.RANDOM,
        seed=0,
        mix=0.5,
        out_dir=None,
        scratch_dir=tmp_path / "scratch",
        clock=FixedClock(NOW),
    )
    mid = manifest["message_ids"][0]
    app.label(mid, MessageLabel.TRASH)
    sql = "SELECT regime FROM {} WHERE message_id = ?"
    assert conn.execute(sql.format("message_labels"), (mid,)).fetchone()[0] == "value"
    assert conn.execute(sql.format("label_events"), (mid,)).fetchone()[0] == "value"
    assert mid not in effective_message_labels_with_source(conn)
    assert mid not in effective_message_labels(conn)
    assert mid in effective_message_labels_with_source(conn, frozenset({LabelRegime.VALUE}))
