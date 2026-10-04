import json
import sqlite3
from pathlib import Path

from infovore.cli import ExitCode
from infovore.db.annotations import Annotation, record_annotation
from infovore.db.connection import open_database
from tests.triage.test_cascade import NOW, build
from tests.triage.test_command import run


def derived(
    conn: sqlite3.Connection, eid: int, stage: str, label: str | None, version: int = 1
) -> None:
    record_annotation(
        conn,
        Annotation(
            "exchange",
            eid,
            f"relevance_{stage}",
            version,
            "derived",
            score=0.5,
            label=label,
            recipe={"t": 1},
        ),
        NOW,
    )


def judged(conn: sqlite3.Connection, eid: int, label: str, ref: str | None) -> None:
    record_annotation(
        conn,
        Annotation("exchange", eid, "human_exchange", 2, "recorded", label=label, source_ref=ref),
        NOW,
    )


def prepare(tmp_path: Path) -> dict[str, str]:
    env, conn = build(tmp_path)
    conn.execute("INSERT INTO channels (id, guild_id, name, kind) VALUES (1, 9, 'general', 'text')")
    conn.execute("INSERT INTO channels (id, guild_id, name, kind) VALUES (2, 9, 'lounge', 'text')")
    conn.execute("INSERT INTO channels (id, guild_id, name, kind) VALUES (3, 9, 'side', 'thread')")
    conn.execute("UPDATE channels SET parent_id = 1 WHERE id = 3")
    conn.execute("UPDATE exchanges SET channel_id = 2 WHERE id = 4")
    conn.execute("UPDATE exchanges SET channel_id = 3 WHERE id = 6")
    derived(conn, 1, "lexicon", "relevant")
    derived(conn, 2, "lexicon", None)
    derived(conn, 2, "embed", "irrelevant")
    derived(conn, 3, "lexicon", None)
    derived(conn, 3, "residue", "residue")
    derived(conn, 7, "lexicon", "relevant")
    derived(conn, 7, "lexicon", None, 2)
    derived(conn, 7, "short_no_tech", "irrelevant")
    judged(conn, 2, "irrelevant", "judge:likely-irrelevant:4")
    judged(conn, 3, "bad_grouping", "judge:s1:3")
    judged(conn, 5, "irrelevant", "rechunk:12")
    judged(conn, 6, "relevant", "judge:gold:1")
    conn.commit()
    conn.close()
    return {**env, "INFOVORE_EXCLUDE_CHANNELS": "lounge"}


def export(env: dict[str, str], tmp_path: Path, *extra: str) -> tuple[list[dict[str, object]], str]:
    path = tmp_path / "out" / "labels.jsonl"
    code, out, _ = run(["labels", "export", "--jsonl", str(path), *extra], env)
    assert code == ExitCode.OK
    return [json.loads(line) for line in path.read_text().splitlines()], out


def test_export_rows_carry_label_slice_stage_and_source(tmp_path: Path) -> None:
    rows, _ = export(prepare(tmp_path), tmp_path)
    by_id = {int(str(r["id"])): r for r in rows}

    assert [r["id"] for r in rows] == [1, 2, 5, 6, 7]
    assert by_id[2] == {
        "id": 2,
        "text": "a: scsi disk boot again",
        "label": "irrelevant",
        "held_out": False,
        "slice": "s1",
        "channel": "general",
        "cascade_stage": "embed",
        "label_source": "likely-irrelevant",
    }
    assert by_id[1]["held_out"] is True and by_id[1]["slice"] == "gold"
    assert by_id[1]["cascade_stage"] == "lexicon" and by_id[1]["label_source"] is None
    assert by_id[5]["slice"] == "s1" and by_id[5]["cascade_stage"] is None
    assert by_id[5]["label_source"] == "rechunk"
    assert by_id[6]["label_source"] == "gold" and by_id[6]["label"] == "relevant"
    assert by_id[6]["channel"] == "general"
    assert by_id[7]["cascade_stage"] == "short_no_tech"


def test_text_is_the_full_llm_score_render(tmp_path: Path) -> None:
    rows, _ = export(prepare(tmp_path), tmp_path)

    assert next(r for r in rows if r["id"] == 7)["text"] == "a: scsi disk\na: lol lunch"


def test_excluded_channels_are_omitted_unless_included(tmp_path: Path) -> None:
    env = prepare(tmp_path)
    rows, _ = export(env, tmp_path, "--include-excluded-channels")
    four = next(r for r in rows if r["id"] == 4)

    assert [r["id"] for r in rows] == [1, 2, 4, 5, 6, 7]
    assert four["channel"] == "lounge"


def test_counts_are_printed(tmp_path: Path) -> None:
    _, out = export(prepare(tmp_path), tmp_path)

    assert "labels: 5" in out
    assert "label relevant: 2" in out and "label irrelevant: 3" in out
    assert "held_out true: 2" in out and "held_out false: 3" in out
    assert "cascade_stage lexicon: 1" in out and "cascade_stage embed: 1" in out
    assert "cascade_stage none: 2" in out


def test_undecided_pile_labels_are_exported_with_their_source(tmp_path: Path) -> None:
    env = prepare(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    judged(conn, 3, "relevant", "judge:undecided:3")
    conn.commit()
    conn.close()

    rows, _ = export(env, tmp_path)
    by_id = {int(str(r["id"])): r for r in rows}

    assert by_id[3]["label_source"] == "undecided" and by_id[3]["label"] == "relevant"
