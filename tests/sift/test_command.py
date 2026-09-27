import io
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.cli import ExitCode, builtin_commands, main
from infovore.db.connection import migrate, open_database
from infovore.db.message_labels import set_message_label
from infovore.rows import MessageLabel, MessageLabelSource

NOW = datetime(2026, 1, 1, tzinfo=UTC).isoformat()


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "secret-token",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1,2",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_SCRATCH_DIR": str(tmp_path / "scratch"),
    }


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def _channel(conn: sqlite3.Connection, channel_id: int, name: str) -> None:
    conn.execute(
        "INSERT INTO channels (id, guild_id, name, kind) VALUES (?, 1, ?, 'text')",
        (channel_id, name),
    )


def _message_with_exchange(
    conn: sqlite3.Connection,
    message_id: int,
    channel_id: int,
    author_id: int = 1,
    author_name: str = "alice",
    content: str = "hello there",
    p_trash: float | None = None,
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json, p_trash)"
        " VALUES (?, ?, 1, ?, ?, ?, ?, ?, '{}', ?)",
        (message_id, channel_id, author_id, author_name, NOW, content, NOW, p_trash),
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


def seed(db_path: str) -> None:
    conn = open_database(db_path)
    migrate(conn)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    for i in range(1, 4):
        _message_with_exchange(conn, i, 1, content=f"general chatter number {i}")
    for i in range(4, 7):
        _message_with_exchange(conn, i, 2, content=f"food talk number {i}")
    conn.close()


def test_sift_is_a_builtin_command() -> None:
    assert "sift" in [command.name for command in builtin_commands()]


def test_sift_export_requires_out(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    code, _, err = run(["sift", "export", "--size", "2"], env)
    assert code == ExitCode.CONFIG
    assert "--out" in err


def test_sift_export_writes_batch_files(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch1"

    code, out, _ = run(["sift", "export", "--size", "4", "--seed", "1", "--out", str(out_dir)], env)

    assert code == ExitCode.OK
    assert (out_dir / "batch.log").exists()
    assert (out_dir / "infovore-sift.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert "lnav -i" in out
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest["message_ids"]) == 4
    assert manifest["strategy"] == "random"
    assert manifest["seed"] == 1


def test_sift_export_excludes_opted_out_authors(tmp_path: Path) -> None:
    env = environment(tmp_path)
    db_path = env["INFOVORE_DB_PATH"]
    conn = open_database(db_path)
    migrate(conn)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1, author_id=1)
    _message_with_exchange(conn, 2, 1, author_id=2)
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (2, ?)", (NOW,))
    conn.close()

    out_dir = tmp_path / "batch"
    code, _, _ = run(["sift", "export", "--size", "5", "--out", str(out_dir)], env)

    assert code == ExitCode.OK
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["message_ids"] == [1]


def test_sift_export_excludes_already_human_labeled_messages(tmp_path: Path) -> None:
    env = environment(tmp_path)
    db_path = env["INFOVORE_DB_PATH"]
    seed(db_path)
    conn = open_database(db_path)
    set_message_label(
        conn,
        1,
        MessageLabel.TRASH,
        MessageLabelSource.HUMAN,
        None,
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    conn.close()

    out_dir = tmp_path / "batch"
    code, _, _ = run(["sift", "export", "--size", "6", "--out", str(out_dir)], env)

    assert code == ExitCode.OK
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert 1 not in manifest["message_ids"]


def test_sift_export_rejects_mix_outside_zero_one(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"
    code, _, err = run(
        ["sift", "export", "--size", "2", "--mix", "1.5", "--out", str(out_dir)], env
    )
    assert code == ExitCode.CONFIG
    assert "--mix" in err


def test_sift_export_uncertain_without_a_trained_model_exits_config(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"

    code, _, err = run(
        ["sift", "export", "--size", "2", "--strategy", "uncertain", "--out", str(out_dir)],
        env,
    )

    assert code == ExitCode.CONFIG
    assert "p_trash" in err
    assert not out_dir.exists()


def test_sift_export_mixed_without_a_trained_model_falls_back_to_random(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"

    code, out, _ = run(
        ["sift", "export", "--size", "2", "--strategy", "mixed", "--out", str(out_dir)],
        env,
    )

    assert code == ExitCode.OK
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest["message_ids"]) == 2
    assert "wrote 2" in out


def test_sift_import_records_labels_and_prints_counts(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"
    export_code, _, _ = run(
        ["sift", "export", "--size", "6", "--out", str(out_dir)], env
    )
    assert export_code == ExitCode.OK
    (out_dir / "kept.csv").write_text("msg\n4\n5\n6\n")

    code, out, _ = run(["sift", "import", str(out_dir)], env)

    assert code == ExitCode.OK
    assert "keep=3" in out
    assert "trash=3" in out
    assert "#general: keep=0 trash=3" in out
    assert "#food: keep=3 trash=0" in out

    conn = open_database(env["INFOVORE_DB_PATH"])
    rows = conn.execute(
        "SELECT message_id, label FROM message_labels WHERE source = 'human' ORDER BY message_id"
    ).fetchall()
    assert [(r["message_id"], r["label"]) for r in rows] == [
        (1, "trash"),
        (2, "trash"),
        (3, "trash"),
        (4, "keep"),
        (5, "keep"),
        (6, "keep"),
    ]


def test_sift_import_missing_manifest_exits_config(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    code, _, err = run(["sift", "import", str(empty_dir)], env)

    assert code == ExitCode.CONFIG
    assert "manifest.json" in err


def test_sift_import_with_no_result_files_exits_config_with_lnav_help(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"
    run(["sift", "export", "--size", "6", "--out", str(out_dir)], env)

    code, _, err = run(["sift", "import", str(out_dir)], env)

    assert code == ExitCode.CONFIG
    assert "kept.csv" in err
    assert "trash-regexes.csv" in err
    assert "write-csv-to" in err


def test_sift_import_save_rules_requires_a_regex_file(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"
    run(["sift", "export", "--size", "6", "--out", str(out_dir)], env)
    (out_dir / "kept.csv").write_text("msg\n4\n5\n6\n")

    code, _, err = run(
        ["sift", "import", str(out_dir), "--save-rules", "food-chatter"], env
    )

    assert code == ExitCode.CONFIG
    assert "--save-rules" in err


def test_sift_import_save_rules_writes_a_file(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"
    run(["sift", "export", "--size", "6", "--out", str(out_dir)], env)
    (out_dir / "trash-regexes.csv").write_text("pattern\ngeneral chatter\n")

    code, out, _ = run(
        ["sift", "import", str(out_dir), "--save-rules", "food-chatter"], env
    )

    assert code == ExitCode.OK
    assert "food-chatter" in out
    saved = tmp_path / "scratch" / "sift_trash_rules" / "food-chatter.json"
    assert saved.exists()
