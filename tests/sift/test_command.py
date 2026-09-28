import io
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

import infovore.sift.command as sift_command
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


def test_sift_export_channels_restricts_the_pool(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"

    code, _, _ = run(
        ["sift", "export", "--size", "10", "--channels", "general", "--out", str(out_dir)], env
    )

    assert code == ExitCode.OK
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert sorted(manifest["message_ids"]) == [1, 2, 3]


def test_sift_export_channels_accepts_hash_prefix_and_mixed_case(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"

    code, _, _ = run(
        ["sift", "export", "--size", "10", "--channels", "#General", "--out", str(out_dir)], env
    )

    assert code == ExitCode.OK
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert sorted(manifest["message_ids"]) == [1, 2, 3]


def test_sift_export_unknown_channel_exits_config_listing_known_channels(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"

    code, _, err = run(
        ["sift", "export", "--size", "10", "--channels", "nope", "--out", str(out_dir)], env
    )

    assert code == ExitCode.CONFIG
    assert "nope" in err
    assert "general" in err
    assert "food" in err


def test_sift_export_denylist_wins_over_channels(tmp_path: Path) -> None:
    env = environment(tmp_path)
    env["INFOVORE_EXCLUDE_CHANNELS"] = "general"
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"

    code, _, _ = run(
        ["sift", "export", "--size", "10", "--channels", "general", "--out", str(out_dir)], env
    )

    assert code == ExitCode.OK
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["message_ids"] == []


def test_sift_export_denylist_excludes_a_channel_without_channels_flag(tmp_path: Path) -> None:
    env = environment(tmp_path)
    env["INFOVORE_EXCLUDE_CHANNELS"] = "food"
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"

    code, _, _ = run(["sift", "export", "--size", "10", "--out", str(out_dir)], env)

    assert code == ExitCode.OK
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert sorted(manifest["message_ids"]) == [1, 2, 3]


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
    export_code, _, _ = run(["sift", "export", "--size", "6", "--out", str(out_dir)], env)
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

    code, _, err = run(["sift", "import", str(out_dir), "--save-rules", "food-chatter"], env)

    assert code == ExitCode.CONFIG
    assert "--save-rules" in err


def test_sift_import_save_rules_writes_a_file(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"
    run(["sift", "export", "--size", "6", "--out", str(out_dir)], env)
    (out_dir / "trash-regexes.csv").write_text("pattern\ngeneral chatter\n")

    code, out, _ = run(["sift", "import", str(out_dir), "--save-rules", "food-chatter"], env)

    assert code == ExitCode.OK
    assert "food-chatter" in out
    saved = tmp_path / "scratch" / "sift_trash_rules" / "food-chatter.json"
    assert saved.exists()


def seed_exchange_with_messages(
    conn: sqlite3.Connection, exchange_id: int, channel_id: int, message_ids: list[int]
) -> None:
    for message_id in message_ids:
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, ?, 1, 1, 'alice', ?, 'x', ?, '{}')",
            (message_id, channel_id, NOW, NOW),
        )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, 'quiet_gap', ?)",
        (
            exchange_id,
            channel_id,
            message_ids[0],
            message_ids[-1],
            NOW,
            NOW,
            len(message_ids),
            f"h{exchange_id}",
        ),
    )
    for position, message_id in enumerate(message_ids, start=1):
        conn.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, ?)",
            (exchange_id, message_id, position),
        )


def seed_labeled_corpus(db_path: str) -> None:
    from infovore.db.message_labels import set_message_label
    from infovore.rows import MessageLabel, MessageLabelSource

    conn = open_database(db_path)
    migrate(conn)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    for i in range(20):
        message_id = 1 + i
        _message_with_exchange(conn, message_id, 1, content=f"PROM 6.5.22 manual detail {i}")
        set_message_label(
            conn,
            message_id,
            MessageLabel.KEEP,
            MessageLabelSource.HUMAN,
            None,
            datetime(2026, 1, 1, tzinfo=UTC),
        )
    for i in range(20):
        message_id = 100 + i
        _message_with_exchange(conn, message_id, 2, content=f"lol gg no cap {i}")
        set_message_label(
            conn,
            message_id,
            MessageLabel.TRASH,
            MessageLabelSource.CITATION,
            None,
            datetime(2026, 1, 1, tzinfo=UTC),
        )
    # A citation-labeled `keep` class too -- issue #135's citation model
    # needs both classes among citation-only examples (this fixture's human
    # labels are all `keep`, none `trash`, so it stays below the human
    # minimum and falls back to citation-only; see the tests below).
    for i in range(15):
        message_id = 200 + i
        _message_with_exchange(conn, message_id, 2, content=f"PROM 6.5.22 manual detail {i}")
        set_message_label(
            conn,
            message_id,
            MessageLabel.KEEP,
            MessageLabelSource.CITATION,
            None,
            datetime(2026, 1, 1, tzinfo=UTC),
        )
    conn.close()


def seed_full_ensemble_corpus(db_path: str) -> None:
    """Enough citation *and* human labels of both classes (>= the default
    30/class minimum) to exercise the full two-model ensemble rather than
    the citation-only fallback -- issue #135."""
    from infovore.db.message_labels import set_message_label
    from infovore.rows import MessageLabel, MessageLabelSource

    conn = open_database(db_path)
    migrate(conn)
    _channel(conn, 1, "general")
    _channel(conn, 2, "food")
    when = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(35):
        message_id = 1 + i
        _message_with_exchange(conn, message_id, 1, content=f"PROM 6.5.22 manual detail {i}")
        set_message_label(conn, message_id, MessageLabel.KEEP, MessageLabelSource.HUMAN, None, when)
    for i in range(35):
        message_id = 1000 + i
        _message_with_exchange(conn, message_id, 1, content=f"lol gg no cap {i}")
        set_message_label(
            conn, message_id, MessageLabel.TRASH, MessageLabelSource.HUMAN, None, when
        )
    for i in range(40):
        message_id = 2000 + i
        _message_with_exchange(conn, message_id, 2, content=f"PROM 6.5.22 manual detail {i}")
        set_message_label(
            conn, message_id, MessageLabel.KEEP, MessageLabelSource.CITATION, None, when
        )
    for i in range(40):
        message_id = 3000 + i
        _message_with_exchange(conn, message_id, 2, content=f"lol gg no cap {i}")
        set_message_label(
            conn, message_id, MessageLabel.TRASH, MessageLabelSource.CITATION, None, when
        )
    conn.close()


def test_sift_citations_reports_keep_and_trash_counts(tmp_path: Path) -> None:
    from infovore.db.claims import NewClaim, record_run
    from infovore.rows import ClaimKind, ExtractionRunRow, RunMode, RunOutcome

    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    _channel(conn, 1, "general")
    seed_exchange_with_messages(conn, 1, 1, [1, 2])
    conn.execute(
        "INSERT INTO prompt_versions (version, text_sha256, created_at) VALUES ('v1', 'sha', ?)",
        (NOW,),
    )
    run_row = ExtractionRunRow(
        id=None,
        exchange_id=1,
        model="claude",
        prompt_version="v1",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        finished_at=datetime(2026, 1, 1, tzinfo=UTC),
        input_tokens=1,
        output_tokens=1,
        mode=RunMode.LIVE,
        outcome=RunOutcome.OK,
        error=None,
    )
    claim = NewClaim(
        exchange_id=1,
        statement="s",
        subject="subj",
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question="q?",
        permalink="https://x",
        supersedes_claim_id=None,
        source_message_ids=(1,),
    )
    record_run(conn, run_row, [claim])
    conn.close()

    code, out, _ = run(["sift", "citations"], env)

    assert code == ExitCode.OK
    assert "keep=1" in out
    assert "trash=1" in out


def test_sift_train_requires_minimum_labels(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])

    code, _, err = run(["sift", "train"], env)

    assert code == ExitCode.CONFIG
    assert "10" in err


def test_sift_train_trains_and_scores_messages(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled_corpus(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["sift", "train"], env)

    assert code == ExitCode.OK
    assert "trained message model" in out
    assert "scored" in out
    # seed_labeled_corpus has only 20 human `keep` labels and 0 human `trash`
    # labels -- well below the default per-class minimum, so this falls back
    # to the citation-only model (issue #135 design point 4).
    assert "fallback" in out
    assert "ablation" in out

    conn = open_database(env["INFOVORE_DB_PATH"])
    rows = conn.execute("SELECT p_trash FROM messages WHERE p_trash IS NOT NULL").fetchall()
    assert len(rows) == 55


def test_sift_train_reports_citation_human_and_combined_auc_when_not_fallback(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    seed_full_ensemble_corpus(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["sift", "train"], env)

    assert code == ExitCode.OK
    assert "fallback" not in out
    assert "citation-only: auc=" in out
    assert "human-only: auc=" in out
    assert "combined: auc=" in out
    assert "discard pile" in out
    assert "citation-holdout" in out
    assert "ablation" in out
    assert "with context features" in out
    assert "without context features" in out


def test_sift_train_human_weight_flag_is_a_documented_removal_error(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled_corpus(env["INFOVORE_DB_PATH"])

    code, _, err = run(["sift", "train", "--human-weight", "3"], env)

    assert code == ExitCode.CONFIG
    assert "--human-weight" in err
    assert "#135" in err


def test_sift_train_accepts_a_min_human_labels_flag(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_full_ensemble_corpus(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["sift", "train", "--min-human-labels", "1000"], env)

    assert code == ExitCode.OK
    assert "fallback" in out


def test_sift_export_strategy_uncertain_works_after_train(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled_corpus(env["INFOVORE_DB_PATH"])
    run(["sift", "train"], env)
    out_dir = tmp_path / "batch2"

    code, _, _ = run(
        ["sift", "export", "--size", "5", "--strategy", "uncertain", "--out", str(out_dir)], env
    )

    assert code == ExitCode.OK


def _no_block(event: threading.Event) -> None:
    return None


def test_sift_serve_requires_dir_or_new(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    code, _, err = run(["sift", "serve"], env)
    assert code == ExitCode.CONFIG
    assert "DIR or --new" in err


def test_sift_serve_rejects_dir_and_new_together(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"
    run(["sift", "export", "--size", "2", "--out", str(out_dir)], env)
    code, _, err = run(["sift", "serve", str(out_dir), "--new", "--out", str(out_dir)], env)
    assert code == ExitCode.CONFIG
    assert "DIR or --new" in err


def test_sift_serve_new_requires_out(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    code, _, err = run(["sift", "serve", "--new"], env)
    assert code == ExitCode.CONFIG
    assert "--out" in err


def test_sift_serve_rejects_mix_outside_zero_one(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"
    code, _, err = run(["sift", "serve", "--new", "--out", str(out_dir), "--mix", "1.5"], env)
    assert code == ExitCode.CONFIG
    assert "--mix" in err


def test_sift_serve_missing_manifest_exits_config(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    code, _, err = run(["sift", "serve", str(empty_dir)], env)
    assert code == ExitCode.CONFIG
    assert "manifest.json" in err


def test_sift_serve_uncertain_without_a_trained_model_exits_config(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"
    code, _, err = run(
        ["sift", "serve", "--new", "--strategy", "uncertain", "--out", str(out_dir)], env
    )
    assert code == ExitCode.CONFIG
    assert "p_trash" in err


def test_sift_serve_starts_and_prints_listening_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sift_command, "block_until_interrupted", _no_block)
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"
    run(["sift", "export", "--size", "3", "--out", str(out_dir)], env)

    code, out, _ = run(["sift", "serve", str(out_dir), "--host", "127.0.0.1", "--port", "0"], env)

    assert code == ExitCode.OK
    assert "listening on http://127.0.0.1:" in out
    assert "serving batch" in out


def test_sift_serve_new_starts_with_default_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sift_command, "block_until_interrupted", _no_block)
    monkeypatch.setattr(sift_command, "default_hosts", lambda: ["127.0.0.1"])
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"

    code, out, _ = run(
        ["sift", "serve", "--new", "--size", "2", "--out", str(out_dir), "--port", "0"], env
    )

    assert code == ExitCode.OK
    assert "listening on http://127.0.0.1:" in out


def test_sift_serve_new_channels_restricts_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sift_command, "block_until_interrupted", _no_block)
    monkeypatch.setattr(sift_command, "default_hosts", lambda: ["127.0.0.1"])
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"

    code, out, _ = run(
        [
            "sift",
            "serve",
            "--new",
            "--size",
            "10",
            "--channels",
            "general",
            "--out",
            str(out_dir),
            "--port",
            "0",
        ],
        env,
    )

    assert code == ExitCode.OK
    assert "serving batch" in out
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert sorted(manifest["message_ids"]) == [1, 2, 3]


def test_sift_serve_new_unknown_channel_exits_config(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"

    code, _, err = run(["sift", "serve", "--new", "--channels", "nope", "--out", str(out_dir)], env)

    assert code == ExitCode.CONFIG
    assert "nope" in err


def test_sift_serve_new_denylist_wins_over_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sift_command, "block_until_interrupted", _no_block)
    monkeypatch.setattr(sift_command, "default_hosts", lambda: ["127.0.0.1"])
    env = environment(tmp_path)
    env["INFOVORE_EXCLUDE_CHANNELS"] = "general"
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"

    code, _out, _ = run(
        [
            "sift",
            "serve",
            "--new",
            "--size",
            "10",
            "--channels",
            "general",
            "--out",
            str(out_dir),
            "--port",
            "0",
        ],
        env,
    )

    assert code == ExitCode.OK
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["message_ids"] == []


def test_sift_serve_reports_a_port_already_in_use_as_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    monkeypatch.setattr(sift_command, "block_until_interrupted", _no_block)
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_dir = tmp_path / "batch"
    run(["sift", "export", "--size", "2", "--out", str(out_dir)], env)

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        code, _, err = run(
            ["sift", "serve", str(out_dir), "--host", "127.0.0.1", "--port", str(port)], env
        )
        assert code == ExitCode.CONFIG
        assert "could not bind" in err
    finally:
        blocker.close()
