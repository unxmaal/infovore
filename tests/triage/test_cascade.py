import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.cli import ExitCode
from infovore.db.annotations import annotation_history
from infovore.db.connection import migrate, open_database
from infovore.rows import Label
from infovore.triage.cascade import (
    PRECISION_TARGET,
    Outcome,
    decide_bayes,
    decide_lexicon,
    stage_reports,
    tune_high,
)
from infovore.triage.lexicon import LexiconScore
from tests.triage.test_command import environment, run
from tests.triage.test_human import human, seed
from tests.triage.test_lexicon import corpus

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_tuning_takes_the_lowest_share_that_keeps_precision() -> None:
    samples = [(0.2, False), (0.4, True), (0.5, True), (0.9, True), (1.0, True)]

    assert tune_high(samples, 1.0) == 0.4
    assert tune_high([(0.2, False), (0.3, False)], 1.0) > 1.0
    assert tune_high([(0.0, True), (0.5, True)], 1.0) == 0.5
    assert PRECISION_TARGET > 0.9


def test_the_lexicon_abstains_without_hits_and_decides_on_share() -> None:
    assert decide_lexicon(LexiconScore(0.0, 0, 3), 0.3) is None
    assert decide_lexicon(LexiconScore(0.2, 1, 5), 0.3) is None
    assert decide_lexicon(LexiconScore(0.5, 2, 4), 0.3) == "relevant"


def test_the_bayes_band_decides_only_far_from_a_coin_flip() -> None:
    assert decide_bayes(None) is None
    assert decide_bayes(0.95) == "relevant"
    assert decide_bayes(0.05) == "irrelevant"
    assert decide_bayes(0.5) is None


def outcome(eid: int, stage: str, decision: str) -> Outcome:
    return Outcome(eid, 0.5, 1, 2, None, stage, decision)


def test_stage_reports_count_decisions_confusion_and_residue() -> None:
    outcomes = [
        outcome(1, "lexicon", "relevant"),
        outcome(2, "lexicon", "relevant"),
        outcome(3, "lexicon", "irrelevant"),
        outcome(4, "lexicon", "irrelevant"),
        outcome(5, "residue", "residue"),
    ]
    labels = {1: Label.LORE, 2: Label.NOISE, 3: Label.LORE, 4: Label.NOISE}
    denylist, lexicon, bayes, residue = stage_reports(outcomes, labels)

    assert (lexicon.decided, lexicon.relevant, lexicon.irrelevant) == (4, 2, 2)
    assert (lexicon.tp, lexicon.fp, lexicon.fn, lexicon.tn) == (1, 1, 1, 1)
    assert (lexicon.labelled, lexicon.correct, lexicon.accuracy) == (4, 2, 0.5)
    assert denylist.decided == 0 and denylist.accuracy is None
    assert bayes.decided == 0 and bayes.accuracy is None
    assert (residue.decided, residue.share) == (1, 0.2)
    assert stage_reports([], {})[3].share == 0.0


def build(tmp_path: Path) -> tuple[dict[str, str], sqlite3.Connection]:
    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    ids = [
        seed(conn, 1, "my scsi disk will not boot"),
        seed(conn, 2, "scsi disk boot again"),
        seed(conn, 3, "the kernel driver compile failed"),
        seed(conn, 4, "lol great lunch"),
        seed(conn, 5, "nice weather, hello"),
        seed(conn, 6, "gold exchange about a scsi disk"),
        seed(conn, 7, "scsi disk"),
    ]
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (107, 1, 9, 7, 'a', ?, 'lol lunch', ?, '{}')",
        (NOW.isoformat(), NOW.isoformat()),
    )
    conn.execute(
        "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, 107, 2)",
        (ids[6],),
    )
    for eid, label in zip(
        ids, ["relevant"] * 3 + ["irrelevant"] * 2 + ["relevant", "irrelevant"], strict=True
    ):
        human(conn, eid, label)
    for name, members in (("s1", [*ids[:5], ids[6]]), ("gold", [ids[5], ids[0]])):
        for position, eid in enumerate(members, start=1):
            conn.execute(
                "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
                " VALUES (?, ?, ?, 't', 0, ?)",
                (name, eid, position, NOW.isoformat()),
            )
    conn.commit()
    return env, conn


def test_the_cascade_reports_each_stage_per_slice(tmp_path: Path) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = run(["relevance", "cascade", "--slices", "s1,gold"], env)

    assert code == ExitCode.OK
    assert "slice s1 (tuning, held-out excluded): n=5" in out
    assert "slice gold (held-out): n=2" in out
    assert "stage lexicon: decided=" in out
    assert "stage bayes: abstains on everything" in out
    assert "stage residue:" in out
    assert "accuracy=1.000" in out
    assert "irrelevant=0" in out
    assert "residue: n=3 share=0.600" in out
    assert "lexicon v=lx-" in out


def test_write_records_derived_annotations_per_stage(tmp_path: Path) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = run(["relevance", "cascade", "--slices", "s1", "--write"], env)

    assert code == ExitCode.OK
    assert "wrote" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    rows = annotation_history(conn, "exchange", 2, "relevance_lexicon")
    assert rows[0]["reproducibility"] == "derived"
    assert '"lexicon_version": "lx-' in rows[0]["recipe_json"]
    assert rows[0]["label"] == "relevant"
    assert annotation_history(conn, "exchange", 7, "relevance_residue")[0]["label"] == "residue"
    assert annotation_history(conn, "exchange", 2, "relevance_bayes") == []
    code, out, _ = run(["relevance", "cascade", "--slices", "s1", "--write"], env)
    assert code == ExitCode.OK
    assert len(annotation_history(conn, "exchange", 2, "relevance_lexicon")) == 2
    code, out, _ = run(["triage", "--human-report", "--scorer", "relevance_lexicon"], env)
    assert code == ExitCode.OK
    assert "scorer relevance_lexicon v2" in out


def test_explain_shows_hits_per_message(tmp_path: Path) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = run(["relevance", "cascade", "--explain", "1"], env)

    assert code == ExitCode.OK
    assert "scsi" in out and "share=1.00" in out
    code, out, _ = run(["relevance", "cascade", "--explain", "999"], env)
    assert code == ExitCode.CONFIG


def test_the_slices_are_validated(tmp_path: Path) -> None:
    env, conn = build(tmp_path)
    conn.close()

    assert run(["relevance", "cascade", "--slices", "nope"], env)[0] == ExitCode.CONFIG
    assert run(["relevance", "cascade"], env)[0] == ExitCode.CONFIG


def test_bayes_scores_what_the_lexicon_abstained_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, conn = build(tmp_path)
    conn.close()
    monkeypatch.setattr("infovore.triage.cascade.MIN_PER_CLASS", 1)

    code, out, _ = run(["relevance", "cascade", "--slices", "s1", "--write"], env)

    assert code == ExitCode.OK
    assert "abstains on everything" not in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    assert annotation_history(conn, "exchange", 7, "relevance_bayes")[0]["score"] is not None
    assert annotation_history(conn, "exchange", 2, "relevance_bayes") == []


def test_mine_prints_candidates(tmp_path: Path) -> None:
    env = environment(tmp_path)
    corpus(tmp_path / "infovore.db").close()

    code, out, _ = run(
        ["relevance", "mine", "--tech", "tech", "--off", "chat", "--min-count", "2"], env
    )

    assert code == ExitCode.OK
    assert "zorp" in out
    code, out, _ = run(
        ["relevance", "mine", "--tech", "tech", "--off", "chat", "--min-count", "99"], env
    )
    assert "no candidates" in out


def test_collisions_lists_lexicon_terms_common_off_topic(tmp_path: Path) -> None:
    env = environment(tmp_path)
    corpus(tmp_path / "infovore.db").close()

    code, out, _ = run(
        [
            "relevance",
            "collisions",
            "--tech",
            "tech",
            "--off",
            "chat",
            "--max-off",
            "1",
            "--min-ratio",
            "99",
            "--terms",
            "zorp,lunch",
        ],
        env,
    )

    assert code == ExitCode.OK
    assert "zorp\t2\t1" in out
    code, out, _ = run(
        ["relevance", "collisions", "--tech", "tech", "--off", "chat", "--terms", "zorp"], env
    )
    assert "no collisions" in out
    code, out, _ = run(["relevance", "collisions", "--tech", "tech", "--off", "chat"], env)
    assert code == ExitCode.OK


def test_the_denylist_decides_irrelevant_before_the_lexicon(tmp_path: Path) -> None:
    env, conn = build(tmp_path)
    for cid, name in ((1, "tech"), (2, "food")):
        conn.execute(
            "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
            " VALUES (?, 9, NULL, ?, 'text')",
            (cid, name),
        )
    conn.execute("UPDATE exchanges SET channel_id = 2 WHERE id IN (1, 4)")
    conn.commit()
    conn.close()
    env["INFOVORE_EXCLUDE_CHANNELS"] = "food"

    code, out, _ = run(["relevance", "cascade", "--slices", "s1", "--write"], env)

    assert code == ExitCode.OK
    assert "stage denylist: decided=2 relevant=0 irrelevant=2 labelled=2" in out
    assert "accuracy=0.500" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    row = annotation_history(conn, "exchange", 1, "relevance_denylist")[0]
    assert (row["label"], row["reproducibility"]) == ("irrelevant", "derived")
    assert annotation_history(conn, "exchange", 1, "relevance_lexicon") == []
    assert annotation_history(conn, "exchange", 2, "relevance_denylist") == []
