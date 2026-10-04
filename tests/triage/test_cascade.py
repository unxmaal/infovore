import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.cli import ExitCode
from infovore.db.annotations import Annotation, annotation_history, record_annotation
from infovore.db.connection import migrate, open_database
from infovore.rows import Label
from infovore.triage.cascade import (
    PRECISION_TARGET,
    EmbedStage,
    Outcome,
    decide_embed,
    decide_lexicon,
    residue_channels,
    stage_reports,
    tune_high,
    tuning_labels,
    tuning_samples,
)
from infovore.triage.human import HUMAN_SCORER
from infovore.triage.lexicon import LexiconScore, load_lexicon
from tests.triage.test_command import environment, run
from tests.triage.test_embed import FakeEmbedder
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


def test_the_embed_band_decides_only_outside_the_thresholds() -> None:
    stage = EmbedStage(lambda ids: {}, 0.3, 0.8, {})

    assert decide_embed(None, stage) is None
    assert decide_embed(0.95, stage) == "relevant"
    assert decide_embed(0.8, stage) == "relevant"
    assert decide_embed(0.29, stage) == "irrelevant"
    assert decide_embed(0.3, stage) is None
    assert decide_embed(0.5, stage) is None
    assert decide_embed(0.0, EmbedStage.abstaining("why")) is None
    assert decide_embed(1.0, EmbedStage.abstaining("why")) is None


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
    denylist, no_text, lexicon, short, embed, residue = stage_reports(outcomes, labels)

    assert (lexicon.decided, lexicon.relevant, lexicon.irrelevant) == (4, 2, 2)
    assert (lexicon.tp, lexicon.fp, lexicon.fn, lexicon.tn) == (1, 1, 1, 1)
    assert (lexicon.labelled, lexicon.correct, lexicon.accuracy) == (4, 2, 0.5)
    assert denylist.decided == 0 and denylist.accuracy is None
    assert embed.decided == 0 and embed.accuracy is None
    assert short.decided == 0
    assert (residue.decided, residue.share) == (1, 0.2)
    assert no_text.decided == 0
    assert stage_reports([], {})[5].share == 0.0


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
    assert "stage embed: abstains on everything" in out
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
    assert annotation_history(conn, "exchange", 2, "relevance_embed") == []
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


@pytest.fixture
def fitted_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("infovore.triage.embed_stage.FOLDS", 2)
    monkeypatch.setattr("infovore.triage.embed_stage.load_embedder", lambda *_: FakeEmbedder())


def test_the_embed_stage_scores_what_the_lexicon_abstained_on(
    tmp_path: Path, fitted_embed: None
) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = run(["relevance", "cascade", "--slices", "s1", "--write"], env)

    assert code == ExitCode.OK
    assert "abstains on everything" not in out
    assert "embed fake/keywords@r1 pool=first irrelevant_below=" in out
    assert "relevant_at_or_above=" in out
    assert "cv irrelevant: n=" in out and "cv relevant: n=" in out
    assert "precision=" in out and "recall=" in out
    assert "labels={'relevant': 2, 'irrelevant': 3}" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    row = annotation_history(conn, "exchange", 7, "relevance_embed")[0]
    assert row["score"] is not None
    recipe = json.loads(row["recipe_json"])
    assert recipe["model"] == "fake/keywords"
    assert recipe["revision"] == "r1"
    assert recipe["pool"] == "first"
    assert recipe["labels"] == {"relevant": 2, "irrelevant": 3}
    assert set(recipe["thresholds"]) >= {"irrelevant_below", "relevant_at_or_above"}
    assert annotation_history(conn, "exchange", 2, "relevance_embed") == []
    assert (tmp_path / "embed-cache.db").exists()


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
    conn.execute("UPDATE exchanges SET channel_id = 2 WHERE id IN (2, 4)")
    conn.commit()
    conn.close()
    env["INFOVORE_EXCLUDE_CHANNELS"] = "food"

    code, out, _ = run(["relevance", "cascade", "--slices", "s1", "--write"], env)

    assert code == ExitCode.OK
    assert "stage denylist: decided=2 relevant=0 irrelevant=2 labelled=2" in out
    assert "accuracy=0.500" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    row = annotation_history(conn, "exchange", 2, "relevance_denylist")[0]
    assert (row["label"], row["reproducibility"]) == ("irrelevant", "derived")
    assert annotation_history(conn, "exchange", 2, "relevance_lexicon") == []
    assert annotation_history(conn, "exchange", 3, "relevance_denylist") == []


def test_text_less_exchanges_are_set_aside_before_the_lexicon(tmp_path: Path) -> None:
    env, conn = build(tmp_path)
    conn.execute("UPDATE messages SET content = '  ' WHERE id = 2")
    conn.execute("UPDATE messages SET content = '[redacted]' WHERE id = 4")
    conn.execute("UPDATE messages SET deleted_at = ? WHERE id = 5", (NOW.isoformat(),))
    conn.commit()
    conn.close()

    code, out, _ = run(["relevance", "cascade", "--slices", "s1", "--write"], env)

    assert code == ExitCode.OK
    assert "stage no_text: n=3 share=0.600 (set aside, not scored)" in out
    assert "stage lexicon: decided=1" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    row = annotation_history(conn, "exchange", 2, "relevance_no_text")[0]
    assert (row["label"], row["reproducibility"]) == ("no_text", "derived")
    assert annotation_history(conn, "exchange", 2, "relevance_lexicon") == []
    assert annotation_history(conn, "exchange", 1, "relevance_no_text") == []


def test_the_denylist_outranks_no_text(tmp_path: Path) -> None:
    env, conn = build(tmp_path)
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (2, 9, NULL, 'food', 'text')"
    )
    conn.execute("UPDATE exchanges SET channel_id = 2 WHERE id = 2")
    conn.execute("UPDATE messages SET content = '' WHERE id = 2")
    conn.commit()
    conn.close()
    env["INFOVORE_EXCLUDE_CHANNELS"] = "food"

    code, out, _ = run(["relevance", "cascade", "--slices", "s1", "--write"], env)

    assert code == ExitCode.OK
    assert "stage denylist: decided=1" in out
    assert "stage no_text: n=0" in out


def test_all_runs_every_current_exchange_and_reports_share_and_residue_channels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env, conn = build(tmp_path)
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 9, NULL, 'general', 'text')"
    )
    conn.execute("UPDATE exchanges SET superseded_by_recipe = 2 WHERE id = 7")
    conn.commit()
    conn.close()

    code, out, _ = run(["relevance", "cascade", "--all"], env)

    assert code == ExitCode.OK
    assert "corpus: n=6" in out
    assert "stage lexicon: decided=4 relevant=4 irrelevant=0 share=0.667" in out
    assert "stage embed: abstains on everything" in out
    assert "residue: n=2 share=0.333" in out
    assert "residue channel general: 2" in out
    assert "labelled=" in out and "accuracy=" in out
    assert "progress 6/6" in capsys.readouterr().err


def test_all_with_write_annotates_exchanges_outside_every_slice(tmp_path: Path) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = run(["relevance", "cascade", "--all", "--write"], env)

    assert code == ExitCode.OK
    assert "wrote 7 exchanges" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    assert annotation_history(conn, "exchange", 6, "relevance_lexicon")
    assert annotation_history(conn, "exchange", 2, "relevance_lexicon")[0]["label"] == "relevant"


def test_all_with_a_fitted_embed_stage_reports_the_stage(
    tmp_path: Path, fitted_embed: None
) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = run(["relevance", "cascade", "--all"], env)

    assert code == ExitCode.OK
    assert "abstains on everything" not in out
    assert "stage embed: decided=" in out


def test_residue_channels_counts_by_channel_name_and_falls_back_to_the_id(
    tmp_path: Path,
) -> None:
    _, conn = build(tmp_path)
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 9, NULL, 'general', 'text')"
    )
    conn.execute("UPDATE exchanges SET channel_id = 5 WHERE id IN (3, 4)")
    conn.commit()
    outcomes = [outcome(1, "residue", "residue"), outcome(2, "lexicon", "relevant")]
    outcomes += [outcome(3, "residue", "residue"), outcome(4, "residue", "residue")]

    assert residue_channels(conn, outcomes, 15) == [("5", 2), ("general", 1)]
    assert residue_channels(conn, outcomes, 1) == [("5", 2)]
    assert residue_channels(conn, [], 15) == []


def test_batching_gives_the_same_outcomes_as_one_pass(tmp_path: Path) -> None:
    from infovore.triage.cascade import run_cascade, run_cascade_batched
    from infovore.triage.lexicon import load_lexicon

    _, conn = build(tmp_path)
    ids = [1, 2, 3, 4, 5, 6, 7]
    seen: list[tuple[int, int]] = []

    batched = run_cascade_batched(
        conn,
        ids,
        load_lexicon(),
        0.5,
        EmbedStage.abstaining("x"),
        frozenset(),
        3,
        lambda d, t: seen.append((d, t)),
    )

    assert batched == run_cascade(
        conn, ids, load_lexicon(), 0.5, EmbedStage.abstaining("x"), frozenset()
    )
    assert seen == [(3, 7), (6, 7), (7, 7)]


def queue_label(conn: sqlite3.Connection, eid: int, label: str, ref: str) -> None:
    record_annotation(
        conn,
        Annotation("exchange", eid, HUMAN_SCORER, 1, "recorded", label=label, source_ref=ref),
        NOW,
    )


@pytest.mark.parametrize("ref", ["judge:likely-irrelevant:3", "judge:uncertain:9", "judge:c1:2"])
def test_tuning_ignores_queue_sourced_labels(tmp_path: Path, ref: str) -> None:
    _, conn = build(tmp_path)
    lexicon = load_lexicon()
    before = tuning_samples(conn, lexicon)
    ids = [row[0] for row in conn.execute("SELECT id FROM exchanges ORDER BY id")]
    queue_label(conn, ids[1], "irrelevant", ref)
    queue_label(conn, ids[2], "irrelevant", ref)
    extra = seed(conn, 50, "scsi disk boot")
    conn.execute(
        "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
        " VALUES ('s1', ?, 99, 't', 0, ?)",
        (extra, NOW.isoformat()),
    )
    queue_label(conn, extra, "relevant", ref)
    conn.commit()

    assert tuning_samples(conn, lexicon) == before
    assert len(tuning_labels(conn)) == len(before)
    assert extra not in tuning_labels(conn)


def test_a_random_slice_label_after_a_queue_label_still_counts(tmp_path: Path) -> None:
    _, conn = build(tmp_path)
    eid = conn.execute("SELECT id FROM exchanges ORDER BY id LIMIT 1 OFFSET 3").fetchone()[0]
    queue_label(conn, eid, "irrelevant", "judge:likely-irrelevant:1")
    human(conn, eid, "relevant")
    conn.commit()

    assert tuning_labels(conn)[eid] is Label.LORE


def test_the_cascade_prints_the_tuning_population(tmp_path: Path) -> None:
    env, conn = build(tmp_path)
    conn.close()

    _, out, _ = run(["relevance", "cascade", "--slices", "s1"], env)

    assert "tuned on s1 random-slice labels, held-out and queue-sourced excluded: n=5" in out
    assert "relevant=2 irrelevant=3" in out


def test_the_relevant_precision_flag_reaches_the_recorded_recipe(
    tmp_path: Path, fitted_embed: None
) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, _, _ = run(
        ["relevance", "cascade", "--slices", "s1", "--write", "--relevant-precision", "0.5"], env
    )

    assert code == ExitCode.OK
    conn = open_database(env["INFOVORE_DB_PATH"])
    row = annotation_history(conn, "exchange", 7, "relevance_embed")[0]
    assert json.loads(row["recipe_json"])["thresholds"]["relevant_precision_target"] == 0.5
