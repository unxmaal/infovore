import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.annotations import Annotation, record_annotation
from infovore.db.connection import migrate, open_database
from infovore.db.labels import set_label
from infovore.rows import Label, LabelSource, MessageRow
from infovore.triage.human import (
    HUMAN_SCORER,
    SCORER,
    GazetteerError,
    InsufficientHumanLabelsError,
    LlmLabelsNotTrainableError,
    NoScorerAnnotationsError,
    evaluate_scorer,
    fit_human,
    human_features,
    load_gazetteer,
    next_version,
    parse_gazetteer,
    rule_names,
    score_human,
    training_labels,
)
from tests.triage.test_command import environment, run

NOW = datetime(2026, 1, 1, tzinfo=UTC)
LORE = "my Indy runs IRIX 6.5.22, PROM says 030-1234-001, hinv shows nothing"
NOISE = "lol gg"


def message(content: str, *, bot: bool = False, mid: int = 1) -> MessageRow:
    return MessageRow(
        id=mid,
        channel_id=1,
        guild_id=1,
        author_id=mid,
        author_name_at_time="a",
        author_is_bot=bot,
        created_at=NOW,
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        deleted_at=None,
        ingested_at=NOW,
        raw_json="{}",
    )


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def seed(conn: sqlite3.Connection, index: int, content: str) -> int:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, content, ingested_at, raw_json)"
        " VALUES (?, 1, 9, ?, 'a', 0, ?, ?, ?, '{}')",
        (index, index, NOW.isoformat(), content, NOW.isoformat()),
    )
    cursor = conn.execute(
        "INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash, extraction_status)"
        " VALUES (1, ?, ?, ?, ?, 1, 'quiet_gap', ?, 'pending')",
        (index, index, NOW.isoformat(), NOW.isoformat(), f"h{index}"),
    )
    exchange_id = int(cursor.lastrowid or 0)
    conn.execute(
        "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
        (exchange_id, index),
    )
    return exchange_id


def human(conn: sqlite3.Connection, exchange_id: int, label: str) -> None:
    record_annotation(
        conn,
        Annotation("exchange", exchange_id, HUMAN_SCORER, 1, "recorded", label=label),
        NOW,
    )


def seed_labeled(conn: sqlite3.Connection, relevant: int, irrelevant: int) -> list[int]:
    ids = []
    for index in range(relevant):
        ids.append(seed(conn, index + 1, f"{LORE} {index}"))
        human(conn, ids[-1], "relevant")
    for index in range(irrelevant):
        ids.append(seed(conn, relevant + index + 1, f"{NOISE} {index}"))
        human(conn, ids[-1], "irrelevant")
    conn.commit()
    return ids


def test_rule_hits_become_rule_tokens_next_to_text_tokens() -> None:
    tokens = human_features([message(LORE)])

    assert "indy" in tokens
    assert {
        "__RULE_gaz_models",
        "__RULE_gaz_irix_version",
        "__RULE_gaz_prom",
        "__RULE_gaz_part_number",
        "__RULE_gaz_tools",
        "__RULE_part_number",
    } <= tokens


def test_error_shapes_fire() -> None:
    assert "gaz_error_shape" in rule_names([message("sh: foo: command not found")])


def test_negative_rules_fire() -> None:
    assert "gaz_bot_author" in rule_names([message("deploy ok " * 5, bot=True)])
    assert "gaz_very_short" in rule_names([message("ok")])
    links = [message("https://x.example/a", mid=1), message("<https://y.example>", mid=2)]
    assert "gaz_link_only" in rule_names(links)
    assert "gaz_greeting" in rule_names([message("hello"), message("gm", mid=2)])
    assert "gaz_meme" in rule_names([message("lmao"), message("based", mid=2)])


def test_empty_exchange_has_no_rules() -> None:
    assert rule_names([]) == frozenset()


def test_gazetteer_loads_and_versions() -> None:
    assert load_gazetteer().version.startswith("g-")
    assert load_gazetteer() is load_gazetteer()


@pytest.mark.parametrize(
    "text",
    ['[text]\na = ["x"]', '[text]\na = "x"\n[message_share]', "text = 1\nmessage_share = 2"],
)
def test_bad_gazetteer_is_refused(text: str) -> None:
    with pytest.raises(GazetteerError):
        parse_gazetteer(text)


def test_latest_label_wins_and_bad_grouping_excludes(tmp_path: Path) -> None:
    conn = db(tmp_path)
    a, b, c = (seed(conn, i, "x") for i in (1, 2, 3))
    human(conn, a, "relevant")
    human(conn, a, "irrelevant")
    human(conn, b, "relevant")
    human(conn, b, "bad_grouping")
    human(conn, c, "relevant")

    labels, last_id = training_labels(conn)

    assert labels == {a: Label.NOISE, c: Label.LORE}
    assert last_id == 5


def test_llm_labels_are_refused_explicitly(tmp_path: Path) -> None:
    conn = db(tmp_path)
    with pytest.raises(LlmLabelsNotTrainableError, match="llm"):
        training_labels(conn, "llm")


def test_llm_exchange_labels_never_reach_training(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ids = seed_labeled(conn, 3, 3)
    extra = seed(conn, 100, "llm only")
    set_label(conn, extra, Label.LORE, LabelSource.LLM, None, NOW)
    set_label(conn, ids[0], Label.NOISE, LabelSource.HUMAN, None, NOW)

    labels, _ = training_labels(conn)

    assert extra not in labels
    assert labels[ids[0]] is Label.LORE


def test_training_refuses_below_the_minimum(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled(conn, 5, 2)

    with pytest.raises(InsufficientHumanLabelsError) as info:
        fit_human(conn, minimum=10)

    assert (info.value.more_relevant, info.value.more_irrelevant) == (5, 8)
    assert "label 5 more relevant and 8 more irrelevant" in str(info.value)


def test_fit_reports_holdout_and_learns_rule_weights(tmp_path: Path) -> None:
    conn = db(tmp_path)
    seed_labeled(conn, 40, 40)

    fit = fit_human(conn, minimum=40)

    assert fit.report.relevant == 40
    assert fit.report.holdout_size > 0
    assert fit.report.auc == 1.0
    assert len(fit.report.metrics) == 9
    assert fit.recipe["label_scorer"] == HUMAN_SCORER
    assert fit.model.counts["__RULE_gaz_models"][0] > 0


def test_score_writes_derived_annotations_and_leaves_p_lore(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ids = seed_labeled(conn, 40, 40)
    fit = fit_human(conn, minimum=40)

    version, written = score_human(conn, fit, NOW, limit=10)
    assert (version, written) == (1, 10)
    assert next_version(conn) == 2
    _, written_all = score_human(conn, fit, NOW)

    rows = conn.execute(
        "SELECT * FROM annotations WHERE scorer = ? AND scorer_version = 1", (SCORER,)
    ).fetchall()
    assert len(rows) == 10
    assert all(row["reproducibility"] == "derived" and row["recipe_json"] for row in rows)
    assert written_all == len(ids)
    assert (
        conn.execute("SELECT COUNT(*) FROM exchanges WHERE p_lore IS NOT NULL").fetchone()[0] == 0
    )


def test_cli_report_says_how_many_more_labels(tmp_path: Path) -> None:
    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    seed_labeled(conn, 5, 2)
    conn.close()

    code, out, _ = run(["triage", "--human-report"], env)

    assert code == 0
    assert "label 195 more relevant and 198 more irrelevant" in out


def test_cli_train_human_refuses_then_trains(tmp_path: Path) -> None:
    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    seed_labeled(conn, 30, 30)
    conn.close()

    code, _, err = run(["triage", "--train-human"], env)
    assert code != 0
    assert "--human-limit N or --all-exchanges" in err

    code, _, err = run(["triage", "--train-human", "--all-exchanges"], env)
    assert code != 0
    assert "need at least 200" in err

    code, out, _ = run(["triage", "--human-report", "--min-per-class", "30"], env)
    assert code == 0
    assert "relevant=30 irrelevant=30" in out
    assert "wrote" not in out

    code, out, _ = run(
        ["triage", "--train-human", "--min-per-class", "30", "--human-limit", "5"], env
    )
    assert code == 0
    assert f"wrote 5 derived annotations as {SCORER} v1" in out


def derive(conn: sqlite3.Connection, scorer: str, version: int, scores: dict[int, float]) -> None:
    for exchange_id, score in scores.items():
        record_annotation(
            conn,
            Annotation(
                "exchange", exchange_id, scorer, version, "derived", score=score, recipe={"k": 1}
            ),
            NOW,
        )


def labeled_four(conn: sqlite3.Connection) -> list[int]:
    ids = seed_labeled(conn, 2, 2)
    return ids


def test_evaluate_scorer_scores_any_scorer_against_the_human_labels(tmp_path: Path) -> None:
    conn = db(tmp_path)
    a, b, c, d = labeled_four(conn)
    extra = seed(conn, 99, "unlabelled")
    derive(conn, "local-model", 3, {a: 0.9, b: 0.6, c: 0.7, d: 0.1, extra: 0.99})

    result = evaluate_scorer(conn, "local-model", include_training=True)

    assert (result.scorer, result.version, result.evaluated) == ("local-model", 3, 4)
    assert (result.relevant, result.irrelevant) == (2, 2)
    assert result.auc == 0.75
    at_half = next(m for m in result.metrics if m.threshold == 0.5)
    assert (at_half.tp, at_half.fp, at_half.fn, at_half.tn) == (2, 1, 0, 1)
    assert (at_half.precision, at_half.recall) == (2 / 3, 1.0)


def test_evaluate_scorer_defaults_to_the_latest_version_and_honours_an_explicit_one(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    a, b, c, d = labeled_four(conn)
    derive(conn, "local-model", 1, {a: 0.9, b: 0.9, c: 0.1, d: 0.1})
    derive(conn, "local-model", 2, {a: 0.1, b: 0.1, c: 0.9, d: 0.9})

    assert evaluate_scorer(conn, "local-model", include_training=True).auc == 0.0
    assert evaluate_scorer(conn, "local-model", 1, include_training=True).auc == 1.0


def test_evaluate_scorer_ignores_labels_the_scorer_did_not_score(tmp_path: Path) -> None:
    conn = db(tmp_path)
    a, _, c, _ = labeled_four(conn)
    derive(conn, "local-model", 1, {a: 0.9, c: 0.1})

    result = evaluate_scorer(conn, "local-model", include_training=True)

    assert (result.evaluated, result.relevant, result.irrelevant) == (2, 1, 1)


def test_evaluate_scorer_ignores_other_scorers_and_non_derived_rows(tmp_path: Path) -> None:
    conn = db(tmp_path)
    a, b, c, d = labeled_four(conn)
    derive(conn, "other", 1, {a: 0.9, b: 0.9, c: 0.1, d: 0.1})

    with pytest.raises(NoScorerAnnotationsError):
        evaluate_scorer(conn, "local-model", include_training=True)
    with pytest.raises(NoScorerAnnotationsError):
        evaluate_scorer(conn, "other", 7, include_training=True)
    with pytest.raises(NoScorerAnnotationsError):
        evaluate_scorer(conn, HUMAN_SCORER, include_training=True)


def test_cli_human_report_with_a_scorer_compares_it_head_to_head(tmp_path: Path) -> None:
    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    a, b, c, d = labeled_four(conn)
    derive(conn, "local-model", 2, {a: 0.9, b: 0.6, c: 0.7, d: 0.1})
    conn.commit()
    conn.close()

    code, out, _ = run(
        ["triage", "--human-report", "--scorer", "local-model", "--include-training"], env
    )
    assert code == 0
    assert "scorer local-model v2: evaluated=4 relevant=2 irrelevant=2 auc=0.750" in out
    assert "threshold  tp  fp  fn  tn  precision  recall  f1" in out

    code, out, _ = run(
        [
            "triage",
            "--human-report",
            "--scorer",
            "local-model",
            "--scorer-version",
            "2",
            "--include-training",
        ],
        env,
    )
    assert code == 0
    assert "local-model v2" in out


def test_cli_human_report_with_an_unknown_scorer_is_a_config_error(tmp_path: Path) -> None:
    env = environment(tmp_path)

    code, _, err = run(["triage", "--human-report", "--scorer", "nope", "--include-training"], env)

    assert code != 0
    assert "no derived annotations for scorer 'nope'" in err


def test_cli_human_report_with_a_scorer_and_no_overlap_says_auc_is_undefined(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    lone = seed(conn, 1, "x")
    derive(conn, "local-model", 1, {lone: 0.5})
    conn.commit()
    conn.close()

    code, out, _ = run(
        ["triage", "--human-report", "--scorer", "local-model", "--include-training"], env
    )

    assert code == 0
    assert "evaluated=0" in out
    assert "auc=n/a" in out


def freeze(conn: sqlite3.Connection, name: str, ids: list[int]) -> None:
    for position, exchange_id in enumerate(ids, start=1):
        conn.execute(
            "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
            " VALUES (?, ?, ?, 'p', 1, ?)",
            (name, exchange_id, position, NOW.isoformat()),
        )
    conn.commit()


def test_evaluate_scorer_defaults_to_held_out_s2_and_gold(tmp_path: Path) -> None:
    conn = db(tmp_path)
    a, b, c, d = labeled_four(conn)
    derive(conn, "local-model", 1, {a: 0.9, b: 0.2, c: 0.95, d: 0.1})
    freeze(conn, "s2", [a, c])
    freeze(conn, "gold", [d])
    freeze(conn, "s1", [b])

    held = evaluate_scorer(conn, "local-model")
    everything = evaluate_scorer(conn, "local-model", include_training=True)

    assert held.population == "held-out (s2, gold)"
    assert (held.evaluated, held.relevant, held.irrelevant) == (3, 1, 2)
    assert held.auc == 0.5
    assert everything.population == "all human labels (includes training)"
    assert everything.evaluated == 4


def test_evaluate_scorer_held_out_is_empty_without_frozen_slices(tmp_path: Path) -> None:
    conn = db(tmp_path)
    a, b, c, d = labeled_four(conn)
    derive(conn, "local-model", 1, {a: 0.9, b: 0.2, c: 0.8, d: 0.1})

    assert evaluate_scorer(conn, "local-model").evaluated == 0


def test_fit_human_never_trains_on_s2_or_gold_exchanges(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ids = seed_labeled(conn, 40, 40)
    freeze(conn, "s2", ids[:5] + ids[40:45])
    freeze(conn, "gold", ids[5:8])
    freeze(conn, "s1", ids[8:10])

    fit = fit_human(conn, minimum=30)

    assert (fit.report.relevant, fit.report.irrelevant) == (40 - 8, 40 - 5)
    assert fit.recipe["excludes_slices"] == ["s2", "gold"]


def test_cli_scorer_report_names_the_population_and_n(tmp_path: Path) -> None:
    env = environment(tmp_path)
    conn = open_database(env["INFOVORE_DB_PATH"])
    migrate(conn)
    a, b, c, d = labeled_four(conn)
    derive(conn, "local-model", 1, {a: 0.9, b: 0.6, c: 0.7, d: 0.1})
    freeze(conn, "s2", [a, d])
    conn.close()

    code, out, _ = run(["triage", "--human-report", "--scorer", "local-model"], env)
    assert code == 0
    assert "population: held-out (s2, gold) n=2" in out
    assert "evaluated=2" in out

    code, out, _ = run(
        ["triage", "--human-report", "--scorer", "local-model", "--include-training"], env
    )
    assert "population: all human labels (includes training) n=4" in out


def _channels(conn: sqlite3.Connection) -> None:
    for cid, name, parent in ((1, "tech", None), (2, "food", None), (3, "food-thread", 2)):
        conn.execute(
            "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
            " VALUES (?, 9, ?, ?, 'text')",
            (cid, parent, name),
        )


def test_excluded_channel_labels_never_train_bayes(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ids = seed_labeled(conn, 10, 10)
    _channels(conn)
    conn.execute("UPDATE exchanges SET channel_id = 2 WHERE id IN (?, ?, ?)", ids[10:13])
    conn.execute("UPDATE exchanges SET channel_id = 3 WHERE id = ?", (ids[13],))

    excluded = frozenset({"food"})
    assert len(training_labels(conn)[0]) == 20
    assert len(training_labels(conn, exclude_channels=excluded)[0]) == 16
    fit = fit_human(conn, minimum=6, exclude_channels=excluded)
    assert (fit.report.relevant, fit.report.irrelevant) == (10, 6)
    with pytest.raises(InsufficientHumanLabelsError):
        fit_human(conn, minimum=7, exclude_channels=excluded)
