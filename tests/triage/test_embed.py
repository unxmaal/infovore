import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from infovore.cli import ExitCode
from infovore.db.connection import migrate, open_database
from infovore.rows import Label
from infovore.triage.embed import (
    EmbeddingCache,
    NotEnoughLabelsError,
    cross_validate,
    embed_exchanges,
    fit_head,
    pool_vectors,
    predict_head,
    render_exchange,
    residue_ids,
    stratified_folds,
    summarize,
    window_parts,
)
from infovore.triage.lexicon import load_lexicon
from tests.triage.test_command import environment, run
from tests.triage.test_human import LORE, db, message, seed, seed_labeled

KEYWORDS = ("irix", "prom", "hinv", "lol", "gg")


class FakeEmbedder:
    model_id = "fake/keywords"
    revision = "r1"

    def __init__(self, max_tokens: int = 512) -> None:
        self.calls: list[list[str]] = []
        self.max_tokens = max_tokens

    def token_count(self, text: str) -> int:
        return len(text.split())

    def split(self, text: str, limit: int) -> list[str]:
        words = text.split()
        return [" ".join(words[i : i + limit]) for i in range(0, len(words), limit)]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [
            [float(text.lower().count(word)) + 0.01 * len(text) for word in KEYWORDS]
            for text in texts
        ]


def labelled_db(tmp_path: Path, relevant: int, irrelevant: int) -> Path:
    path = tmp_path / "infovore.db"
    conn = open_database(path)
    migrate(conn)
    seed_labeled(conn, relevant, irrelevant)
    conn.close()
    return path


def test_render_joins_non_empty_messages_in_order_and_truncates() -> None:
    messages = [message("one", mid=1), message("  ", mid=2), message("two", mid=3)]

    assert render_exchange(messages, 100) == "one\ntwo"
    assert render_exchange(messages, 5) == "one\nt"
    assert render_exchange([], 5) == ""


def test_folds_are_stratified_deterministic_and_complete() -> None:
    labels = {i: Label.LORE if i <= 10 else Label.NOISE for i in range(1, 31)}

    folds = stratified_folds(labels, 5)

    assert folds == stratified_folds(labels, 5)
    assert set(folds) == set(labels)
    for fold in range(5):
        members = [eid for eid, f in folds.items() if f == fold]
        assert sum(1 for eid in members if labels[eid] is Label.LORE) == 2
        assert sum(1 for eid in members if labels[eid] is Label.NOISE) == 4


def test_head_separates_a_linear_problem() -> None:
    xs = [[float(i), 1.0] for i in range(10)] + [[float(i) + 20, 1.0] for i in range(10)]
    ys = [False] * 10 + [True] * 10

    head = fit_head(xs, ys)

    assert predict_head(head, [1.0, 1.0]) < 0.5 < predict_head(head, [25.0, 1.0])
    empty = fit_head([], [])
    assert empty.weights == []
    assert predict_head(empty, []) == 0.5


def test_summarize_reports_auc_best_f1_and_irrelevant_recall() -> None:
    scored = [(0.9, Label.LORE), (0.8, Label.LORE), (0.7, Label.NOISE), (0.2, Label.NOISE)]

    summary = summarize(scored)

    assert summary.relevant == 2
    assert summary.irrelevant == 2
    assert summary.auc == 1.0
    assert (summary.precision, summary.recall, summary.f1) == (1.0, 1.0, 1.0)
    assert summary.irrelevant_recall == 1.0


def test_summarize_with_one_class_has_no_auc_or_threshold() -> None:
    summary = summarize([(0.5, Label.NOISE)])

    assert summary.auc is None
    assert summary.f1 == 0.0
    assert summary.irrelevant_recall is None
    assert summarize([]).relevant == 0


def test_summarize_irrelevant_recall_needs_the_relevant_recall_floor() -> None:
    scored = [(0.9, Label.LORE), (0.4, Label.LORE), (0.6, Label.NOISE), (0.3, Label.NOISE)]

    assert summarize(scored, 0.5).irrelevant_recall == 1.0
    assert summarize(scored, 1.0).irrelevant_recall == 0.5


def test_cache_round_trips_and_keys_on_model_revision_and_hash(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "sub" / "c.db")
    cache.put("m", "r", 1, "h", [0.5, 1.5])

    assert cache.get("m", "r", 1, "h") == [0.5, 1.5]
    assert cache.get("m", "r2", 1, "h") is None
    assert cache.get("m", "r", 1, "other") is None
    assert cache.get("m2", "r", 1, "h") is None
    cache.put("m", "r", 1, "h", [2.0])
    assert cache.get("m", "r", 1, "h") == [2.0]
    cache.close()


def test_embedding_is_cached_so_reruns_do_not_call_the_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ids = seed_labeled(conn, 2, 2)
    cache = EmbeddingCache(tmp_path / "c.db")
    fake = FakeEmbedder()

    first = embed_exchanges(conn, ids, fake, cache, 100, batch_size=3)
    again = embed_exchanges(conn, ids, fake, cache, 100, batch_size=3)

    assert first == again
    assert sorted(first) == sorted(ids)
    assert [len(call) for call in fake.calls] == [3, 1]


def test_exchanges_without_text_are_not_embedded(tmp_path: Path) -> None:
    conn = db(tmp_path)
    blank = seed(conn, 1, "   ")
    real = seed(conn, 2, LORE)
    conn.commit()
    cache = EmbeddingCache(tmp_path / "c.db")

    assert list(embed_exchanges(conn, [blank, real], FakeEmbedder(), cache, 50)) == [real]


def test_residue_is_the_labelled_exchanges_the_lexicon_abstains_on(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ids = seed_labeled(conn, 1, 1)
    lexicon = load_lexicon()

    assert residue_ids(conn, ids, lexicon, 1.01, frozenset()) == set(ids)
    assert residue_ids(conn, ids, lexicon, 0.0, frozenset()) == {ids[1]}


def test_cross_validation_scores_both_models_on_the_same_folds(tmp_path: Path) -> None:
    conn = open_database(labelled_db(tmp_path, 10, 20))

    result = cross_validate(
        conn, FakeEmbedder(), EmbeddingCache(tmp_path / "c.db"), folds=5, max_chars=200
    )

    assert result.folds == 5
    assert result.all_labels.bayes.relevant == 10
    assert result.all_labels.embed.irrelevant == 20
    assert result.all_labels.embed.auc == 1.0
    assert result.all_labels.bayes.auc == 1.0
    assert result.residue is None
    assert result.recipe["model"] == "fake/keywords"
    assert result.recipe["revision"] == "r1"
    assert result.recipe["max_chars"] == 200


def test_cross_validation_residue_subset(tmp_path: Path) -> None:
    conn = open_database(labelled_db(tmp_path, 10, 20))

    result = cross_validate(
        conn,
        FakeEmbedder(),
        EmbeddingCache(tmp_path / "c.db"),
        folds=5,
        max_chars=200,
        residue=True,
    )

    assert result.residue is not None
    assert result.residue.embed.relevant + result.residue.embed.irrelevant == 30


def test_cross_validation_needs_a_label_per_fold_in_each_class(tmp_path: Path) -> None:
    conn = open_database(labelled_db(tmp_path, 2, 20))
    cache = EmbeddingCache(tmp_path / "c.db")

    with pytest.raises(NotEnoughLabelsError):
        cross_validate(conn, FakeEmbedder(), cache, folds=5, max_chars=9)


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    fake = FakeEmbedder()
    monkeypatch.setattr("infovore.triage.relevance_command.load_embedder", lambda *_: fake)
    return fake


def test_cli_compare_prints_the_table(tmp_path: Path, fake_backend: FakeEmbedder) -> None:
    labelled_db(tmp_path, 10, 20)

    code, out, _ = run(["relevance", "compare", "--cv", "5", "--residue"], environment(tmp_path))

    assert code == ExitCode.OK
    assert "model=fake/keywords revision=r1" in out
    assert "all labels" in out
    assert "residue" in out
    assert "naive_bayes" in out
    assert "embed_lr" in out
    assert (tmp_path / "embed-cache.db").exists()


def test_cli_compare_refuses_too_few_labels(tmp_path: Path, fake_backend: FakeEmbedder) -> None:
    labelled_db(tmp_path, 1, 1)

    code, _, err = run(["relevance", "compare", "--cv", "5"], environment(tmp_path))

    assert code == ExitCode.CONFIG
    assert "at least" in err


def test_cli_embed_score_writes_out_of_fold_annotations_for_labelled_only(
    tmp_path: Path, fake_backend: FakeEmbedder
) -> None:
    path = tmp_path / "infovore.db"
    conn = open_database(path)
    migrate(conn)
    ids = seed_labeled(conn, 10, 20)
    stray = seed(conn, 99, "unlabelled chatter")
    conn.commit()
    conn.close()
    env = environment(tmp_path)

    code, out, _ = run(["relevance", "embed-score", "--cv", "5"], env)

    assert code == ExitCode.OK
    assert "wrote 30" in out
    conn = open_database(path)
    rows = conn.execute(
        "SELECT subject_id, reproducibility, recipe_json FROM annotations"
        " WHERE scorer = 'p_relevant_embed'"
    ).fetchall()
    assert sorted(r["subject_id"] for r in rows) == sorted(ids)
    assert stray not in {r["subject_id"] for r in rows}
    assert {r["reproducibility"] for r in rows} == {"derived"}
    assert "fake/keywords" in rows[0]["recipe_json"]

    assert run(["relevance", "embed-score", "--cv", "5"], env)[0] == ExitCode.OK
    versions = conn.execute(
        "SELECT DISTINCT scorer_version FROM annotations WHERE scorer = 'p_relevant_embed'"
    ).fetchall()
    assert len(versions) == 2


def test_cli_embed_score_also_scores_held_out_labels(
    tmp_path: Path, fake_backend: FakeEmbedder
) -> None:
    path = tmp_path / "infovore.db"
    conn = open_database(path)
    migrate(conn)
    ids = seed_labeled(conn, 10, 20)
    conn.execute(
        "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
        " VALUES ('gold', ?, 1, 't', 0, '2026-01-01')",
        (ids[0],),
    )
    conn.commit()
    conn.close()

    code, out, _ = run(["relevance", "embed-score", "--cv", "5"], environment(tmp_path))

    assert code == ExitCode.OK
    assert "wrote 30" in out


def test_cli_compare_without_residue_skips_that_table(
    tmp_path: Path, fake_backend: FakeEmbedder
) -> None:
    labelled_db(tmp_path, 10, 20)

    code, out, _ = run(["relevance", "compare", "--cv", "5"], environment(tmp_path))

    assert code == ExitCode.OK
    assert "all labels" in out
    assert "residue" not in out


def test_windows_pack_whole_messages_up_to_the_token_limit() -> None:
    fake = FakeEmbedder(max_tokens=4)

    assert window_parts(["a b", "c", "d e f", "g"], fake) == ["a b\nc", "d e f\ng"]
    assert window_parts(["a b c d"], fake) == ["a b c d"]
    assert window_parts([], fake) == []


def test_a_single_oversized_message_is_split_and_neighbours_start_a_new_window() -> None:
    fake = FakeEmbedder(max_tokens=3)

    assert window_parts(["x", "a b c d e", "y"], fake) == ["x", "a b c", "d e\ny"]


def test_pooling_first_mean_max() -> None:
    vectors = [[1.0, 4.0], [3.0, 2.0]]

    assert pool_vectors(vectors, "first") == [1.0, 4.0]
    assert pool_vectors(vectors, "mean") == [2.0, 3.0]
    assert pool_vectors(vectors, "max") == [3.0, 4.0]
    with pytest.raises(ValueError, match="pool"):
        pool_vectors(vectors, "median")


def long_exchange_db(tmp_path: Path) -> tuple[sqlite3.Connection, int]:
    conn = db(tmp_path)
    eid = seed(conn, 1, " ".join(f"w{i}" for i in range(10)))
    conn.commit()
    return conn, eid


def test_windowed_pooling_embeds_every_window_and_pools(tmp_path: Path) -> None:
    conn, eid = long_exchange_db(tmp_path)
    fake = FakeEmbedder(max_tokens=4)
    cache = EmbeddingCache(tmp_path / "c.db")

    mean = embed_exchanges(conn, [eid], fake, cache, 5, pool="mean")[eid]
    biggest = embed_exchanges(conn, [eid], fake, cache, 5, pool="max")[eid]
    first = embed_exchanges(conn, [eid], fake, cache, 5, pool="first")[eid]

    assert fake.calls[0] == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]
    assert fake.calls[2] == ["w0 w1"]
    assert len(set(map(tuple, (mean, biggest, first)))) == 3
    assert biggest[0] == max(
        FakeEmbedder().embed([t])[0][0] for t in ("w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9")
    )


def test_cache_key_separates_pool_and_window_size(tmp_path: Path) -> None:
    conn, eid = long_exchange_db(tmp_path)
    cache = EmbeddingCache(tmp_path / "c.db")
    four = FakeEmbedder(max_tokens=4)

    embed_exchanges(conn, [eid], four, cache, 5, pool="mean")
    embed_exchanges(conn, [eid], four, cache, 5, pool="mean")
    assert len(four.calls) == 1
    embed_exchanges(conn, [eid], four, cache, 5, pool="max")
    assert len(four.calls) == 2
    embed_exchanges(conn, [eid], FakeEmbedder(max_tokens=3), cache, 5, pool="mean")
    embed_exchanges(conn, [eid], four, cache, 5, pool="first")
    rows = cache._conn.execute("SELECT DISTINCT model FROM embeddings").fetchall()
    assert len(rows) == 4
    assert "fake/keywords" in {r[0] for r in rows}


def test_cli_pool_is_recorded_in_header_and_recipe(
    tmp_path: Path, fake_backend: FakeEmbedder
) -> None:
    path = labelled_db(tmp_path, 10, 20)
    env = environment(tmp_path)

    code, out, _ = run(["relevance", "compare", "--cv", "5", "--pool", "mean"], env)
    assert code == ExitCode.OK
    assert "pool=mean" in out
    assert run(["relevance", "embed-score", "--cv", "5", "--pool", "max"], env)[0] == ExitCode.OK

    conn = open_database(path)
    recipe = conn.execute("SELECT recipe_json FROM annotations LIMIT 1").fetchone()[0]
    assert '"pool": "max"' in recipe
    assert "windows" in recipe
    assert run(["relevance", "compare", "--pool", "median"], env)[0] != ExitCode.OK
