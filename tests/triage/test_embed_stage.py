import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from infovore.config import ConfigError
from infovore.db.connection import migrate, open_database
from infovore.triage import embed_backend
from infovore.triage.cascade import EmbedStage, run_cascade
from infovore.triage.embed import EmbeddingCache
from infovore.triage.embed_stage import (
    build_embed_stage,
    default_cache_path,
    fit_embed_stage,
    irrelevant_threshold,
)
from infovore.triage.lexicon import load_lexicon
from tests.triage.test_embed import FakeEmbedder
from tests.triage.test_human import seed, seed_labeled


def labelled(tmp_path: Path, relevant: int, irrelevant: int) -> sqlite3.Connection:
    conn = open_database(tmp_path / "infovore.db")
    migrate(conn)
    seed_labeled(conn, relevant, irrelevant)
    return conn


def test_the_irrelevant_threshold_spares_the_recall_floor_of_relevant_scores() -> None:
    scores = [i / 100 for i in range(100)]

    t = irrelevant_threshold(scores, 0.95)

    assert sum(1 for s in scores if s < t) == 5
    assert irrelevant_threshold([0.4, 0.6], 0.95) == 0.4


def test_the_cache_sits_beside_the_database() -> None:
    assert default_cache_path(Path("/x/y/infovore.db")) == Path("/x/y/embed-cache.db")


def test_a_fitted_stage_keeps_relevant_recall_and_records_its_recipe(tmp_path: Path) -> None:
    conn = labelled(tmp_path, 20, 20)
    cache = EmbeddingCache(tmp_path / "c.db")

    stage = fit_embed_stage(conn, FakeEmbedder(), cache, frozenset())

    assert stage.t_irrelevant <= stage.t_relevant
    assert stage.recipe["labels"] == {"relevant": 20, "irrelevant": 20}
    thresholds = stage.recipe["thresholds"]
    assert isinstance(thresholds, dict)
    assert thresholds["relevant_recall_floor"] == 0.95
    assert thresholds["irrelevant_below"] == stage.t_irrelevant
    assert stage.recipe["pool"] == "first"
    ids = [row[0] for row in conn.execute("SELECT id FROM exchanges ORDER BY id")]
    probabilities = stage.score(ids)
    assert set(probabilities) == set(ids)
    assert all(0.0 <= p <= 1.0 for p in probabilities.values())


def _no_load(model: str, revision: str) -> FakeEmbedder:
    raise AssertionError("must not load a model without labels")


def test_a_database_with_too_few_labels_gets_an_abstaining_stage(tmp_path: Path) -> None:
    conn = labelled(tmp_path, 2, 20)

    stage = build_embed_stage(conn, frozenset(), tmp_path / "c.db", loader=_no_load)

    assert "need 5 trainable labels per class" in stage.abstain_reason
    assert "relevant=2 irrelevant=20" in stage.abstain_reason


def test_building_a_stage_loads_the_pinned_model_and_fits(tmp_path: Path) -> None:
    conn = labelled(tmp_path, 10, 10)
    seen: list[tuple[str, str]] = []

    def load(model: str, revision: str) -> FakeEmbedder:
        seen.append((model, revision))
        return FakeEmbedder()

    stage = build_embed_stage(conn, frozenset(), tmp_path / "c.db", loader=load)

    assert seen == [("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")]
    assert stage.abstain_reason == ""


def test_a_missing_embed_extra_fails_loudly_with_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = labelled(tmp_path, 10, 10)

    def missing(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(embed_backend, "import_module", missing)

    with pytest.raises(ConfigError, match=r"uv run --extra embed"):
        build_embed_stage(conn, frozenset(), tmp_path / "c.db")


def test_the_cascade_sends_only_lexicon_abstentions_to_the_embed_stage(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "infovore.db")
    migrate(conn)
    hit = seed(conn, 1, "my scsi disk will not boot")
    miss = seed(conn, 2, "nice weather, hello")
    asked: list[list[int]] = []

    def score(ids: Sequence[int]) -> dict[int, float]:
        asked.append(list(ids))
        return {miss: 0.01}

    stage = EmbedStage(score, 0.2, 0.9, {})

    outcomes = run_cascade(conn, [hit, miss], load_lexicon(), 0.3, stage)

    assert asked == [[miss]]
    assert [(o.stage, o.decision) for o in outcomes] == [
        ("lexicon", "relevant"),
        ("embed", "irrelevant"),
    ]
