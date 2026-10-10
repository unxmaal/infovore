from pathlib import Path

from infovore.db.connection import migrate, open_database
from infovore.reputation.compare import compare, embed_scores
from infovore.rows import Label
from tests.triage.test_embed import FakeEmbedder
from tests.triage.test_human import seed, seed_labeled

LORE, NOISE = Label.LORE, Label.NOISE


def test_embed_reputation_and_their_rank_average_are_scored_on_shared_exchanges() -> None:
    labels = {1: NOISE, 2: NOISE, 3: LORE, 4: LORE, 5: LORE}
    embed = {1: 0.1, 2: 0.2, 3: 0.8, 4: 0.9}
    reputation = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0, 6: 9.0}

    result = compare(embed, reputation, labels, seed=0)

    assert result.embed.auc == 1.0
    assert result.reputation.auc == 0.0
    assert result.combined.auc == 0.5
    assert result.combined.n == 4


def test_no_shared_exchanges_leave_every_auc_undefined() -> None:
    result = compare({}, {1: 1.0}, {1: LORE}, seed=0)

    assert (result.embed.auc, result.reputation.auc, result.combined.auc) == (None, None, None)


def test_too_few_labels_for_the_embedding_give_no_scores(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "infovore.db")
    migrate(conn)
    seed_labeled(conn, 2, 20)

    def no_load(model: str, revision: str) -> FakeEmbedder:
        raise AssertionError("must not load a model")

    assert embed_scores(conn, frozenset(), tmp_path / "c.db", [1], loader=no_load) is None


def test_held_out_scores_come_from_the_fitted_head_and_the_rest_are_out_of_fold(
    tmp_path: Path,
) -> None:
    conn = open_database(tmp_path / "infovore.db")
    migrate(conn)
    ids = seed_labeled(conn, 10, 10)
    extra = seed(conn, 100, "my Indy runs IRIX 6.5.22 and hinv shows it")
    conn.commit()
    seen: list[tuple[str, str]] = []

    def load(model: str, revision: str) -> FakeEmbedder:
        seen.append((model, revision))
        return FakeEmbedder()

    scores = embed_scores(conn, frozenset(), tmp_path / "c.db", [extra], loader=load)

    assert scores is not None
    assert len(seen) == 1
    assert set(scores.held_out) == {extra}
    assert set(scores.out_of_fold) == set(ids)
    assert all(0.0 <= p <= 1.0 for p in scores.out_of_fold.values())
