from collections.abc import Sequence

import pytest

from infovore.reputation import stats
from infovore.rows import Label

LORE, NOISE = Label.LORE, Label.NOISE
CLEAN = [(1.0, NOISE), (2.0, NOISE), (3.0, LORE), (4.0, LORE)]


def test_a_perfect_separation_has_auc_one_and_an_interval_inside_the_unit_range() -> None:
    first = stats.auc_interval(CLEAN, seed=0)

    assert first == stats.auc_interval(CLEAN, seed=0)
    assert (first.n, first.relevant, first.irrelevant, first.auc) == (4, 2, 2, 1.0)
    assert first.low is not None and first.high is not None
    assert 0.0 <= first.low <= 1.0 <= first.high <= 1.0


def test_a_noisy_ranking_gets_an_interval_around_its_auc() -> None:
    scored = [(float(i), NOISE if i % 3 else LORE) for i in range(30)]

    for seed in range(3):
        result = stats.auc_interval(scored, seed=seed, resamples=200)
        assert result.auc is not None and result.low is not None and result.high is not None
        assert 0.0 <= result.low <= result.auc <= result.high <= 1.0


def test_one_class_has_no_auc_or_interval() -> None:
    result = stats.auc_interval([(1.0, LORE), (2.0, LORE)], seed=0)

    assert (result.n, result.relevant, result.irrelevant) == (2, 2, 0)
    assert (result.auc, result.low, result.high) == (None, None, None)


def test_no_usable_resample_leaves_the_interval_open(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = [0]

    def once(scored: Sequence[tuple[float, Label]]) -> float | None:
        calls[0] += 1
        return 0.75 if calls[0] == 1 else None

    monkeypatch.setattr(stats, "auc", once)

    result = stats.auc_interval(CLEAN, seed=0, resamples=10)

    assert (result.auc, result.low, result.high) == (0.75, None, None)


def test_the_percentile_interpolates_and_handles_a_single_value() -> None:
    assert stats.percentile([1.0], 97.5) == 1.0
    assert stats.percentile([0.0, 1.0], 50.0) == 0.5
    assert stats.percentile([0.0, 1.0, 2.0], 50.0) == 1.0


def test_youden_picks_the_separating_threshold_and_the_lowest_on_a_tie() -> None:
    assert stats.youden_threshold(CLEAN) == 3.0
    assert stats.youden_threshold([(1.0, NOISE), (2.0, LORE), (3.0, NOISE), (4.0, LORE)]) == 2.0
    assert stats.youden_threshold([(1.0, LORE), (2.0, LORE)]) is None
    assert stats.youden_threshold([]) is None


def test_accuracy_is_reported_against_the_majority_baseline() -> None:
    result = stats.accuracy_at(3.0, CLEAN)
    assert result == stats.Accuracy(n=4, threshold=3.0, accuracy=1.0, baseline=0.5)

    skewed = stats.accuracy_at(4.0, [(1.0, NOISE), (2.0, NOISE), (3.0, NOISE), (4.0, LORE)])
    assert skewed == stats.Accuracy(n=4, threshold=4.0, accuracy=1.0, baseline=0.75)

    wrong = stats.accuracy_at(10.0, [(1.0, LORE), (2.0, LORE), (3.0, NOISE)])
    assert wrong.accuracy == pytest.approx(1 / 3)
    assert wrong.baseline == pytest.approx(2 / 3)
    assert (wrong.accuracy, wrong.baseline) == (pytest.approx(1 / 3), pytest.approx(2 / 3))


def test_accuracy_needs_a_threshold_and_data() -> None:
    assert stats.accuracy_at(None, CLEAN) is None
    assert stats.accuracy_at(1.0, []) is None


def test_ranks_average_over_ties() -> None:
    assert stats.average_ranks([3.0, 1.0, 2.0, 1.0]) == [4.0, 1.5, 3.0, 1.5]
    assert stats.average_ranks([]) == []


def test_rank_average_blends_two_orderings() -> None:
    assert stats.rank_average([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == [2.0, 2.0, 2.0]
    with pytest.raises(ValueError):
        stats.rank_average([1.0, 2.0], [1.0])


def test_spearman_is_a_rank_correlation() -> None:
    assert stats.spearman([1.0, 2.0, 3.0], [10.0, 20.0, 400.0]) == pytest.approx(1.0)
    assert stats.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert stats.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert stats.spearman([1.0], [1.0]) is None
    with pytest.raises(ValueError):
        stats.spearman([1.0, 2.0], [1.0])
