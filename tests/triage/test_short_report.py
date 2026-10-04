from pathlib import Path

import pytest

from infovore.cli import ExitCode
from infovore.rows import Label
from infovore.triage.short_report import (
    PRECISION,
    Cutoff,
    Row,
    best_cutoff,
    bucket_of,
    bucket_table,
    cutoff_line,
)
from tests.triage.test_cascade import NOW, build
from tests.triage.test_command import run


def test_buckets_cover_every_length() -> None:
    assert [bucket_of(n) for n in (0, 99, 100, 199, 200, 499, 500, 999, 1000, 5000)] == [
        "<100",
        "<100",
        "100-200",
        "100-200",
        "200-500",
        "200-500",
        "500-1000",
        "500-1000",
        "1000+",
        "1000+",
    ]


def test_the_table_splits_build_and_held_out_labels() -> None:
    rows = [Row(1, 50, 0), Row(2, 60, 0), Row(3, 70, 0), Row(4, 80, 2), Row(5, 5000, 0)]
    labels = {1: Label.NOISE, 2: Label.LORE, 3: Label.NOISE, 5: Label.LORE}
    lines = {(b.bucket, b.tech): b for b in bucket_table(rows, labels, {3})}

    short = lines[("<100", False)]
    assert (short.corpus, short.relevant, short.irrelevant) == (3, 1, 1)
    assert (short.held_relevant, short.held_irrelevant) == (0, 1)
    assert short.percent_irrelevant == 50.0
    assert lines[("<100", True)].corpus == 1
    assert lines[("<100", True)].percent_irrelevant is None
    assert lines[("1000+", False)].relevant == 1
    assert len(lines) == 10


def test_the_cutoff_is_the_largest_that_holds_precision() -> None:
    rows = [Row(i, 10 + i, 0) for i in range(1, 41)] + [Row(99, 500, 1)]
    labels = {i: Label.NOISE for i in range(1, 41)}
    labels[5] = Label.LORE
    labels[6] = Label.LORE
    labels[99] = Label.LORE
    held = {40}

    cutoff = best_cutoff(rows, labels, held)

    assert cutoff is None
    clean = {i: Label.NOISE for i in range(1, 41)}
    cutoff = best_cutoff(rows, clean, held)
    assert cutoff is not None
    assert (cutoff.limit, cutoff.n, cutoff.precision) == (50, 39, 1.0)
    assert PRECISION == 0.95
    assert best_cutoff(rows[:5], clean, set()) is None


def test_the_command_prints_the_table(tmp_path: Path) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = run(["relevance", "short-report"], env)

    assert code == ExitCode.OK
    assert "no tech words" in out and "tech words" in out
    assert "<100" in out and "1000+" in out
    assert "no cutoff qualifies" in out


def test_the_cutoff_line_names_the_result_or_the_reason() -> None:
    assert cutoff_line(Cutoff(120, 0.97, 33)) == "cutoff L=120 precision=0.970 n=33\n"
    assert "no cutoff qualifies" in cutoff_line(None)


def test_the_stage_decides_short_conversations_without_tech_words(tmp_path: Path) -> None:
    from infovore.triage.cascade import EmbedStage, run_cascade, write_outcomes
    from infovore.triage.lexicon import load_lexicon

    _, conn = build(tmp_path)
    ids = [1, 4, 5, 7]

    outcomes = run_cascade(
        conn, ids, load_lexicon(), 0.9, EmbedStage.abstaining("x"), frozenset(), short_limit=17
    )

    stages = {o.exchange_id: (o.stage, o.decision) for o in outcomes}
    assert stages[4] == ("short_no_tech", "irrelevant")
    assert stages[5] == ("residue", "residue")
    assert stages[1][0] == "lexicon"
    assert stages[7][0] == "residue"
    write_outcomes(
        conn, outcomes, load_lexicon(), 0.5, EmbedStage.abstaining("x"), NOW, short_limit=17
    )
    row = conn.execute(
        "SELECT label, recipe_json FROM annotations WHERE scorer = 'relevance_short_no_tech'"
    ).fetchone()
    assert row["label"] == "irrelevant" and '"short_limit": 17' in row["recipe_json"]


def test_no_limit_means_no_stage(tmp_path: Path) -> None:
    from infovore.triage.cascade import EmbedStage, run_cascade
    from infovore.triage.lexicon import load_lexicon

    _, conn = build(tmp_path)

    outcomes = run_cascade(conn, [4], load_lexicon(), 0.5, EmbedStage.abstaining("x"), frozenset())

    assert outcomes[0].stage == "residue"


def test_the_limit_comes_from_the_labels_and_is_none_when_none_qualifies(tmp_path: Path) -> None:
    from infovore.triage.lexicon import load_lexicon
    from infovore.triage.short_report import short_limit

    _, conn = build(tmp_path)

    assert short_limit(conn, load_lexicon(), 0.5, frozenset()) is None


def test_the_cascade_reports_the_limit_it_chose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("infovore.triage.relevance_command.short_limit", lambda *a: 17)
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = run(["relevance", "cascade", "--slices", "s1"], env)

    assert code == ExitCode.OK
    assert "short_no_tech limit=17" in out
    assert "stage short_no_tech: decided=" in out
