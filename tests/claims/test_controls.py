import json
import random
import sqlite3
from pathlib import Path

import pytest

from infovore.claims.controls import (
    Stratum,
    _length_bucket,
    drop_rate_by_verdict,
    matched_recall,
    overlap_floor,
    read_drops,
    shuffled_citations,
    worst,
)
from infovore.cli import ExitCode
from infovore.db.claims_v2 import ClaimIn, ExchangeOutcome, record_exchange, review_rows
from infovore.db.wiki_articles import create_article_run, record_section
from infovore.db.wiki_tags import create_tag_run
from infovore.triage.lexicon import load_lexicon
from tests.claims.seed import NOW, conversation, db
from tests.claims.test_check import run_cli
from tests.claims.test_store import make_run

LEXICON = load_lexicon()


def _row(**fields: object) -> str:
    return json.dumps({"topic": "T", "section": "General", **fields}) + "\n"


def seeded(tmp_path: Path) -> sqlite3.Connection:
    conn = db(tmp_path)
    run = make_run(conn)
    e1, ids = conversation(
        conn, [(1, "ann", "my O2 is 400mhz"), (2, "bob", "ok"), (3, "cy", "Indy has 24-bit")], 1
    )
    e2, more = conversation(conn, [(4, "dee", "the Octane needs a V12 for that")], 2)
    record_exchange(
        conn,
        run,
        e1,
        ExchangeOutcome("ok", None, 1, 1, 1, 1.0),
        [
            ClaimIn("u", "The O2 runs at 400 MHz", (ids[0],)),
            ClaimIn("u", "Indy has 24-bit color", (ids[2],)),
        ],
        [],
    )
    record_exchange(
        conn,
        run,
        e2,
        ExchangeOutcome("ok", None, 1, 1, 1, 1.0),
        [ClaimIn("u", "Octane needs a V12", (more[0],))],
        [],
    )
    conn.commit()
    return conn


def test_shuffled_citations_score_worse_than_real_ones(tmp_path: Path) -> None:
    rows = review_rows(seeded(tmp_path), 1)

    real, shuffled, within = shuffled_citations(rows, LEXICON, 0.35, random.Random(0))

    assert real.n == shuffled.n == 3
    assert real.mean_overlap > shuffled.mean_overlap
    assert real.rate("supported") >= shuffled.rate("supported")
    assert within == pytest.approx(2 / 3)


def test_overlap_floor_reports_quantiles_and_shares(tmp_path: Path) -> None:
    rows = review_rows(seeded(tmp_path), 1)

    floor = overlap_floor(rows, LEXICON, 0.35, random.Random(1))

    assert floor.n == 3 and set(floor.quantiles) == {"p50", "p90", "p99"}
    assert 0.0 <= floor.above_threshold <= floor.above_floor <= 1.0
    empty = overlap_floor([], LEXICON, 0.35, random.Random(1))
    assert empty.n == 0 and empty.quantiles == {"p50": 0.0, "p90": 0.0, "p99": 0.0}


def test_worst_verdict_orders_by_severity() -> None:
    assert worst(["supported", "low_overlap"]) == "low_overlap"
    assert worst(["uncheckable", "unsupported_fact"]) == "unsupported_fact"
    assert worst([None, "supported"]) == "unchecked"
    assert worst([]) == "unchecked"


def test_drop_rate_reads_the_log_and_buckets_by_worst_verdict(tmp_path: Path) -> None:
    log = tmp_path / "drops.jsonl"
    log.write_text(
        json.dumps({"topic": "T", "section": "General", "error": "boom"})
        + "\n"
        + json.dumps(
            {"topic": "T", "section": "General", "text": "x", "cited": ["A"], "reason": "r"}
        )
        + "\n"
        + json.dumps(
            {"topic": "T", "section": "General", "text": "y", "cited": ["B", "A"], "reason": "r"}
        )
        + "\n"
    )
    sections = {"T": {"General": [("kept one", [1]), ("kept two", [2, 3])]}}
    verdict_of = {1: "supported", 2: "supported", 3: "uncheckable"}

    rates = drop_rate_by_verdict(sections, read_drops(log), verdict_of, {"A": 1, "B": 3})

    assert rates == {"uncheckable": (1, 1), "supported": (1, 1)}


def test_matched_recall_strata_split_cited_from_uncited(tmp_path: Path) -> None:
    conn = seeded(tmp_path)

    strata = matched_recall(conn, [1], LEXICON, random.Random(0), sample=10)

    assert all(isinstance(s, Stratum) for s in strata)
    assert sum(s.cited for s in strata) == 3 and sum(s.uncited for s in strata) == 1
    assert {s.position for s in strata} <= {"first", "middle", "last"}
    assert matched_recall(conn, [1], LEXICON, random.Random(0), sample=1)


def test_controls_command_prints_every_table(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    tag_run = create_tag_run(
        conn, endpoint="e", model_alias="m", model_id="m", prompt_hash="p", now=NOW
    )
    article = create_article_run(
        conn,
        endpoint="e",
        model_alias="m",
        model_id="m",
        prompt_hash="p",
        tag_run_id=tag_run,
        now=NOW,
    )
    record_section(conn, article, "T", "General", [("kept", [1])], 1)
    conn.commit()
    conn.close()
    log = tmp_path / "drops.jsonl"
    log.write_text(
        json.dumps(
            {
                "topic": "T",
                "section": "General",
                "text": "x",
                "cited": ["Indy has 24-bit color"],
                "reason": "low_overlap",
            }
        )
        + "\n"
    )

    code, out, _ = run_cli(tmp_path, "controls", "--run", "1", "--sample", "2")
    assert code == ExitCode.OK
    assert "controls: runs=1 claims=3 sample=2" in out
    assert "shuffled citations" in out and "random-pair overlap floor: n=2" in out
    assert "lexicon hit rate" in out and "skipped (needs --article-run" in out

    code, out, _ = run_cli(
        tmp_path, "controls", "--run", "1", "--article-run", str(article), "--drops-log", str(log)
    )
    assert code == ExitCode.OK and "writer drop rate by the worst verdict" in out
    assert "unchecked: dropped 1/2 (0.500)" in out

    code, out, _ = run_cli(tmp_path, "controls", "--run", "1", "--json")
    assert code == ExitCode.OK and json.loads(out)["claims"] == 3


@pytest.mark.parametrize(
    "argv",
    [
        ["--run", "9"],
        ["--run", "1", "--article-run", "1"],
        ["--run", "1", "--sample", "1"],
        ["--run", "1", "--article-run", "1", "--drops-log", "/nonexistent/x.jsonl"],
    ],
)
def test_controls_bad_arguments_are_config_errors(tmp_path: Path, argv: list[str]) -> None:
    seeded(tmp_path).close()

    code, _, _ = run_cli(tmp_path, "controls", *argv)

    assert code == ExitCode.CONFIG


def test_length_buckets_cover_every_length() -> None:
    assert [_length_bucket(n) for n in (0, 29, 30, 79, 80, 199, 200, 5000)] == [
        "0-29",
        "0-29",
        "30-79",
        "30-79",
        "80-199",
        "80-199",
        "200+",
        "200+",
    ]
