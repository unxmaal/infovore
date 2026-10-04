import io
import sqlite3
from pathlib import Path

import pytest

from infovore.claims.check import (
    DEFAULT_THRESHOLD,
    Fact,
    Scored,
    check_claim,
    confusion,
    extract_facts,
    flag_at,
    flagged,
    normalize,
    overlap,
    rates,
    tune,
)
from infovore.cli import ExitCode, main
from infovore.db.claim_checks import CheckRow, current_checks, record_checks
from infovore.db.claims_v2 import ClaimIn, ExchangeOutcome, record_exchange, record_review
from infovore.triage.lexicon import load_lexicon
from tests.claims.seed import conversation, db, environment
from tests.claims.test_store import AT, make_run

LEXICON = load_lexicon()
GAZ = LEXICON.gazetteer


def facts(text: str) -> set[tuple[str, str]]:
    return {(f.kind, f.value) for f in extract_facts(text, GAZ)}


def verdict(statement: str, *sources: str, threshold: float = DEFAULT_THRESHOLD) -> str:
    return check_claim(statement, sources, LEXICON, threshold).verdict


def test_normalize_spellings() -> None:
    assert normalize("An R12k, 1,024 MB; <@123> user-ab12 \u201cx\u201d it\u2019s") == (
        'an r12000, 1024 mb; "x" it\'s'
    )
    assert normalize("three disks, one kilowatt, one of them, $50k, O350") == (
        "3 disks, 1 kilowatt, one of them, $50000, origin 350"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("runs at 400MHz", ("quantity", "400 mhz")),
        ("has 1,024 megabytes", ("quantity", "1024 mb")),
        ("one kilowatt", ("quantity", "1 kw")),
        ("IRIX 6.5.22m", ("version", "6.5.22")),
        ("an R12k CPU", ("cpu", "r12000")),
        ("in 2019", ("year", "2019")),
        ("part 030-1234-001", ("part", "030-1234-001")),
        ("an Origin   350", ("model", "origin 350")),
        ("see `inst.log`", ("file", "inst.log")),
        ('says "no such file"', ("quote", "no such file")),
        ("quad 700", ("number", "700")),
    ],
)
def test_fact_kinds(text: str, expected: tuple[str, str]) -> None:
    assert expected in facts(text)


def test_facts_are_not_double_counted() -> None:
    assert facts("6.5 GHz") == {("quantity", "6.5 ghz")}
    assert facts("the Origin 2000 in 2001") == {("model", "origin 2000"), ("year", "2001")}
    assert facts("user-ab12 said hi") == set()


def test_supported_when_every_fact_is_present() -> None:
    assert (
        verdict("The O2 runs at 400 MHz with an R12000", "my o2 is 400mhz with an r12k cpu")
        == "supported"
    )


@pytest.mark.parametrize(
    ("statement", "source", "kind"),
    [
        ("The O350 has quad 700 CPUs", "O350 with quad 600 cpus", "number"),
        ("It draws one kilowatt", "it draws a kW", "quantity"),
        ("Runs IRIX 6.5.22", "runs irix 6.5.30 here", "version"),
        ("It has an R10k", "it has an r12k", "cpu"),
        ("Built in 2019", "built in 2018 for sure", "year"),
        ("Part 030-1234-001", "part 030-1234-002", "part"),
        ("An Indy", "an indigo2 box", "model"),
        ("see `fx.log`", "see fx.txt", "file"),
        ('says "kernel panic"', "no panic here", "quote"),
    ],
)
def test_unsupported_fact_names_the_failing_fact(statement: str, source: str, kind: str) -> None:
    checked = check_claim(statement, [source], LEXICON, DEFAULT_THRESHOLD)
    assert checked.verdict == "unsupported_fact"
    assert kind in [f.kind for f in checked.missing()]


def test_lenient_matches() -> None:
    def missing(statement: str, source: str) -> list[Fact]:
        return check_claim(statement, [source], LEXICON, 0.0).missing()

    assert missing("IRIX 6.5", "irix 6.5.22 works") == []
    assert missing("an Octane", "my octane2 box") == []
    assert missing("an Octane2", "my octane box") == []
    assert missing("see `fx.log`", "see FX.LOG") == []
    assert missing("has 15 disks", "15 disks") == []


def test_low_overlap_and_uncheckable() -> None:
    assert verdict("zebra giraffe lemur", "the indy is nice") == "low_overlap"
    assert verdict("the machine is nice", "the machine is nice") == "uncheckable"
    assert verdict("the machine is nice") == "low_overlap"


def test_overlap_edges() -> None:
    assert overlap("said user-ab12", ["x"], LEXICON) == 1.0
    assert overlap("kernel zebra", ["kernel"], LEXICON) == pytest.approx(2 / 3)
    assert overlap("disks", ["a disk"], LEXICON) == 1.0


def test_flag_and_rates() -> None:
    assert flagged("low_overlap") and flagged("unsupported_fact")
    assert not flagged("supported") and not flagged("uncheckable")
    assert flag_at(Scored(True, 1.0, "unsupported_fact"), 0.0)
    assert flag_at(Scored(False, 0.2, "low_overlap"), 0.3)
    assert not flag_at(Scored(False, 0.4, "supported"), 0.3)
    items = [
        (Scored(True, 1.0, "unsupported_fact"), "made_up"),
        (Scored(False, 0.2, "low_overlap"), "wrong"),
        (Scored(False, 0.9, "supported"), "wrong"),
        (Scored(False, 0.2, "low_overlap"), "good"),
        (Scored(False, 0.9, "supported"), "good"),
        (Scored(False, 0.9, "supported"), "not_useful"),
    ]
    r = rates(items, 0.3)
    assert (r.caught, r.positives, r.false_alarms, r.negatives, r.flags) == (2, 3, 1, 2, 3)
    assert r.precision == pytest.approx(2 / 3)
    assert r.recall == pytest.approx(2 / 3)
    assert r.f1 == pytest.approx(2 / 3)
    assert tune(items).threshold == 0.95


def test_rates_degenerate() -> None:
    r = rates([], 0.5)
    assert (r.precision, r.recall, r.f1) == (0.0, 0.0, 0.0)


def test_confusion_table() -> None:
    table = confusion([("supported", "good"), ("supported", "good"), ("low_overlap", "wrong")])
    lines = table.splitlines()
    assert lines[0].split() == ["check\\eric", "good", "not_useful", "wrong", "made_up"]
    assert lines[1].split() == ["supported", "2", "0", "0", "0"]
    assert lines[3].split() == ["low_overlap", "0", "0", "1", "0"]


def test_fact_is_hashable_data() -> None:
    assert Fact("year", "2019") == Fact("year", "2019")


def test_store_is_append_only_and_current_view(tmp_path: Path) -> None:
    conn, claim_id = seeded(tmp_path)
    record_checks(conn, [CheckRow(claim_id, "low_overlap", 0.1, [])], {"r": 1}, AT)
    facts_ = [{"kind": "year", "value": "2019", "found": False}]
    record_checks(conn, [CheckRow(claim_id, "unsupported_fact", 0.2, facts_)], {"r": 2}, AT)
    stored = current_checks(conn, 1)
    assert stored[claim_id] == CheckRow(claim_id, "unsupported_fact", 0.2, facts_)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE claim_checks SET overlap = 0")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM claim_checks")


def seeded(tmp_path: Path) -> tuple[sqlite3.Connection, int]:
    conn = db(tmp_path)
    run = make_run(conn)
    assert run == 1
    e1, ids = conversation(
        conn,
        [(1, "ann", "my O2 is 400mhz"), (2, "bob", "ok")],
        1,
    )
    claims = [
        ClaimIn("u", "The O2 runs at 400 MHz", (ids[0],)),
        ClaimIn("u", "The O2 runs at 500 MHz", (ids[0],)),
        ClaimIn("u", "Lamborghini zebra", (ids[1],)),
    ]
    record_exchange(conn, run, e1, ExchangeOutcome("ok", None, 1, 1, 1, 1.0), claims, [])
    return conn, 1


def run_cli(tmp_path: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["claims", *argv], environ=environment(tmp_path), dotenv_path=None, stdout=out, stderr=err
    )
    return code, out.getvalue(), err.getvalue()


def test_check_command_dry_then_write_then_show(tmp_path: Path) -> None:
    conn, _ = seeded(tmp_path)
    record_review(conn, 1, "good", AT)
    record_review(conn, 2, "wrong", AT)
    conn.close()

    code, out, _ = run_cli(tmp_path, "check", "--run", "1")
    assert code == ExitCode.OK
    assert "supported 1, unsupported_fact 1, low_overlap 1, uncheckable 0" in out
    assert "2\tunsupported_fact" in out and "quantity:500 mhz\twrong" in out
    assert "3\tlow_overlap" in out and "\t-\t-\n" in out
    assert "not written" in out and "2 reviewed claims" in out
    assert "caught 1 of 1 wrong+made_up, false alarms 0 of 1 good" in out
    assert "tuned threshold" in out

    code, out, _ = run_cli(tmp_path, "show", "--run", "1", "--check")
    assert code == ExitCode.OK
    assert out.count("\t-\t") == 3

    code, out, _ = run_cli(tmp_path, "check", "--run", "1", "--write", "--threshold", "0.2")
    assert "written\n" in out

    code, out, _ = run_cli(tmp_path, "show", "--run", "1", "--check")
    lines = out.splitlines()
    assert lines[0].startswith("1\tsupported\toverlap 0.67\t")
    assert lines[1].endswith("\tmissing quantity:500 mhz")
    assert lines[2].startswith("3\tlow_overlap")
    assert "missing" not in lines[2]


def test_check_command_unknown_run(tmp_path: Path) -> None:
    seeded(tmp_path)[0].close()
    code, _, err = run_cli(tmp_path, "check", "--run", "9")
    assert code == ExitCode.CONFIG and "unknown run 9" in err
