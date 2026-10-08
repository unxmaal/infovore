import io
import sqlite3
from pathlib import Path

import pytest

from infovore.claims.compare import Comparison, best_match, compare, format_comparison
from infovore.claims.value import tokens
from infovore.cli import ExitCode, main
from infovore.db.claims_v2 import ClaimIn, ExchangeOutcome, record_exchange, record_review
from tests.claims.seed import conversation, db, environment
from tests.claims.test_store import AT, make_run

OK = ExchangeOutcome("ok", None, 1, 1, 1, 1.0)
FAILED = ExchangeOutcome("failed", "boom", 0, 0, 0, 0.0)


def run_compare(tmp_path: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["claims", "compare", *argv],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


def populate(tmp_path: Path) -> sqlite3.Connection:
    conn = db(tmp_path)
    base = make_run(conn)
    a, _ = conversation(conn, [(1, "ann", "x")], 1)
    b, _ = conversation(conn, [(1, "ann", "x")], 2)
    c, _ = conversation(conn, [(1, "ann", "x")], 3)
    old = [
        ClaimIn("u", "O2 has a quiet fan unit", ()),
        ClaimIn("u", "I would love an Onyx someday", ()),
    ]
    record_exchange(conn, base, a, OK, old, [])
    record_exchange(conn, base, b, OK, [ClaimIn("u", "Octane uses a loud power supply", ())], [])
    record_exchange(conn, base, c, OK, [ClaimIn("u", "Indy has a slow boot disk", ())], [])
    ids = [r[0] for r in conn.execute("SELECT id FROM claims_v2 ORDER BY id")]
    for claim_id, verdict in zip(ids, ["good", "not_useful", "good", "wrong"], strict=True):
        record_review(conn, claim_id, verdict, AT)
    new = make_run(conn)
    fresh = [
        ClaimIn("u", "The O2 has a quiet fan unit", ()),
        ClaimIn("u", "IRIX 6.5.22 runs on the O2", ()),
    ]
    record_exchange(conn, new, a, OK, fresh, [])
    record_exchange(conn, new, b, OK, [], [])
    record_exchange(conn, new, c, FAILED, [], [])
    conn.commit()
    return conn


def test_best_match_is_max_jaccard_and_zero_without_overlap_or_tokens() -> None:
    sets = [tokens("O2 quiet fan"), tokens("Octane loud")]
    assert best_match("O2 quiet fan", sets) == 1.0
    assert best_match("O2 fan", sets) == pytest.approx(2 / 3)
    assert best_match("anything", []) == 0.0
    assert best_match("the a", sets) == 0.0
    assert best_match("the a", [frozenset()]) == 0.0


def test_compare_counts_reproduced_reviews_and_unmatched_new(tmp_path: Path) -> None:
    conn = populate(tmp_path)
    result = compare(conn, [1], 2, 0.5)
    assert result == Comparison(
        reviewed={"good": 2, "not_useful": 1},
        reproduced={"good": 1},
        new_claims=2,
        unmatched_new=1,
        exchanges=2,
    )


def test_format_reports_each_verdict_and_good_rate() -> None:
    lines = format_comparison(Comparison({"good": 2}, {"good": 1}, 3, 1, 2))
    assert lines[0] == "conversations 2, new claims 3, unmatched new 1"
    assert "  good: reproduced 1 of 2" in lines
    assert "  made_up: reproduced 0 of 0" in lines
    assert lines[-1].startswith("reproduced good rate 100.0% (")
    assert format_comparison(Comparison({}, {}, 0, 0, 0))[-1] == "reproduced good rate n/a"


def test_cli_prints_comparison(tmp_path: Path) -> None:
    populate(tmp_path).close()
    code, out, _ = run_compare(tmp_path, "--base-runs", "1", "--run", "2")
    assert code == ExitCode.OK
    assert "  not_useful: reproduced 0 of 1" in out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--base-runs", "x", "--run", "2"], "--base-runs must be comma-separated run ids"),
        (["--base-runs", "1,7", "--run", "2"], "unknown run 7"),
        (["--base-runs", "1", "--run", "9"], "unknown run 9"),
        (["--base-runs", "1", "--run", "2", "--threshold", "0"], "--threshold must be in (0, 1]"),
    ],
)
def test_cli_rejects_bad_arguments(tmp_path: Path, argv: list[str], message: str) -> None:
    populate(tmp_path).close()
    code, _, err = run_compare(tmp_path, *argv)
    assert code != ExitCode.OK
    assert message in err
