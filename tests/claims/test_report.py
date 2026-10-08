import io
from pathlib import Path

from infovore.cli import ExitCode, main
from infovore.db.claims_v2 import (
    ClaimIn,
    ExchangeOutcome,
    Rejection,
    record_exchange,
    record_review,
)
from tests.claims.seed import conversation, db, environment
from tests.claims.test_store import AT, make_run


def report(tmp_path: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["claims", "report", *argv],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


def populate(tmp_path: Path) -> int:
    conn = db(tmp_path)
    run = make_run(conn)
    e1, i1 = conversation(conn, [(1, "ann", "a")], 1)
    e2, _ = conversation(conn, [(1, "ann", "b")], 2)
    ok = ExchangeOutcome("ok", None, 1, 100, 10, 3.0)
    claims = [ClaimIn("u", f"u said {n}", (i1[0],)) for n in range(4)]
    record_exchange(conn, run, e1, ok, claims, [])
    record_exchange(conn, run, e2, ok, [], [])
    for claim_id, verdict, interface in (
        (1, "good", "cited-only"),
        (2, "made_up", "conversation"),
        (3, "not_useful", "conversation"),
    ):
        record_review(conn, claim_id, verdict, AT, interface)
    conn.close()
    return run


def test_report_prints_the_trial_numbers_for_a_run(tmp_path: Path) -> None:
    run = populate(tmp_path)

    code, out, _ = report(tmp_path, "--run", str(run))

    assert code == ExitCode.OK
    assert f"run {run}: real/4b (model_info) prompt abc" in out
    assert "conversations: 2 (failed 0)" in out
    assert "claims: 4 (2.00 per conversation, rejected 0)" in out
    assert "zero-claim conversations: 1 (50.0%)" in out
    assert "reviews: good 1, wrong 0, made_up 1, not_useful 1, unreviewed 1" in out
    assert "  cited-only: good 1, wrong 0, made_up 0, not_useful 0" in out
    assert "  conversation: good 0, wrong 0, made_up 1, not_useful 1" in out
    assert "made-up rate: 33.3% (1 of 3 reviewed)" in out
    assert "tokens: 200 in, 20 out" in out
    assert "seconds per conversation: 3.00" in out


def test_report_without_a_run_covers_every_run_and_empty_is_fine(tmp_path: Path) -> None:
    db(tmp_path).close()
    code, out, _ = report(tmp_path)
    assert code == ExitCode.OK and "no runs" in out

    populate(tmp_path)
    code, out, _ = report(tmp_path)
    assert out.count("run ") == 1


def test_report_on_a_run_with_nothing_reviewed_or_processed(tmp_path: Path) -> None:
    conn = db(tmp_path)
    run = make_run(conn)
    conn.close()

    _, out, _ = report(tmp_path, "--run", str(run))

    assert "made-up rate: n/a" in out and "seconds per conversation: n/a" in out
    assert "zero-claim conversations: 0 (n/a)" in out
    assert "claims: 0 (n/a per conversation" in out


def test_an_unknown_run_is_refused(tmp_path: Path) -> None:
    db(tmp_path).close()

    code, _, err = report(tmp_path, "--run", "9")

    assert code == ExitCode.CONFIG and "unknown run 9" in err


def show(tmp_path: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["claims", "show", *argv],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


def test_show_prints_claims_and_rejections_as_plain_text(tmp_path: Path) -> None:
    run = populate(tmp_path)
    conn = db(tmp_path)
    eid, _ = conversation(conn, [(1, "ann", "c")], 3)
    ok = ExchangeOutcome("ok", None, 1, 1, 1, 1.0)
    record_exchange(conn, run, eid, ok, [], [Rejection("user-x", "why?", "[1]", "empty statement")])
    conn.close()

    code, out, _ = show(tmp_path, "--run", str(run))
    assert code == ExitCode.OK
    assert out.splitlines()[0].endswith("u: u said 0")
    assert len(out.splitlines()) == 4 and "why?" not in out

    code, out, _ = show(tmp_path, "--run", str(run), "--rejected")
    assert code == ExitCode.OK
    assert out.splitlines() == ["user-x: why? [empty statement]"]


def test_show_rejects_an_unknown_run(tmp_path: Path) -> None:
    db(tmp_path).close()
    code, _, err = show(tmp_path, "--run", "99")
    assert code != ExitCode.OK and "unknown run 99" in err


def test_report_omits_interfaces_with_no_reviews(tmp_path: Path) -> None:
    conn = db(tmp_path)
    run = make_run(conn)
    conn.close()

    _, out, _ = report(tmp_path, "--run", str(run))

    assert "    cited-only:" not in out and "    conversation: good" not in out
