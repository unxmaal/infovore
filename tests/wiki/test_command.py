import io
from pathlib import Path

from infovore.cli import ExitCode, main
from tests.claims.seed import environment
from tests.wiki.seed import add_claim, add_exchange, wiki_db


def seeded(tmp_path: Path) -> dict[str, str]:
    conn = wiki_db(tmp_path)
    e1 = add_exchange(conn, 1, "2026-02-01")
    for i in range(3):
        add_claim(conn, e1, "user-aaaa", f"The O2 fact number {i}.")
    add_claim(conn, e1, "user-aaaa", "Octane fact.")
    add_claim(conn, e1, "user-aaaa", "Nothing topical.")
    add_claim(conn, e1, "user-aaaa", "Excluded O2 claim.", review="made_up")
    add_claim(conn, e1, "user-aaaa", "Unverified O2 claim.", check="uncheckable")
    conn.commit()
    conn.close()
    return environment(tmp_path)


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str]:
    out = io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=io.StringIO())
    return code, out.getvalue()


def test_build_writes_pages_for_topics_over_the_minimum(tmp_path: Path) -> None:
    env = seeded(tmp_path)
    out_dir = tmp_path / "site"
    code, out = run(["wiki", "build", "--out", str(out_dir), "--min-claims", "3"], env)
    assert code == ExitCode.OK
    assert sorted(p.name for p in out_dir.iterdir()) == ["index.md", "o2.md"]
    assert "pages: 1" in out


def test_build_default_minimum_is_three(tmp_path: Path) -> None:
    env = seeded(tmp_path)
    out_dir = tmp_path / "site"
    run(["wiki", "build", "--out", str(out_dir)], env)
    assert (out_dir / "o2.md").exists()
    assert not (out_dir / "octane.md").exists()


def test_stats_reports_topics_pages_distribution_and_unassigned(tmp_path: Path) -> None:
    env = seeded(tmp_path)
    code, out = run(["wiki", "stats", "--min-claims", "3"], env)
    assert code == ExitCode.OK
    assert "topics: 2" in out
    assert "pages: 1" in out
    assert "unassigned: 1" in out
    assert "excluded: 1 (review 1)" in out
    assert "3-4: 1" in out


def test_verdicts_supported_only_excludes_uncheckable_and_bad_names_are_refused(
    tmp_path: Path,
) -> None:
    env = seeded(tmp_path)
    code, out = run(["wiki", "stats", "--min-claims", "3", "--verdicts", "supported"], env)
    assert code == ExitCode.OK
    assert "excluded: 2 (check:uncheckable 1, review 1)" in out
    code, out = run(["wiki", "stats", "--min-claims", "3", "--verdicts", "any"], env)
    assert code == ExitCode.OK and "excluded: 1 (review 1)" in out
    code, _ = run(["wiki", "stats", "--verdicts", "bogus"], env)
    assert code == ExitCode.CONFIG


def test_claim_gate_drops_techless_claims_from_stats_and_build(tmp_path: Path) -> None:
    env = seeded(tmp_path)
    code, out = run(["wiki", "stats", "--min-claims", "3", "--claim-gate"], env)
    assert code == ExitCode.OK
    assert "unassigned: 0" in out
    assert "excluded: 2 (no_tech 1, review 1)" in out
    out_dir = tmp_path / "site"
    run(["wiki", "build", "--out", str(out_dir), "--claim-gate"], env)
    assert (out_dir / "o2.md").exists()


def test_runs_limits_stats_and_rejects_bad_ids(tmp_path: Path) -> None:
    env = seeded(tmp_path)
    code, out = run(["wiki", "stats", "--runs", "1"], env)
    assert code == ExitCode.OK and "claims: 6" in out
    code, out = run(["wiki", "stats", "--runs", "1,9"], env)
    assert code != ExitCode.OK
    code, out = run(["wiki", "stats", "--runs", "x"], env)
    assert code != ExitCode.OK
