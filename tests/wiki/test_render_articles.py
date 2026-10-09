import io
from pathlib import Path

from infovore.cli import ExitCode, main
from infovore.db.wiki_articles import create_article_run, record_section
from infovore.db.wiki_tags import create_tag_run, record_tags
from infovore.wiki.build import WikiClaim, load_claims, render_page
from tests.claims.seed import NOW, db, environment
from tests.wiki.seed import add_claim, add_exchange, wiki_db


def claim(
    claim_id: int, statement: str, topics: tuple[str, ...], check: str | None = "supported"
) -> WikiClaim:
    return WikiClaim(
        claim_id,
        10 + claim_id,
        "2026-02-01",
        f"user-{claim_id}",
        statement,
        frozenset(topics),
        check,
    )


A = claim(1, "Indigo2 has an R4400 cpu", ("Indigo2",))
B = claim(2, "Indigo2 ships with IRIX", ("Indigo2",), None)
C = claim(3, "Indigo2 has an R4400 processor", ("Indigo2",))


def test_article_page_has_prose_with_citations_and_a_source_list() -> None:
    article = {"General": [("It has an R4400 cpu.", [1]), ("It ships with IRIX.", [2])]}
    text = render_page("Indigo2", [A, B], ["Indigo2"], article)
    assert text == (
        "# Indigo2\n\nClaims: 2\n\n"
        "It has an R4400 cpu. [1] It ships with IRIX. [2]\n\n"
        "## Sources\n\n"
        "1. Indigo2 has an R4400 cpu (user-1, 2026-02-01, exchange 11, check: supported)\n"
        "2. Indigo2 ships with IRIX (user-2, 2026-02-01, exchange 12, check: unchecked)\n"
    )


def test_numbers_follow_first_citation_and_are_reused() -> None:
    article = {"General": [("One.", [2, 1]), ("Two.", [1, 2, 2]), ("Three.", [1])]}
    text = render_page("Indigo2", [A, B], [], article)
    assert "One. [1][2] Two. [2][1] Three. [2]" in text
    assert text.index("1. Indigo2 ships") < text.index("2. Indigo2 has")


def test_unknown_citations_are_skipped_and_an_empty_section_falls_back() -> None:
    article = {"General": [("Ghost.", [99])]}
    text = render_page("Indigo2", [A], [], article)
    assert "Ghost" not in text and "## Sources" not in text
    assert "- Indigo2 has an R4400 cpu (user-1, exchange 11, 2026-02-01)" in text


def test_fallback_groups_similar_claims() -> None:
    text = render_page("Indigo2", [A, C, B], ["Indigo2"])
    assert "- Indigo2 has an R4400 cpu (+1 similar) (user-1, exchange 11, 2026-02-01)\n" in text
    assert "- Indigo2 ships with IRIX (user-2, exchange 12, 2026-02-01)\n" in text
    assert "user-3" not in text


def test_only_sections_without_text_fall_back_and_headings_follow_the_rule() -> None:
    a = claim(1, "Indigo2 runs IRIX", ("Indigo2", "IRIX"))
    b = claim(2, "Indigo2 has an R4400 cpu", ("Indigo2",))
    article = {"With IRIX": [("It runs IRIX.", [1])]}
    text = render_page("Indigo2", [a, b], [], article)
    assert "## With IRIX\n\nIt runs IRIX. [1]\n" in text
    assert "## General\n\n- Indigo2 has an R4400 cpu (user-2" in text
    single = render_page("Indigo2", [b], [], {"General": [("It has a cpu.", [2])]})
    assert "## General" not in single and "It has a cpu. [1]" in single


def seed(tmp_path: Path) -> dict[str, str]:
    conn = wiki_db(tmp_path)
    exchange = add_exchange(conn, 1, "2026-02-01")
    ids = [
        add_claim(conn, exchange, "user-aaaa", "Indigo2 has an R4400 cpu", check="low_overlap"),
        add_claim(conn, exchange, "user-bbbb", "Indigo2 ships with IRIX", check=None),
    ]
    tag_run = create_tag_run(
        conn, endpoint="e", model_alias="m", model_id="m", prompt_hash="p", now=NOW
    )
    record_tags(conn, tag_run, {ids[0]: ["Indigo2"], ids[1]: ["Indigo2"]})
    article_run = create_article_run(
        conn,
        endpoint="e",
        model_alias="m",
        model_id="m",
        prompt_hash="p",
        tag_run_id=tag_run,
        now=NOW,
    )
    record_section(conn, article_run, "Indigo2", "General", [("It has an R4400.", [ids[0]])], 0)
    conn.commit()
    conn.close()
    return environment(tmp_path)


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str]:
    err = io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=io.StringIO(), stderr=err)
    return code, err.getvalue()


def test_load_claims_carries_the_check_verdict(tmp_path: Path) -> None:
    seed(tmp_path)
    claims, _ = load_claims(db(tmp_path))
    assert [c.check for c in claims] == ["low_overlap", None]


def test_cli_builds_article_pages(tmp_path: Path) -> None:
    env = seed(tmp_path)
    site = tmp_path / "site"
    base = ["wiki", "build", "--out", str(site), "--min-claims", "1", "--tag-run", "1"]
    code, _ = run([*base, "--article-run", "1"], env)
    assert code == ExitCode.OK
    page = (site / "indigo2.md").read_text()
    assert "It has an R4400. [1]" in page and "check: low_overlap)" in page


def test_cli_rejects_a_bad_article_run(tmp_path: Path) -> None:
    env = seed(tmp_path)
    base = ["wiki", "build", "--out", str(tmp_path / "s"), "--min-claims", "1"]
    code, err = run([*base, "--article-run", "1"], env)
    assert code == ExitCode.CONFIG and "--article-run needs --tag-run" in err
    code, err = run([*base, "--tag-run", "1", "--article-run", "9"], env)
    assert code == ExitCode.CONFIG and "unknown article run 9" in err
