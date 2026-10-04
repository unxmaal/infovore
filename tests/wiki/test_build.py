from pathlib import Path

from infovore.wiki.build import (
    WikiClaim,
    build_site,
    compute_stats,
    load_claims,
    render_index,
    render_page,
)
from tests.wiki.seed import add_claim, add_exchange, wiki_db

GOLDEN = Path(__file__).parent / "golden"


def claim(
    claim_id: int, statement: str, topics: set[str], *, exchange: int = 1, day: str = "2026-01-02"
) -> WikiClaim:
    return WikiClaim(claim_id, exchange, day, "user-aaaa", statement, frozenset(topics))


CLAIMS = [
    claim(3, "The O2 uses an R12000.", {"O2", "R12000"}, exchange=7, day="2026-01-03"),
    claim(1, "The O2 has a unified memory architecture.", {"O2"}, exchange=5),
    claim(2, "An O2 can run IRIX 6.5.", {"O2", "IRIX 6.5", "IRIX"}, exchange=6),
    claim(4, "The O2 maxes out at 1 GB   of\nRAM.", {"O2", "R12000"}, exchange=8),
    claim(5, "Octane is faster.", {"Octane"}, exchange=9),
]


def test_golden_page() -> None:
    page = render_page("O2", CLAIMS[:4], {"O2", "R12000", "IRIX 6.5"})
    assert page == (GOLDEN / "o2.md").read_text()


def test_page_without_cooccurrence_has_no_groups_or_see_also() -> None:
    page = render_page("Octane", [CLAIMS[4]], {"Octane"})
    assert "## " not in page
    assert "See also" not in page
    assert page.startswith("# Octane\n")


def test_see_also_only_links_topics_with_pages() -> None:
    page = render_page("O2", CLAIMS[:4], {"O2"})
    assert "See also" not in page
    assert "## With R12000" in page


def test_index_lists_pages_by_name_with_counts() -> None:
    index = render_index({"O2": 4, "IRIX 6.5": 1})
    assert index == "# Index\n\n- [IRIX 6.5](irix-6-5.md) (1)\n- [O2](o2.md) (4)\n"


def test_load_claims_applies_eligibility_and_topics(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    e1 = add_exchange(conn, 1, "2026-02-01")
    e2 = add_exchange(conn, 2, "2026-02-02")
    add_claim(conn, e1, "user-aaaa", "The O2 is quiet.")
    add_claim(conn, e1, "user-bbbb", "The O2 is loud.", review="wrong")
    add_claim(conn, e2, "user-cccc", "The O2 has 1GB.", check="unsupported_fact")
    add_claim(conn, e2, "user-dddd", "Unchecked O2 claim.", check=None)
    add_claim(conn, e2, "user-eeee", "Good but untopical.", review="good", check="uncheckable")
    claims, excluded = load_claims(conn)
    assert [(c.speaker, c.date, c.exchange_id, c.topics) for c in claims] == [
        ("user-aaaa", "2026-02-01", e1, frozenset({"O2"})),
        ("user-eeee", "2026-02-02", e2, frozenset()),
    ]
    assert excluded == 3


def test_load_claims_collapses_duplicates_across_runs(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    e1 = add_exchange(conn, 1, "2026-02-01")
    add_claim(conn, e1, "user-aaaa", "The O2 is quiet.")
    add_claim(conn, e1, "user-aaaa", "The O2 is quiet.")
    claims, _ = load_claims(conn)
    assert len(claims) == 1


def test_stats_counts_pages_distribution_and_unassigned() -> None:
    claims = [*CLAIMS, claim(6, "No topic here.", set())]
    stats = compute_stats(claims, 2)
    assert stats.topics == 5
    assert stats.pages == 2
    assert stats.unassigned == 1
    assert stats.claims_per_page == {"O2": 4, "R12000": 2}
    assert stats.distribution() == {"2": 1, "3-4": 1}


def test_stats_with_no_pages() -> None:
    stats = compute_stats([], 3)
    assert (stats.topics, stats.pages, stats.unassigned) == (0, 0, 0)
    assert stats.distribution() == {}


def test_stats_distribution_covers_every_bucket() -> None:
    claims = [
        claim(i, f"s{i}", {f"T{n}"})
        for n, size in enumerate([1, 3, 5, 10, 25, 100])
        for i in range(n * 1000, n * 1000 + size)
    ]
    assert compute_stats(claims, 1).distribution() == {
        "1": 1,
        "3-4": 1,
        "5-9": 1,
        "10-24": 1,
        "25-99": 1,
        "100+": 1,
    }


def test_build_site_writes_pages_and_index_deterministically(tmp_path: Path) -> None:
    out = tmp_path / "wiki"
    written = build_site(CLAIMS, 2, out)
    assert written == ["index.md", "o2.md", "r12000.md"]
    assert sorted(p.name for p in out.iterdir()) == written
    first = {p.name: p.read_text() for p in out.iterdir()}
    build_site(list(reversed(CLAIMS)), 2, out)
    assert {p.name: p.read_text() for p in out.iterdir()} == first
    assert "[R12000](r12000.md)" in first["o2.md"]


def test_pages_carry_no_discord_links(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    e1 = add_exchange(conn, 1, "2026-02-01")
    add_claim(conn, e1, "user-aaaa", "The O2 is quiet.")
    claims, _ = load_claims(conn)
    page = render_page("O2", claims, {"O2"})
    assert "discord" not in page.lower()
    assert "(user-aaaa, exchange" in page
