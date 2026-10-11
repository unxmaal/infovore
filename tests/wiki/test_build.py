import re
from dataclasses import replace
from pathlib import Path

from infovore.db.novelty import Novelty, store_novelty
from infovore.wiki.build import (
    UNSORTED,
    Article,
    Navigation,
    WikiClaim,
    _by_topic,
    article_parts,
    build_site,
    compute_stats,
    established,
    load_claims,
    page_files,
    part_key,
    render_all,
    render_disambiguation,
    render_index,
    render_page,
    render_redirect,
    sections_of,
)
from infovore.wiki.eligibility import SUPPORTED
from tests.claims.seed import NOW
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
    page = render_page("O2", CLAIMS[:4], page_files({"O2", "R12000", "IRIX 6.5"}))
    assert page == (GOLDEN / "o2.md").read_text()


def test_page_without_cooccurrence_has_no_groups_or_see_also() -> None:
    page = render_page("Octane", [CLAIMS[4]], page_files({"Octane"}))
    assert "## " not in page
    assert "See also" not in page
    assert page.startswith("# Octane\n")


def test_see_also_only_links_topics_with_pages() -> None:
    page = render_page("O2", CLAIMS[:4], page_files({"O2"}))
    assert "See also" not in page
    assert "## With R12000" in page


def test_index_lists_pages_by_name_with_counts() -> None:
    index = render_index({"O2": 4, "IRIX 6.5": 1}, page_files({"O2", "IRIX 6.5"}))
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
        ("user-cccc", "2026-02-02", e2, frozenset({"O2"})),
        ("user-dddd", "2026-02-02", e2, frozenset({"O2"})),
        ("user-eeee", "2026-02-02", e2, frozenset()),
    ]
    assert excluded.total == 1 and excluded.by_reason == {"review": 1}


def test_load_claims_verdict_policy_excludes_unsupported_and_unchecked(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    e1 = add_exchange(conn, 1, "2026-02-01")
    add_claim(conn, e1, "user-aaaa", "The O2 is quiet.")
    add_claim(conn, e1, "user-bbbb", "The O2 is loud.", review="wrong")
    add_claim(conn, e1, "user-cccc", "The O2 has 1GB.", check="unsupported_fact")
    add_claim(conn, e1, "user-dddd", "Unchecked O2 claim.", check=None)
    add_claim(conn, e1, "user-eeee", "Good but unverified.", review="good", check="uncheckable")
    add_claim(conn, e1, "user-ffff", "Maybe an O2 thing.", check="uncheckable")

    claims, excluded = load_claims(conn, verdicts=SUPPORTED)

    assert [c.speaker for c in claims] == ["user-aaaa", "user-eeee"]
    assert excluded.by_reason == {
        "review": 1,
        "check:unsupported_fact": 1,
        "check:unchecked": 1,
        "check:uncheckable": 1,
    }
    assert str(excluded) == (
        "4 (check:uncheckable 1, check:unchecked 1, check:unsupported_fact 1, review 1)"
    )
    assert str(load_claims(conn, verdicts=frozenset({"supported", "uncheckable"}))[1]) == (
        "3 (check:unchecked 1, check:unsupported_fact 1, review 1)"
    )


def test_load_claims_collapses_duplicates_across_runs(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    e1 = add_exchange(conn, 1, "2026-02-01")
    add_claim(conn, e1, "user-aaaa", "The O2 is quiet.")
    add_claim(conn, e1, "user-aaaa", "The O2 is quiet.")
    claims, _ = load_claims(conn)
    assert len(claims) == 1


def test_load_claims_keeps_only_the_chosen_runs(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path, runs=3)
    e1 = add_exchange(conn, 1, "2026-02-01")
    add_claim(conn, e1, "user-aaaa", "The O2 is quiet.", run_id=1)
    add_claim(conn, e1, "user-aaaa", "The Octane is loud.", run_id=2)
    add_claim(conn, e1, "user-aaaa", "The Indy is small.", run_id=3)
    claims, _ = load_claims(conn, runs=[2, 3])
    assert sorted(c.statement for c in claims) == ["The Indy is small.", "The Octane is loud."]
    assert len(load_claims(conn)[0]) == 3


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
    page = render_page("O2", claims, page_files({"O2"}))
    assert "discord" not in page.lower()
    assert "(user-aaaa, exchange" in page


def test_load_claims_claim_gate_drops_techless_claims(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    e1 = add_exchange(conn, 1, "2026-02-01")
    add_claim(conn, e1, "user-aaaa", "The O2 is quiet.")
    add_claim(conn, e1, "user-bbbb", "People are friendly.")
    claims, excluded = load_claims(conn, tech_only=True)
    assert [c.speaker for c in claims] == ["user-aaaa"] and excluded.by_reason == {"no_tech": 1}
    assert len(load_claims(conn)[0]) == 2


def test_small_co_topic_sections_fold_into_general() -> None:
    sections = sections_of("O2", CLAIMS[:4], min_section=2)
    assert list(sections) == ["With R12000", "General"]
    assert [c.claim_id for c in sections["With R12000"]] == [4, 3]
    assert [c.claim_id for c in sections["General"]] == [1, 2]


def test_build_site_folds_small_sections(tmp_path: Path) -> None:
    build_site(CLAIMS, 2, tmp_path, min_section=2)
    page = (tmp_path / "o2.md").read_text()
    assert "## With R12000" in page and "## With IRIX" not in page


def test_untagged_claims_land_on_the_unsorted_page_and_count_as_on_pages() -> None:
    orphan = WikiClaim(9, 1, "2026-01-01", "user-aaaa", "No topic at all.", frozenset())
    claims = [*CLAIMS[:4], orphan]

    assert _by_topic([orphan]) == {}
    assert [c.claim_id for c in _by_topic([orphan], UNSORTED)[UNSORTED]] == [9]
    stats = compute_stats(claims, 1)
    assert stats.unassigned == 1 and UNSORTED in stats.claims_per_page
    assert stats.on_pages == 5
    assert compute_stats(claims, 2).on_pages == 4


def test_article_parts_follow_numbered_continuations_without_gaps() -> None:
    article: Article = {
        "General": [("one", [1])],
        "General#2": [("two", [2])],
        "General#4": [("four", [4])],
    }

    assert article_parts(article, "General") == [("one", [1]), ("two", [2])]
    assert article_parts(article, "Missing") == []
    assert (part_key("General", 1), part_key("General", 3)) == ("General", "General#3")


def test_render_page_concatenates_section_parts() -> None:
    key = next(iter(sections_of("O2", CLAIMS[:2])))
    article: Article = {
        key: [("The O2 is compact.", [1])],
        part_key(key, 2): [("It uses an R12000.", [3])],
    }

    page = render_page("O2", CLAIMS[:2], page_files({"O2"}), article)

    assert "The O2 is compact. [1] It uses an R12000. [2]" in page


def test_page_files_are_unique_and_never_empty_or_index() -> None:
    files = page_files(
        ["O2", "O2机器", "C", "C++", "电源供应", "Index", "o2-2"],
        {"O2": 100, "O2机器": 4, "C": 5, "C++": 9, "电源供应": 3, "Index": 1, "o2-2": 50},
    )
    assert files["O2"] == "o2.md" and files["o2-2"] == "o2-2.md" and files["O2机器"] == "o2-3.md"
    assert files["C++"] == "c.md" and files["C"] == "c-2.md"
    assert files["Index"] == "index-2.md"
    assert re.fullmatch(r"topic-[0-9a-f]{8}\.md", files["电源供应"])
    assert len(set(files.values())) == 7
    assert page_files(["b", "a"]) == {"a": "a.md", "b": "b.md"}


def test_build_site_keeps_colliding_pages_and_links_to_them(tmp_path: Path) -> None:
    claims = [
        claim(1, "The O2 uses an R12000.", {"O2", "R12000"}),
        claim(2, "The O2 is quiet.", {"O2"}),
        claim(3, "O2机器 boots.", {"O2机器", "R12000"}),
        claim(4, "电源供应 fails.", {"电源供应", "R12000"}),
    ]
    written = build_site(claims, 1, tmp_path)
    assert "o2.md" in written and "o2-2.md" in written and ".md" not in written
    assert (tmp_path / "o2.md").read_text().startswith("# O2\n")
    assert (tmp_path / "o2-2.md").read_text().startswith("# O2机器\n")
    index = (tmp_path / "index.md").read_text()
    see_also = (tmp_path / "r12000.md").read_text()
    for text in (index, see_also):
        for target in re.findall(r"\]\(([^)]+)\)", text):
            assert (tmp_path / target).is_file(), target
    assert "[O2机器](o2-2.md)" in see_also


def test_background_claims_leave_the_page_with_one_line_naming_their_reference(
    tmp_path: Path,
) -> None:
    conn = wiki_db(tmp_path)
    e1 = add_exchange(conn, 1, "2026-02-01")
    ids = [
        add_claim(conn, e1, "user-aaaa", "The O2 runs IRIX."),
        add_claim(conn, e1, "user-bbbb", "The O2 PSU fan whines before it fails."),
        add_claim(conn, e1, "user-cccc", "The O2 has an R5000."),
    ]
    store_novelty(
        conn,
        [
            Novelty(ids[0], "O2", "SGI O2", 0.93, "The O2 runs IRIX 6.3.", 1.0),
            Novelty(ids[1], "O2", "SGI O2", 0.41, "far", 0.2),
        ],
        NOW,
    )
    conn.commit()
    plain, _ = load_claims(conn)
    assert all(c.novelty is None for c in plain)
    claims, _ = load_claims(conn, background=0.8)
    by_id = {c.claim_id: c for c in claims}
    assert by_id[ids[0]].novelty == 0.93 and by_id[ids[0]].reference == "SGI O2"
    assert by_id[ids[1]].novelty == 0.41 and by_id[ids[2]].novelty is None
    assert established(by_id[ids[0]], 0.8) and not established(by_id[ids[0]], None)
    assert not established(by_id[ids[1]], 0.8) and not established(by_id[ids[2]], 0.8)
    assert (
        compute_stats(claims, 1, 0.8).background == 1 and compute_stats(claims, 1).background == 0
    )

    page = render_page("O2", claims, page_files({"O2"}), background=0.8)
    link = "[SGI O2](https://en.wikipedia.org/wiki/SGI_O2)"
    assert f"\nBackground: 1 claim already covered by Wikipedia, see {link}.\n" in page
    assert "runs IRIX." not in page and "## Background" not in page
    assert page.index("Background:") < page.index("fan whines")
    assert "Background:" not in render_page("O2", claims, page_files({"O2"}))
    site = tmp_path / "site"
    build_site(claims, 1, site, background=0.8)
    assert "Background: 1 claim already" in (site / "o2.md").read_text()
    bare = replace(by_id[ids[0]], reference=None)
    other = replace(by_id[ids[1]], novelty=0.9, reference="IRIX")
    third = replace(by_id[ids[2]], novelty=0.9, reference="IRIX")
    page = render_page("O2", [bare, other, third], page_files({"O2"}), background=0.8)
    assert "Background: 3 claims already covered by Wikipedia, see [IRIX](" in page
    assert "Background: 1 claim already covered by Wikipedia.\n" in render_page(
        "O2", [bare], page_files({"O2"}), background=0.8
    )
    both = [by_id[ids[0]], other, third]
    page = render_page("O2", both, page_files({"O2"}), background=0.8)
    assert "see [IRIX](https://en.wikipedia.org/wiki/IRIX), [SGI O2](https" in page


def test_index_groups_by_kind_nests_children_and_links_the_all_page() -> None:
    pages = {"O2": 10, "O2 PSU": 3, "O2 fan": 1, "IRIX": 8, "IRIX 6.5": 4, "Foo": 1, "Unsorted": 5}
    files = page_files(pages, pages)
    nav = Navigation(kinds={"O2": "machine", "IRIX": "os", "Foo": "cpu"}, index_min=2)
    index = render_index(pages, files, nav)
    assert index == (
        "# Index\n\n6 topics, 27 claims. [All topics A-Z](all.md)\n\n"
        "## Machines\n\n- [O2](o2.md) (10)\n  - [O2 PSU](o2-psu.md) (3)\n\n"
        "## Operating systems\n\n- [IRIX](irix.md) (8)\n  - [IRIX 6.5](irix-6-5.md) (4)\n\n"
        "## Unsorted\n\n- [Unsorted](unsorted.md) (5)\n"
    )
    assert "Unsorted" not in render_index({"O2": 1}, files, Navigation(index_min=1))
    assert "## Other\n\n- [O2](o2.md) (1)" in render_index(
        {"O2": 1}, files, Navigation(index_min=1)
    )
    assert render_all({"O2": 2, "o2 fan": 1, "6.5": 1}, page_files({"O2", "o2 fan", "6.5"})) == (
        "# All topics\n\n3 pages. [Index](index.md)\n\n## #\n\n- [6.5](6-5.md) (1)\n\n"
        "## O\n\n- [O2](o2.md) (2)\n- [o2 fan](o2-fan.md) (1)\n"
    )


def test_page_links_its_parent_subtopics_and_groups_see_also_by_kind() -> None:
    files = page_files({"O2", "O2 PSU", "R12000", "IRIX 6.5", "IRIX", "SGI"})
    nav = Navigation(kinds={"R12000": "cpu", "IRIX": "os", "IRIX 6.5": "os"})
    claims = [*CLAIMS[:4], claim(9, "The O2 PSU hums.", {"O2", "O2 PSU"})]
    page = render_page(
        "O2", claims, files, nav=nav, parent="SGI", children={"O2 PSU": 1}, min_section=10
    )
    assert "Claims: 5\nPart of [SGI](sgi.md)\n" in page
    assert "## Subtopics\n\n- [O2 PSU](o2-psu.md) (1)\n" in page
    assert page.endswith(
        "## See also\n\n- Processors: [R12000](r12000.md)\n"
        "- Operating systems: [IRIX](irix.md), [IRIX 6.5](irix-6-5.md)\n"
    )
    assert "[SGI](sgi.md)" not in page.split("## See also")[1]
    flat = render_page("O2", claims, files, min_section=10)
    assert "Part of" not in flat and "Subtopics" not in flat
    assert flat.endswith("- [IRIX](irix.md)\n- [IRIX 6.5](irix-6-5.md)\n- [O2 PSU](o2-psu.md)\n")


def test_build_site_writes_redirects_disambiguation_and_all_pages(tmp_path: Path) -> None:
    claims = [
        *CLAIMS,
        claim(6, "FC is fast.", {"Fibre Channel"}),
        claim(7, "FC.", {"Fibre Channel"}),
        claim(8, "The O2 PSU hums.", {"O2", "O2 PSU"}),
    ]
    nav = Navigation(
        kinds={"O2": "machine"},
        redirects={"Fiber Channel": "Fibre Channel", "Ghost": "Nowhere", "O2": "Octane"},
        disambiguations={"Ultra": ["O2", "Nothing"], "Octane": ["O2"], "Gone": ["Nothing"]},
        index_min=1,
    )
    written = build_site(claims, 1, tmp_path, nav=nav)
    assert written == [
        "all.md",
        "fiber-channel.md",
        "fibre-channel.md",
        "index.md",
        "irix-6-5.md",
        "irix.md",
        "o2-psu.md",
        "o2.md",
        "octane.md",
        "r12000.md",
        "ultra.md",
    ]
    assert "## Subtopics\n\n- [O2 PSU](o2-psu.md) (1)\n" in (tmp_path / "o2.md").read_text()
    assert "Claims: 1\nPart of [O2](o2.md)\n" in (tmp_path / "o2-psu.md").read_text()
    assert (tmp_path / "fiber-channel.md").read_text() == (
        "# Fiber Channel\n\nSee [Fibre Channel](fibre-channel.md).\n"
    )
    assert (tmp_path / "ultra.md").read_text() == (
        "# Ultra\n\nUltra may refer to:\n\n- [O2](o2.md) (5)\n- Nothing\n"
    )
    assert (tmp_path / "octane.md").read_text().startswith("# Octane\n\nClaims: 1\n")
    for page in tmp_path.iterdir():
        for target in re.findall(r"\]\(([^)]+\.md)\)", page.read_text()):
            assert (tmp_path / target).is_file(), (page.name, target)
    index = (tmp_path / "index.md").read_text()
    assert "## Machines\n\n- [O2](o2.md) (5)\n  - [O2 PSU](o2-psu.md) (1)" in index
    assert "Fiber Channel" not in index
    assert render_redirect("A", "B", {"B": "b.md"}) == "# A\n\nSee [B](b.md).\n"
    assert render_disambiguation("A", [], {}, {}) == "# A\n\nA may refer to:\n\n"
