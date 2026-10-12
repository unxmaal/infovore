import json
from pathlib import Path

from infovore.wiki.build import Navigation, build_site, page_files
from infovore.wiki.html import SEARCH, body, html_name, inline, page, search_entries, write_html
from tests.wiki.test_build import CLAIMS, claim


def test_inline_escapes_links_and_citations() -> None:
    text = "A & B <c> [O2](o2.md) and [SGI](https://en.wikipedia.org/wiki/SGI) fact. [1][12]"
    assert inline(text) == (
        'A &amp; B &lt;c&gt; <a href="o2.html">O2</a> and'
        ' <a href="https://en.wikipedia.org/wiki/SGI">SGI</a> fact.'
        ' <sup><a href="#s1">1</a></sup><sup><a href="#s12">12</a></sup>'
    )
    assert html_name("irix-6-5.md") == "irix-6-5.html" and html_name("x.txt") == "x.txt"


def test_body_renders_headings_paragraphs_nested_bullets_and_numbered_sources() -> None:
    markdown = (
        "# O2\n\nClaims: 3\nPart of [SGI](sgi.md)\n\n## With IRIX\n\n"
        "First line\nsecond line. [1]\n\n"
        "- [O2](o2.md) (3)\n  - [O2 PSU](o2-psu.md) (1)\n    - deep\n  - [O2 fan](o2-fan.md) (1)\n"
        "- [IRIX](irix.md) (2)\n\n## Sources\n\n1. The O2 runs IRIX. (user-aaaa)\n2. Second.\n"
        "## After\n\n1. Again.\n"
    )
    assert body(markdown) == (
        "<h1>O2</h1>\n"
        '<p>Claims: 3 Part of <a href="sgi.html">SGI</a></p>\n'
        '<h2 id="with-irix">With IRIX</h2>\n'
        '<p>First line second line. <sup><a href="#s1">1</a></sup></p>\n'
        "<ul>\n"
        '<li><a href="o2.html">O2</a> (3)\n'
        "<ul>\n"
        '<li><a href="o2-psu.html">O2 PSU</a> (1)\n'
        "<ul>\n"
        "<li>deep\n"
        "</li></ul>\n"
        "</li>\n"
        '<li><a href="o2-fan.html">O2 fan</a> (1)\n'
        "</li></ul>\n"
        "</li>\n"
        '<li><a href="irix.html">IRIX</a> (2)\n'
        "</li></ul>\n"
        '<h2 id="sources">Sources</h2>\n'
        "<ol>\n"
        '<li id="s1">The O2 runs IRIX. (user-aaaa)</li>\n'
        '<li id="s2">Second.</li>\n'
        "</ol>\n"
        '<h2 id="after">After</h2>\n'
        "<ol>\n"
        '<li id="s1">Again.</li>\n'
        "</ol>\n"
    )
    assert body("") == "\n" and body("plain") == "<p>plain</p>\n"


def test_page_wraps_body_with_title_nav_and_optional_search() -> None:
    text = page("# O2 & co\n\nx\n")
    assert text.startswith("<!doctype html>") and "<title>O2 &amp; co</title>" in text
    assert '<a href="index.html">Index</a>' in text and "<script>" not in text
    index = page("# Index\n", [{"t": "A</script>", "f": "a.html", "n": 1}])
    assert '<input id="q"' in index and '<script>var ITEMS = [{"t": "A<\\/script>"' in index
    assert "})(ITEMS);" in index
    assert "<title>Wiki</title>" in page("no heading\n")


def test_build_site_writes_html_and_search_index(tmp_path: Path) -> None:
    claims = [*CLAIMS, claim(8, "The O2 PSU hums.", {"O2", "O2 PSU"})]
    nav = Navigation(kinds={"O2": "machine"}, redirects={"O2s": "O2"}, index_min=1)
    written = build_site(claims, 1, tmp_path, nav=nav, html=True)
    assert "index.html" in written and "o2s.html" in written and SEARCH in written
    assert len(written) == 2 * 11 + 1
    assert "kind-machine.html" in written
    assert (tmp_path / "o2s.html").read_text().count('<a href="o2.html">O2</a>') == 1
    index = (tmp_path / "index.html").read_text()
    assert (
        '<li><a href="o2.html">O2</a> (5)\n<ul>\n<li><a href="o2-psu.html">O2 PSU</a> (1)' in index
    )
    assert "<script>" in index and "<script>" not in (tmp_path / "o2.html").read_text()
    search = json.loads((tmp_path / SEARCH).read_text())
    assert search[0] == {"t": "O2", "f": "o2.html", "n": 5}
    assert {"t": "O2s", "f": "o2.html", "n": 5, "a": "O2"} in search
    assert [r["t"] for r in search] == [
        "O2",
        "R12000",
        "IRIX",
        "IRIX 6.5",
        "O2 PSU",
        "Octane",
        "O2s",
    ]
    assert build_site(claims, 1, tmp_path / "md") == sorted(
        n
        for n in written
        if n.endswith(".md") and n not in ("o2s.md", "all.md") and "kind-" not in n
    )


def test_write_html_and_search_entries_directly(tmp_path: Path) -> None:
    (tmp_path / "index.md").write_text("# Index\n")
    (tmp_path / "a.md").write_text("# A\n")
    files = page_files({"A"})
    written = write_html(tmp_path, ["index.md", "a.md"], {"A": 2}, {"Alpha": "A"}, files)
    assert written == ["index.html", "a.html", SEARCH]
    assert search_entries({}, {}, {}) == []
