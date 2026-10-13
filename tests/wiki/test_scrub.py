from pathlib import Path

from infovore.wiki.build import load_claims, page_files, render_page
from infovore.wiki.scrub import REMOVED, Hit, scan, scan_text, scrub
from tests.wiki.seed import add_claim, add_exchange, wiki_db


def test_scrub_removes_discord_links_and_nothing_else() -> None:
    text = (
        "See https://discord.com/channels/524976640468713497/938073247935783004/ and"
        " https://discord.gg/abc12 and https://cdn.discordapp.com/attachments/1/2/x.png;"
        " keep https://en.wikipedia.org/wiki/IRIX and discord the word."
    )
    assert scrub(text) == (
        f"See {REMOVED} and {REMOVED} and {REMOVED} keep https://en.wikipedia.org/wiki/IRIX"
        " and discord the word."
    )
    assert scrub("plain") == "plain"


def test_scan_text_reports_links_snowflakes_and_emails_with_line_numbers() -> None:
    text = (
        "An O2 with a R10k @250 MHz, curl -F'file=@image.png' https://0x0.st\n"
        "id 524976640468713497 and version 6.5.22 and 1234567890123456 short\n"
        "mail someone@example.com, link https://discord.com/channels/1/2\n"
    )
    assert list(scan_text("a.html", text)) == [
        Hit("a.html", 2, "snowflake", "524976640468713497"),
        Hit("a.html", 3, "discord-link", "https://discord.com/channels/1/2"),
        Hit("a.html", 3, "email", "someone@example.com"),
    ]
    assert list(scan_text("b.html", "0x0123456789012345678 2026-02-01")) == []


def test_scan_walks_html_md_and_json_only(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.html").write_text("x https://discord.gg/q\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("fine\n", encoding="utf-8")
    (tmp_path / "search.json").write_text('[{"t": "one@two.org"}]', encoding="utf-8")
    (tmp_path / "notes.txt").write_text("https://discord.gg/ignored\n", encoding="utf-8")
    assert scan(tmp_path) == [
        Hit("search.json", 1, "email", "one@two.org"),
        Hit("sub/a.html", 1, "discord-link", "https://discord.gg/q"),
    ]


def test_build_scrubs_discord_links_from_claim_statements(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    e1 = add_exchange(conn, 1, "2026-02-01")
    add_claim(conn, e1, "user-aaaa", "The O2 is quiet.")
    add_claim(conn, e1, "user-bbbb", "The O2 manual is at https://discord.com/channels/1/2/3 now.")
    add_claim(conn, e1, "user-cccc", "The O2 manual is at https://discord.com/channels/9/8/7 too.")
    claims, _ = load_claims(conn)
    assert {c.statement for c in claims} >= {f"The O2 manual is at {REMOVED} now."}
    page = render_page("O2", claims, page_files({"O2"}))
    assert "discord.com" not in page and REMOVED in page
