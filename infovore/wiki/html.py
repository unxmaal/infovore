import html
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from infovore.wiki.gifs import BADGES, write_gifs
from infovore.wiki.topics import slug

SEARCH: Final = "search.json"
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_CITE = re.compile(r"\[(\d+)\]")
_BULLET = re.compile(r"^( *)- (.*)$")
_NUMBERED = re.compile(r"^(\d+)\. (.*)$")
GUESTBOOK: Final = "https://github.com/unxmaal/infovore/issues"
STYLE: Final = (
    'body{background:#000 url(bg.gif);color:#0f0;font:16px/1.4 "Times New Roman",Times,serif;'
    "margin:0;padding:8px}"
    "a:link{color:#ff0}a:visited{color:#f0f}a:hover{color:#0ff;background:#008}"
    'h1{font:bold 2.2em "Comic Sans MS","Comic Sans",cursive;color:#f0f;text-shadow:3px 3px #00f;'
    "margin:0.2em 0}"
    'h2{font:bold 1.3em Impact,"Arial Black",sans-serif;color:#0ff;letter-spacing:1px;'
    "border-bottom:3px ridge #f0f;margin:1em 0 0.4em}"
    "table.frame{width:100%;max-width:62em;margin:0 auto;border:4px outset #c0c0c0;"
    "background:#000080;border-collapse:separate;border-spacing:0}td.main{padding:10px}"
    "marquee{display:block;background:#f00;color:#ff0;font:bold 1.1em Arial,Helvetica,sans-serif;"
    "padding:4px;border:3px double #ff0;max-width:62em;margin:0 auto 8px}"
    "nav{font:bold 1em Arial,Helvetica,sans-serif;color:#fff}nav a{margin:0 0.3em}"
    "img.divider{display:block;width:100%;height:6px;margin:8px 0}"
    "ul{padding-left:1.5em}li{margin:0.2em 0}sup a{text-decoration:none}"
    "ol li{color:#c0c0c0}p{color:#fff}"
    'input{font:1em "Courier New",Courier,monospace;width:100%;padding:0.3em;'
    "box-sizing:border-box;background:#c0c0c0;border:2px inset #808080;color:#000}"
    "#hits{list-style:none;padding:0}form.search{display:flex;gap:8px;align-items:center}"
    ".blink{animation:blink 1s steps(1) infinite}@keyframes blink{50%{opacity:0}}"
    ".center{text-align:center}.badges img{margin:3px;border:1px solid #fff}"
    '.counter{font:bold 1em "Courier New",Courier,monospace;color:#ff0}'
    ".counter img{vertical-align:middle;border:2px inset #808080}"
    "@media(prefers-color-scheme:dark){body{background-color:#000}}"
)
MARQUEE: Final = (
    "*** WELCOME TO THE INFOVORE SGI LORE WIKI *** BEST VIEWED IN NETSCAPE NAVIGATOR 3.0"
    " AT 800X600 *** THIS SITE IS UNDER CONSTRUCTION *** SIGN THE GUESTBOOK ***"
)


SCRIPT: Final = """
(function (items) {
  var q = document.getElementById('q'), hits = document.getElementById('hits');
  function rank(s, i) { return (i.t.toLowerCase().indexOf(s) === 0 ? 0 : 1) * 1e6 - i.n; }
  q.addEventListener('input', function () {
    var s = q.value.trim().toLowerCase();
    hits.textContent = '';
    if (!s) { return; }
    items.filter(function (i) { return i.t.toLowerCase().indexOf(s) >= 0; })
      .sort(function (a, b) { return rank(s, a) - rank(s, b); }).slice(0, 20)
      .forEach(function (i) {
        var li = document.createElement('li'), a = document.createElement('a');
        a.href = i.f;
        a.textContent = i.t + (i.a ? ', see ' + i.a : '') + (i.n ? ' (' + i.n + ')' : '');
        li.appendChild(a);
        hits.appendChild(li);
      });
  });
  q.focus();
})(ITEMS);
"""


def html_name(markdown_name: str) -> str:
    return re.sub(r"\.md$", ".html", markdown_name)


def inline(text: str) -> str:
    out = html.escape(text, quote=False)

    def link(m: re.Match[str]) -> str:
        href = m.group(2)
        if not href.startswith(("http://", "https://")):
            href = html_name(href)
        return f'<a href="{html.escape(href)}">{m.group(1)}</a>'

    out = _LINK.sub(link, out)
    return _CITE.sub(r'<sup><a href="#s\1">\1</a></sup>', out)


def _close(stack: list[str], out: list[str], depth: int) -> None:
    while len(stack) > depth:
        out.append(f"</li></{stack.pop()}>")


def body(markdown: str) -> str:
    out: list[str] = []
    stack: list[str] = []
    paragraph: list[str] = []
    numbered = False

    def flush() -> None:
        if paragraph:
            out.append("<p>" + inline(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    for line in markdown.splitlines():
        if m := _BULLET.match(line):
            flush()
            depth = len(m.group(1)) // 2 + 1
            if depth > len(stack):
                out.append("<ul>")
                stack.append("ul")
            else:
                _close(stack, out, depth)
                out.append("</li>")
            out.append("<li>" + inline(m.group(2)))
        elif m := _NUMBERED.match(line):
            flush()
            if not numbered:
                out.append("<ol>")
                numbered = True
            out.append(f'<li id="s{m.group(1)}">{inline(m.group(2))}</li>')
        else:
            _close(stack, out, 0)
            if numbered and (not line.strip() or line.startswith("#")):
                out.append("</ol>")
                numbered = False
            if line.startswith("# "):
                out.append(f"<h1>{inline(line[2:])}</h1>")
            elif line.startswith("## "):
                flush()
                out.append(f'<h2 id="{slug(line[3:])}">{inline(line[3:])}</h2>')
            elif line.strip():
                paragraph.append(line)
            else:
                flush()
    flush()
    _close(stack, out, 0)
    if numbered:
        out.append("</ol>")
    return "\n".join(out) + "\n"


def page(markdown: str, search: Sequence[Mapping[str, str | int]] | None = None) -> str:
    title = next((ln[2:] for ln in markdown.splitlines() if ln.startswith("# ")), "Wiki")
    search_box = (
        '<form class="search" onsubmit="return false"><img src="spinner.gif" width="24"'
        ' height="24" alt=""><input id="q" type="search" placeholder="Search topics"></form>\n'
        '<ul id="hits"></ul>\n'
    )
    items = json.dumps(list(search or [])).replace("</", "<\\/")
    badges = "".join(
        f'<img src="{name}" width="88" height="31" alt="{top} {bottom}">'
        for name, top, bottom, *_ in BADGES
    )
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n"
        f"<marquee>{MARQUEE}</marquee>\n"
        '<table class="frame"><tr><td class="main">\n'
        '<nav>[ <a href="index.html">Index</a> | <a href="all.html">All topics</a> |'
        f' <a href="{GUESTBOOK}">Guestbook</a> ]'
        ' <img src="new.gif" width="32" height="16" alt="New"></nav>\n'
        '<img class="divider" src="divider.gif" alt="">\n'
        + (search_box if search is not None else "")
        + body(markdown)
        + '<img class="divider" src="divider.gif" alt="">\n'
        + '<p class="center"><img src="construction.gif" width="160" height="40"'
        ' alt="Under construction"></p>\n'
        + f'<p class="center badges">{badges}</p>\n'
        + '<p class="center counter">Claims served: <img src="counter.gif" alt="counter"></p>\n'
        + '<p class="center"><span class="blink">Best viewed at 800x600.</span>'
        f' Sign my <a href="{GUESTBOOK}">guestbook</a>. No frames, no Java, no cookies.</p>\n'
        "</td></tr></table>\n"
        + (f"<script>var ITEMS = {items};{SCRIPT}</script>\n" if search is not None else "")
        + "</body>\n</html>\n"
    )


def search_entries(
    sizes: Mapping[str, int], redirects: Mapping[str, str], files: Mapping[str, str]
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for name in sorted(sizes, key=lambda n: (-sizes[n], n)):
        rows.append({"t": name, "f": html_name(files[name]), "n": sizes[name]})
    for variant in sorted(redirects):
        target = redirects[variant]
        rows.append({"t": variant, "f": html_name(files[target]), "n": sizes[target], "a": target})
    return rows


def write_html(
    out: Path,
    markdown_files: Sequence[str],
    sizes: Mapping[str, int],
    redirects: Mapping[str, str],
    files: Mapping[str, str],
) -> list[str]:
    written = []
    entries = search_entries(sizes, redirects, files)
    for name in markdown_files:
        text = page((out / name).read_text(), entries if name == "index.md" else None)
        (out / html_name(name)).write_text(text)
        written.append(html_name(name))
    (out / SEARCH).write_text(json.dumps(entries))
    return [*written, SEARCH, *write_gifs(out, sum(sizes.values()))]
