import html
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

SEARCH: Final = "search.json"
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_CITE = re.compile(r"\[(\d+)\]")
_BULLET = re.compile(r"^( *)- (.*)$")
_NUMBERED = re.compile(r"^(\d+)\. (.*)$")
STYLE: Final = (
    "body{font:16px/1.5 system-ui,sans-serif;max-width:60em;margin:1em auto;padding:0 1em;"
    "color:#222}nav a{margin-right:1em}h1{margin-top:0.5em}ul{padding-left:1.5em}"
    "li{margin:0.2em 0}sup a{text-decoration:none}input{font:inherit;width:100%;padding:0.4em;"
    "box-sizing:border-box}#hits{list-style:none;padding:0}"
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
                out.append(f"<h2>{inline(line[3:])}</h2>")
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
    search_box = '<input id="q" type="search" placeholder="Search topics">\n<ul id="hits"></ul>\n'
    items = json.dumps(list(search or [])).replace("</", "<\\/")
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n"
        '<nav><a href="index.html">Index</a><a href="all.html">All topics</a></nav>\n'
        + (search_box if search is not None else "")
        + body(markdown)
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
    return [*written, SEARCH]
