import sqlite3
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from infovore.claims.gate import claim_has_tech
from infovore.triage.lexicon import load_lexicon
from infovore.wiki.eligibility import is_publishable
from infovore.wiki.topics import Topics, load_topics, slug

BUCKETS: Final = ((1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 24, "10-24"))
BUCKETS_TAIL: Final = ((25, 99, "25-99"), (100, 10**9, "100+"))
GENERAL: Final = "General"


@dataclass(frozen=True)
class WikiClaim:
    claim_id: int
    exchange_id: int
    date: str
    speaker: str
    statement: str
    topics: frozenset[str]


@dataclass(frozen=True)
class Stats:
    topics: int
    pages: int
    unassigned: int
    claims_per_page: dict[str, int]

    def distribution(self) -> dict[str, int]:
        sizes = Counter(self.claims_per_page.values())
        out: dict[str, int] = {}
        for low, high, label in (*BUCKETS, *BUCKETS_TAIL):
            total = sum(n for size, n in sizes.items() if low <= size <= high)
            if total:
                out[label] = total
        return out


def _line(text: str) -> str:
    return " ".join(text.split())


def subjects(topics: Topics, statement: str) -> frozenset[str]:
    return topics.assign(_line(statement))


def load_claims(conn: sqlite3.Connection, tech_only: bool = False) -> tuple[list[WikiClaim], int]:
    topics = load_topics()
    lexicon = load_lexicon(conn) if tech_only else None
    rows = conn.execute(
        "SELECT c.id, c.exchange_id, c.speaker, c.statement, substr(e.started_at, 1, 10) AS day,"
        " r.verdict AS review, k.verdict AS chk FROM claims_v2 c"
        " JOIN current_exchanges e ON e.id = c.exchange_id"
        " LEFT JOIN current_claim_reviews r ON r.claim_id = c.id"
        " LEFT JOIN current_claim_checks k ON k.claim_id = c.id ORDER BY c.id"
    )
    claims: dict[tuple[int, str, str], WikiClaim] = {}
    excluded = 0
    for row in rows:
        if not is_publishable(row["review"], row["chk"]):
            excluded += 1
            continue
        statement = _line(row["statement"])
        if lexicon and not claim_has_tech(lexicon, topics, statement):
            excluded += 1
            continue
        key = (row["exchange_id"], row["speaker"], statement)
        claims.setdefault(
            key,
            WikiClaim(
                row["id"],
                row["exchange_id"],
                row["day"],
                row["speaker"],
                statement,
                subjects(topics, statement),
            ),
        )
    return list(claims.values()), excluded


def _ordered(claims: Iterable[WikiClaim]) -> list[WikiClaim]:
    return sorted(claims, key=lambda c: (c.date, c.exchange_id, c.claim_id))


def _by_topic(claims: Iterable[WikiClaim]) -> dict[str, list[WikiClaim]]:
    out: dict[str, list[WikiClaim]] = {}
    for c in _ordered(claims):
        for name in sorted(c.topics):
            out.setdefault(name, []).append(c)
    return out


def _entry(c: WikiClaim) -> str:
    return f"- {_line(c.statement)} ({c.speaker}, exchange {c.exchange_id}, {c.date})"


def render_page(name: str, claims: Collection[WikiClaim], page_names: Collection[str]) -> str:
    cooc = Counter(t for c in claims for t in c.topics if t != name)
    groups: dict[str, list[WikiClaim]] = {}
    for c in _ordered(claims):
        others = [t for t in c.topics if t != name]
        key = min(others, key=lambda t: (-cooc[t], t)) if others else GENERAL
        groups.setdefault(key, []).append(c)
    lines = [f"# {name}", "", f"Claims: {len(claims)}"]
    grouped = list(groups) != [GENERAL]
    for key in sorted(groups, key=lambda k: (k == GENERAL, -len(groups[k]), k)):
        lines.append("")
        if grouped:
            lines += [f"## {key if key == GENERAL else f'With {key}'}", ""]
        lines += [_entry(c) for c in groups[key]]
    related = sorted((t for t in cooc if t in page_names), key=lambda t: (-cooc[t], t))
    if related:
        lines += ["", "## See also", ""]
        lines += [f"- [{t}]({slug(t)}.md)" for t in related]
    return "\n".join(lines) + "\n"


def render_index(pages: Mapping[str, int]) -> str:
    lines = ["# Index", ""]
    lines += [f"- [{name}]({slug(name)}.md) ({pages[name]})" for name in sorted(pages)]
    return "\n".join(lines) + "\n"


def compute_stats(claims: Iterable[WikiClaim], min_claims: int) -> Stats:
    claims = list(claims)
    by_topic = _by_topic(claims)
    return Stats(
        topics=len(by_topic),
        pages=sum(1 for v in by_topic.values() if len(v) >= min_claims),
        unassigned=sum(1 for c in claims if not c.topics),
        claims_per_page={n: len(v) for n, v in sorted(by_topic.items()) if len(v) >= min_claims},
    )


def build_site(claims: Iterable[WikiClaim], min_claims: int, out: Path) -> list[str]:
    by_topic = {n: v for n, v in _by_topic(claims).items() if len(v) >= min_claims}
    out.mkdir(parents=True, exist_ok=True)
    files = {"index.md": render_index({n: len(v) for n, v in by_topic.items()})}
    for name, group in by_topic.items():
        files[f"{slug(name)}.md"] = render_page(name, group, by_topic.keys())
    for filename, text in files.items():
        (out / filename).write_text(text)
    return sorted(files)
