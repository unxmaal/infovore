import sqlite3
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from infovore.claims.gate import claim_has_tech
from infovore.claims.speakers import dropped_pairs
from infovore.db.wiki_tags import tags_for
from infovore.triage.lexicon import load_lexicon
from infovore.wiki.canon import alias_names, canonical, display_names
from infovore.wiki.eligibility import is_publishable
from infovore.wiki.topics import Topics, load_topics, slug

BUCKETS: Final = ((1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 24, "10-24"))
BUCKETS_TAIL: Final = ((25, 99, "25-99"), (100, 10**9, "100+"))
GENERAL: Final = "General"
DEFAULT_MIN_SECTION: Final = 10


@dataclass(frozen=True)
class WikiClaim:
    claim_id: int
    exchange_id: int
    date: str
    speaker: str
    statement: str
    topics: frozenset[str]
    check: str | None = None


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


def load_claims(
    conn: sqlite3.Connection,
    tech_only: bool = False,
    salt: str | None = None,
    runs: Sequence[int] | None = None,
    tag_runs: Sequence[int] | None = None,
) -> tuple[list[WikiClaim], int]:
    topics = load_topics()
    lexicon = load_lexicon(conn) if tech_only else None
    dropped = dropped_pairs(conn, salt)
    rows = conn.execute(
        "SELECT c.id, c.exchange_id, c.speaker, c.statement, substr(e.started_at, 1, 10) AS day,"
        " r.verdict AS review, k.verdict AS chk FROM claims_v2 c"
        " JOIN current_exchanges e ON e.id = c.exchange_id"
        " LEFT JOIN current_claim_reviews r ON r.claim_id = c.id"
        " LEFT JOIN current_claim_checks k ON k.claim_id = c.id"
        + (f" WHERE c.run_id IN ({','.join('?' for _ in runs)})" if runs else "")
        + " ORDER BY c.id",
        list(runs or []),
    )
    claims: dict[tuple[int, str, str], WikiClaim] = {}
    excluded = 0
    for row in rows:
        if (row["exchange_id"], row["speaker"]) in dropped or not is_publishable(
            row["review"], row["chk"]
        ):
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
                frozenset() if tag_runs else subjects(topics, statement),
                row["chk"],
            ),
        )
    loaded = list(claims.values())
    return (_tagged(conn, loaded, tag_runs, topics) if tag_runs else loaded), excluded


def _tagged(
    conn: sqlite3.Connection, claims: list[WikiClaim], tag_runs: Sequence[int], topics: Topics
) -> list[WikiClaim]:
    tags = tags_for(conn, tag_runs)
    names = {
        **display_names(t for c in claims for t in tags.get(c.claim_id, [])),
        **alias_names(topics),
    }
    out = []
    for c in claims:
        keys = {canonical(t) for t in tags.get(c.claim_id, [])} - {""}
        out.append(replace(c, topics=frozenset(names[k] for k in keys)))
    return out


def _ordered(claims: Iterable[WikiClaim]) -> list[WikiClaim]:
    return sorted(claims, key=lambda c: (c.date, c.exchange_id, c.claim_id))


def _by_topic(claims: Iterable[WikiClaim]) -> dict[str, list[WikiClaim]]:
    out: dict[str, list[WikiClaim]] = {}
    for c in _ordered(claims):
        for name in sorted(c.topics):
            out.setdefault(name, []).append(c)
    return out


def _entry(c: WikiClaim, similar: int = 0) -> str:
    more = f" (+{similar} similar)" if similar else ""
    return f"- {_line(c.statement)}{more} ({c.speaker}, exchange {c.exchange_id}, {c.date})"


Article = Mapping[str, Sequence[tuple[str, Sequence[int]]]]


def _prose(
    sentences: Iterable[tuple[str, Sequence[int]]],
    known: Mapping[int, WikiClaim],
    numbers: dict[int, int],
) -> str:
    parts = []
    for text, ids in sentences:
        cited = [i for i in dict.fromkeys(ids) if i in known]
        if cited:
            for i in cited:
                numbers.setdefault(i, len(numbers) + 1)
            parts.append(f"{text} " + "".join(f"[{numbers[i]}]" for i in cited))
    return " ".join(parts)


def sections_of(
    name: str, claims: Collection[WikiClaim], min_section: int = 1
) -> dict[str, list[WikiClaim]]:
    cooc = Counter(t for c in claims for t in c.topics if t != name)
    ordered = _ordered(claims)
    keys = []
    for c in ordered:
        others = [t for t in c.topics if t != name]
        keys.append(min(others, key=lambda t: (-cooc[t], t)) if others else GENERAL)
    sizes = Counter(keys)
    groups: dict[str, list[WikiClaim]] = {}
    for c, key in zip(ordered, keys, strict=True):
        if key == GENERAL or sizes[key] < min_section:
            groups.setdefault(GENERAL, []).append(c)
        else:
            groups.setdefault(f"With {key}", []).append(c)
    return {k: groups[k] for k in sorted(groups, key=lambda k: (k == GENERAL, -len(groups[k]), k))}


def render_page(
    name: str,
    claims: Collection[WikiClaim],
    page_names: Collection[str],
    article: Article | None = None,
    min_section: int = 1,
) -> str:
    from infovore.wiki.groups import group_claims

    cooc = Counter(t for c in claims for t in c.topics if t != name)
    groups = sections_of(name, claims, min_section)
    known = {c.claim_id: c for c in claims}
    numbers: dict[int, int] = {}
    lines = [f"# {name}", "", f"Claims: {len(claims)}"]
    for key, members in groups.items():
        lines.append("")
        if list(groups) != [GENERAL]:
            lines += [f"## {key}", ""]
        prose = _prose((article or {}).get(key, []), known, numbers)
        if prose:
            lines.append(prose)
        else:
            lines += [_entry(g.lead, len(g.members) - 1) for g in group_claims(members)]
    if numbers:
        lines += ["", "## Sources", ""]
        for claim_id, n in numbers.items():
            c = known[claim_id]
            lines.append(
                f"{n}. {c.statement} ({c.speaker}, {c.date}, exchange {c.exchange_id},"
                f" check: {c.check or 'unchecked'})"
            )
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


def build_site(
    claims: Iterable[WikiClaim],
    min_claims: int,
    out: Path,
    articles: Mapping[str, Article] | None = None,
    min_section: int = 1,
) -> list[str]:
    by_topic = {n: v for n, v in _by_topic(claims).items() if len(v) >= min_claims}
    out.mkdir(parents=True, exist_ok=True)
    files = {"index.md": render_index({n: len(v) for n, v in by_topic.items()})}
    for name, group in by_topic.items():
        files[f"{slug(name)}.md"] = render_page(
            name, group, by_topic.keys(), (articles or {}).get(name), min_section
        )
    for filename, text in files.items():
        (out / filename).write_text(text)
    return sorted(files)
