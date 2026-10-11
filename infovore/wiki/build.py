import hashlib
import sqlite3
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from infovore.claims.gate import claim_has_tech
from infovore.claims.speakers import dropped_pairs
from infovore.db.novelty import novelty_for, wiki_url
from infovore.db.wiki_tags import tags_for
from infovore.triage.lexicon import load_lexicon
from infovore.wiki.canon import (
    Merged,
    alias_names,
    canonical,
    display_names,
    merge_map,
    spelling_counts,
    summarise_merge,
)
from infovore.wiki.eligibility import REJECTED_REVIEWS, UNCHECKED, is_publishable
from infovore.wiki.kinds import KINDS, LABELS, parents
from infovore.wiki.topics import Topics, disambiguations, load_topics, slug, topic_redirects

if TYPE_CHECKING:
    from infovore.wiki.groups import ClaimGroup, Grouper

BUCKETS: Final = ((1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 24, "10-24"))
BUCKETS_TAIL: Final = ((25, 99, "25-99"), (100, 10**9, "100+"))
GENERAL: Final = "General"
UNSORTED: Final = "Unsorted"
INDEX: Final = "index"
ALL: Final = "all"
INDEX_MIN: Final = 10
PART_SEPARATOR: Final = "#"
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
    novelty: float | None = None
    reference: str | None = None


@dataclass(frozen=True)
class Navigation:
    kinds: Mapping[str, str] = field(default_factory=dict)
    redirects: Mapping[str, str] = field(default_factory=dict)
    disambiguations: Mapping[str, Sequence[str]] = field(default_factory=dict)
    index_min: int = INDEX_MIN


def established(claim: WikiClaim, background: float | None) -> bool:
    return background is not None and claim.novelty is not None and claim.novelty >= background


@dataclass(frozen=True)
class Stats:
    topics: int
    pages: int
    unassigned: int
    claims_per_page: dict[str, int]
    on_pages: int = 0
    background: int = 0

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


@dataclass(frozen=True)
class Excluded:
    by_reason: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.by_reason.values())

    def __str__(self) -> str:
        parts = ", ".join(f"{k} {v}" for k, v in sorted(self.by_reason.items()))
        return f"{self.total}" + (f" ({parts})" if parts else "")


def load_claims(
    conn: sqlite3.Connection,
    tech_only: bool = False,
    salt: str | None = None,
    runs: Sequence[int] | None = None,
    tag_runs: Sequence[int] | None = None,
    verdicts: frozenset[str] | None = None,
    background: float | None = None,
) -> tuple[list[WikiClaim], Excluded]:
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
    excluded: dict[str, int] = {}
    for row in rows:
        if (row["exchange_id"], row["speaker"]) in dropped:
            reason: str | None = "speaker"
        elif row["review"] in REJECTED_REVIEWS:
            reason = "review"
        elif not is_publishable(row["review"], row["chk"], verdicts):
            reason = f"check:{row['chk'] or UNCHECKED}"
        else:
            reason = None
        statement = _line(row["statement"])
        if reason is None and lexicon and not claim_has_tech(lexicon, topics, statement):
            reason = "no_tech"
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
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
    if tag_runs:
        loaded = _tagged(conn, loaded, tag_runs, topics)
    if background is not None:
        loaded = _with_novelty(conn, loaded)
    return loaded, Excluded(excluded)


def _with_novelty(conn: sqlite3.Connection, claims: list[WikiClaim]) -> list[WikiClaim]:
    rows = novelty_for(conn, [c.claim_id for c in claims])
    return [
        replace(c, novelty=rows[c.claim_id].max_cosine, reference=rows[c.claim_id].reference_title)
        if c.claim_id in rows
        else c
        for c in claims
    ]


def _tagged(
    conn: sqlite3.Connection, claims: list[WikiClaim], tag_runs: Sequence[int], topics: Topics
) -> list[WikiClaim]:
    tags = tags_for(conn, tag_runs)
    used = [t for c in claims for t in tags.get(c.claim_id, [])]
    merge = merge_map(spelling_counts(used))
    redirects = {canonical(v): name for v, name in topic_redirects().items()}
    names = {**display_names(used, merge), **alias_names(topics), **redirects}
    ambiguous = set(disambiguations())
    out = []
    for c in claims:
        keys = {canonical(t) for t in tags.get(c.claim_id, [])} - {""}
        topics_of = {names[merge.get(k, k)] for k in keys} - ambiguous
        out.append(replace(c, topics=frozenset(topics_of)))
    return out


def redirect_names(conn: sqlite3.Connection, tag_runs: Sequence[int]) -> dict[str, str]:
    used = [t for names in tags_for(conn, tag_runs).values() for t in names]
    merge = merge_map(spelling_counts(used))
    plain = display_names(used)
    names = {**display_names(used, merge), **alias_names(load_topics())}
    names.update({canonical(v): t for v, t in topic_redirects().items()})
    out = {plain[key]: names[target] for key, target in merge.items()}
    out.update({**alias_names(load_topics()), **topic_redirects()})
    return {variant: name for variant, name in out.items() if variant != name}


def merged_tags(conn: sqlite3.Connection, tag_runs: Sequence[int]) -> Merged:
    spellings = spelling_counts(t for names in tags_for(conn, tag_runs).values() for t in names)
    return summarise_merge(spellings, merge_map(spellings))


def _ordered(claims: Iterable[WikiClaim]) -> list[WikiClaim]:
    return sorted(claims, key=lambda c: (c.date, c.exchange_id, c.claim_id))


def _by_topic(
    claims: Iterable[WikiClaim], unsorted: str | None = None
) -> dict[str, list[WikiClaim]]:
    out: dict[str, list[WikiClaim]] = {}
    for c in _ordered(claims):
        if not c.topics and unsorted:
            out.setdefault(unsorted, []).append(c)
        for name in sorted(c.topics):
            out.setdefault(name, []).append(c)
    return out


def _entry(g: "ClaimGroup") -> str:
    c = g.lead
    similar = len(g.members) - 1
    more = f" (+{similar} similar)" if similar else ""
    first, last = g.span
    when = first if first == last else f"{first} to {last}"
    return (
        f"- {_line(c.statement)}{more} ({c.speaker}, exchange {c.exchange_id}, {when};"
        f" {len(g.speakers)} speaker(s), {len(g.exchanges)} conversation(s))"
    )


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


def part_key(section: str, part: int) -> str:
    return section if part == 1 else f"{section}{PART_SEPARATOR}{part}"


def article_parts(article: Article, key: str) -> list[tuple[str, Sequence[int]]]:
    out = list(article.get(key, []))
    part = 2
    while part_key(key, part) in article:
        out.extend(article[part_key(key, part)])
        part += 1
    return out


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


def page_files(names: Iterable[str], weights: Mapping[str, int] | None = None) -> dict[str, str]:
    taken = {INDEX, ALL}
    out: dict[str, str] = {}
    for name in sorted(names, key=lambda n: (-(weights or {}).get(n, 0), n)):
        base = slug(name) or f"topic-{hashlib.sha1(name.encode()).hexdigest()[:8]}"
        candidate, n = base, 1
        while candidate in taken:
            n += 1
            candidate = f"{base}-{n}"
        taken.add(candidate)
        out[name] = f"{candidate}.md"
    return out


def _background_line(settled: Collection[WikiClaim]) -> str:
    refs = Counter(c.reference for c in settled if c.reference)
    noun = "claim" if len(settled) == 1 else "claims"
    line = f"Background: {len(settled)} {noun} already covered by Wikipedia"
    if refs:
        titles = sorted(refs, key=lambda title: (-refs[title], title))
        line += ", see " + ", ".join(f"[{t}]({wiki_url(t)})" for t in titles)
    return line + "."


def render_page(
    name: str,
    claims: Collection[WikiClaim],
    files: Mapping[str, str],
    article: Article | None = None,
    min_section: int = 1,
    grouper: "Grouper | None" = None,
    background: float | None = None,
    nav: Navigation | None = None,
    parent: str | None = None,
    children: Mapping[str, int] | None = None,
) -> str:
    from infovore.wiki.groups import group_claims

    group = grouper or group_claims
    cooc = Counter(t for c in claims for t in c.topics if t != name)
    settled = [c for c in claims if established(c, background)]
    fresh = [c for c in claims if not established(c, background)]
    groups = sections_of(name, fresh, min_section)
    known = {c.claim_id: c for c in claims}
    numbers: dict[int, int] = {}
    lines = [f"# {name}", "", f"Claims: {len(claims)}"]
    if parent:
        lines.append(f"Part of [{parent}]({files[parent]})")
    if settled:
        lines.append(_background_line(settled))
    for key, members in groups.items():
        lines.append("")
        if list(groups) != [GENERAL]:
            lines += [f"## {key}", ""]
        prose = _prose(article_parts(article or {}, key), known, numbers)
        if prose:
            lines.append(prose)
        else:
            lines += [_entry(g) for g in group(members)]
    if numbers:
        lines += ["", "## Sources", ""]
        for claim_id, n in numbers.items():
            c = known[claim_id]
            lines.append(
                f"{n}. {c.statement} ({c.speaker}, {c.date}, exchange {c.exchange_id},"
                f" check: {c.check or 'unchecked'})"
            )
    if children:
        lines += ["", "## Subtopics", ""]
        lines += [_link(t, files, children) for t in _ranked(children)]
    skip = set(children or {}) | {parent}
    related = sorted((t for t in cooc if t in files and t not in skip), key=lambda t: (-cooc[t], t))
    if related:
        lines += ["", "## See also", ""]
        lines += _see_also(related, files, nav.kinds if nav else {})
    return "\n".join(lines) + "\n"


def _ranked(sizes: Mapping[str, int]) -> list[str]:
    return sorted(sizes, key=lambda n: (-sizes[n], n))


def _link(name: str, files: Mapping[str, str], sizes: Mapping[str, int]) -> str:
    return f"- [{name}]({files[name]}) ({sizes[name]})"


def _see_also(
    related: Sequence[str], files: Mapping[str, str], kinds: Mapping[str, str]
) -> list[str]:
    if not kinds:
        return [f"- [{t}]({files[t]})" for t in related]
    out = []
    for kind in KINDS:
        names = [t for t in related if kinds.get(t, KINDS[-1]) == kind]
        if names:
            out.append(f"- {LABELS[kind]}: " + ", ".join(f"[{t}]({files[t]})" for t in names))
    return out


def render_redirect(name: str, target: str, files: Mapping[str, str]) -> str:
    return f"# {name}\n\nSee [{target}]({files[target]}).\n"


def render_disambiguation(
    name: str, candidates: Sequence[str], sizes: Mapping[str, int], files: Mapping[str, str]
) -> str:
    lines = [f"# {name}", "", f"{name} may refer to:", ""]
    lines += [_link(c, files, sizes) if c in sizes else f"- {c}" for c in candidates]
    return "\n".join(lines) + "\n"


def render_all(pages: Mapping[str, int], files: Mapping[str, str]) -> str:
    lines = ["# All topics", "", f"{len(pages)} pages. [Index]({INDEX}.md)"]
    letter = None
    for name in sorted(pages, key=lambda n: (n.casefold(), n)):
        first = name[0].upper() if name[0].isalpha() else "#"
        if first != letter:
            letter = first
            lines += ["", f"## {letter}", ""]
        lines.append(_link(name, files, pages))
    return "\n".join(lines) + "\n"


def render_index(
    pages: Mapping[str, int], files: Mapping[str, str], nav: Navigation | None = None
) -> str:
    lines = ["# Index", ""]
    if nav is None:
        lines += [_link(name, files, pages) for name in sorted(pages)]
        return "\n".join(lines) + "\n"
    unsorted = {n: pages[n] for n in pages if n == UNSORTED}
    topics = {n: pages[n] for n in pages if n != UNSORTED}
    parent = parents(topics)
    children: dict[str, list[str]] = {}
    for child, p in parent.items():
        children.setdefault(p, []).append(child)
    lines.append(f"{len(topics)} topics, {_count(topics)} claims. [All topics A-Z]({ALL}.md)")
    shown = [n for n in _ranked(topics) if n not in parent and pages[n] >= nav.index_min]
    for kind in KINDS:
        names = [n for n in shown if nav.kinds.get(n, KINDS[-1]) == kind]
        if names:
            lines += ["", f"## {LABELS[kind]}", ""]
            for name in names:
                lines += _tree(name, children, pages, files, nav.index_min, 0)
    if unsorted:
        lines += ["", "## Unsorted", "", _link(UNSORTED, files, unsorted)]
    return "\n".join(lines) + "\n"


def _count(pages: Mapping[str, int]) -> int:
    return sum(pages.values())


def _tree(
    name: str,
    children: Mapping[str, Sequence[str]],
    pages: Mapping[str, int],
    files: Mapping[str, str],
    index_min: int,
    depth: int,
) -> list[str]:
    lines = ["  " * depth + _link(name, files, pages)]
    kids = {c: pages[c] for c in children.get(name, []) if pages[c] >= index_min}
    for child in _ranked(kids):
        lines += _tree(child, children, pages, files, index_min, depth + 1)
    return lines


def compute_stats(
    claims: Iterable[WikiClaim], min_claims: int, background: float | None = None
) -> Stats:
    claims = list(claims)
    by_topic = _by_topic(claims, UNSORTED)
    return Stats(
        topics=len(_by_topic(claims)),
        pages=sum(1 for v in by_topic.values() if len(v) >= min_claims),
        unassigned=sum(1 for c in claims if not c.topics),
        claims_per_page={n: len(v) for n, v in sorted(by_topic.items()) if len(v) >= min_claims},
        on_pages=len({c.claim_id for v in by_topic.values() if len(v) >= min_claims for c in v}),
        background=sum(1 for c in claims if established(c, background)),
    )


def page_groups(
    claims: Iterable[WikiClaim], min_claims: int, grouper: "Grouper"
) -> list[list["ClaimGroup"]]:
    by_topic = {n: v for n, v in _by_topic(claims, UNSORTED).items() if len(v) >= min_claims}
    return [grouper(members) for members in by_topic.values()]


def build_site(
    claims: Iterable[WikiClaim],
    min_claims: int,
    out: Path,
    articles: Mapping[str, Article] | None = None,
    min_section: int = 1,
    grouper: "Grouper | None" = None,
    background: float | None = None,
    nav: Navigation | None = None,
) -> list[str]:
    by_topic = {n: v for n, v in _by_topic(claims, UNSORTED).items() if len(v) >= min_claims}
    out.mkdir(parents=True, exist_ok=True)
    sizes = {n: len(v) for n, v in by_topic.items()}
    redirects = {v: t for v, t in (nav.redirects if nav else {}).items() if t in sizes}
    redirects = {v: t for v, t in redirects.items() if v not in sizes}
    pages = {n: sizes[n] for n in sizes if n != UNSORTED}
    ambiguous = {
        n: c
        for n, c in (nav.disambiguations if nav else {}).items()
        if n not in sizes and any(x in sizes for x in c)
    }
    extra = dict.fromkeys([*redirects, *ambiguous], 0)
    names = page_files({**sizes, **extra}, {**sizes, **extra})
    parent = parents(pages)
    kids: dict[str, dict[str, int]] = {}
    for child, p in parent.items():
        kids.setdefault(p, {})[child] = sizes[child]
    files = {f"{INDEX}.md": render_index(sizes, names, nav)}
    if nav is not None:
        files[f"{ALL}.md"] = render_all(pages, names)
    for name, group in by_topic.items():
        files[names[name]] = render_page(
            name,
            group,
            names,
            (articles or {}).get(name),
            min_section,
            grouper,
            background,
            nav,
            parent.get(name),
            kids.get(name),
        )
    for variant, target in redirects.items():
        files[names[variant]] = render_redirect(variant, target, names)
    for name, candidates in ambiguous.items():
        files[names[name]] = render_disambiguation(name, candidates, sizes, names)
    for filename, text in files.items():
        (out / filename).write_text(text)
    return sorted(files)
