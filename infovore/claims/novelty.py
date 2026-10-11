import argparse
import json
import re
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, TextIO

from infovore.claims.value import tokens
from infovore.config import ConfigError
from infovore.db.claims_v2 import run_ids
from infovore.db.novelty import (
    Novelty,
    Reference,
    references,
    set_descriptions,
    store_novelty,
    store_reference,
    wiki_url,
)
from infovore.triage.embed import DEFAULT_MODEL, DEFAULT_REVISION, EMBED_BATCH, Embedder
from infovore.triage.embed_backend import load_embedder
from infovore.wiki.build import WikiClaim, _by_topic, load_claims
from infovore.wiki.eligibility import parse_verdicts
from infovore.wiki.groups import cosine

if TYPE_CHECKING:
    from infovore.cli import AppContext

WIKI_API: Final = "https://en.wikipedia.org/w/api.php"
USER_AGENT: Final = "infovore/0.1 (https://github.com/unxmaal/infovore)"
TIMEOUT: Final = 30.0
DEFAULT_MIN_CLAIMS: Final = 20
CANDIDATES: Final = 6
BROAD_CANDIDATES: Final = 10
DOMAIN_HINTS: Final = ("computer",)
TITLE_WEIGHT: Final = 0.35
FOCUS_WEIGHT: Final = 2.0
FOCUS_PER_1000: Final = 10.0
DIRECT_MARGIN: Final = 0.3
CO_TOPICS: Final = 20
MIN_MATCH: Final = 0.05
RETRIES: Final = 5
MIN_SENTENCE: Final = 30
COMMON_TOKENS: Final = 50
PAUSE: Final = 0.25
DESCRIBE_BATCH: Final = 20
NEAREST_CHARS: Final = 300
THRESHOLDS: Final = (0.7, 0.8, 0.9)
SHOW: Final = 10
Getter = Callable[[str], Any]
_HEADING = re.compile(r"^\s*=+.*=+\s*$", re.M)
_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_DISAMBIGUATION = re.compile(r"may (also )?refer to", re.I)
pause: Callable[[float], None] = time.sleep


def wiki_get(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRIES - 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.loads(response.read())
        except (OSError, ValueError):
            pause(PAUSE * 4 * (attempt + 1))
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read())


get: Getter = wiki_get


def _api(**params: str | int) -> str:
    return WIKI_API + "?" + urllib.parse.urlencode({**params, "format": "json"})


def search_titles(fetch: Getter, query: str, limit: int = CANDIDATES) -> list[str]:
    body = fetch(_api(action="query", list="search", srsearch=query, srlimit=limit))
    return [str(hit["title"]) for hit in body.get("query", {}).get("search", [])]


def fetch_extract(fetch: Getter, title: str) -> tuple[str, str, str]:
    props = "extracts|description"
    body = fetch(_api(action="query", prop=props, explaintext=1, redirects=1, titles=title))
    for page in body.get("query", {}).get("pages", {}).values():
        if "missing" not in page:
            title, text = str(page.get("title", title)), str(page.get("extract", ""))
            return title, text, str(page.get("description", ""))
    return title, "", ""


def fetch_descriptions(conn: Any, fetch: Getter, log: TextIO) -> int:
    missing = [r.title for r in references(conn).values() if r.found and not r.description]
    found: dict[str, str] = {}
    for start in range(0, len(missing), DESCRIBE_BATCH):
        titles = "|".join(missing[start : start + DESCRIBE_BATCH])
        body = fetch(_api(action="query", prop="description", redirects=1, titles=titles))
        for page in body.get("query", {}).get("pages", {}).values():
            if page.get("description"):
                found[str(page["title"])] = str(page["description"])
        pause(PAUSE)
    set_descriptions(conn, found)
    conn.commit()
    log.write(f"described {len(found)}/{len(missing)} references\n")
    return len(found)


def sentences(text: str) -> list[str]:
    plain = _HEADING.sub("", text)
    return [s.strip() for s in _SPLIT.split(plain) if len(s.strip()) >= MIN_SENTENCE]


def overlap(text: str, claim_tokens: frozenset[str]) -> float:
    words = tokens(text)
    if not words or not claim_tokens:
        return 0.0
    return float(len(words & claim_tokens) / (len(words) * len(claim_tokens)) ** 0.5)


def co_topics(topic: str, claims: Iterable[WikiClaim], n: int = CO_TOPICS) -> list[str]:
    counts = Counter(t for c in claims for t in c.topics if t != topic)
    return sorted(counts, key=lambda t: (-counts[t], t))[:n]


@dataclass(frozen=True)
class Candidate:
    title: str
    text: str
    direct: bool = False
    description: str = ""


def coverage(text: str, co_names: Sequence[str]) -> float:
    words = tokens(text)
    hits = sum(1 for name in co_names if tokens(name) and tokens(name) <= words)
    return hits / len(co_names) if co_names else 0.0


def title_match(topic: str, title: str) -> float:
    topic_words, title_words = tokens(topic), tokens(title)
    shared = len(topic_words & title_words)
    if not shared:
        return 0.0
    precision, recall = shared / len(title_words), shared / len(topic_words)
    return 2 * precision * recall / (precision + recall)


def focus(topic: str, text: str) -> float:
    words = _WORD.findall(text.lower())
    topic_words = tokens(topic)
    if not words or not topic_words:
        return 0.0
    hits = sum(1 for w in words if w in topic_words) / len(topic_words)
    return min(1.0, hits / len(words) * 1000 / FOCUS_PER_1000)


def fit(
    candidate: Candidate, co_names: Sequence[str], claim_tokens: frozenset[str]
) -> float | None:
    if not candidate.text or _DISAMBIGUATION.search(candidate.text[:300]):
        return None
    text = candidate.text
    value = coverage(text, co_names) if co_names else overlap(text, claim_tokens)
    return value if value >= MIN_MATCH else None


def match(
    topic: str, candidate: Candidate, co_names: Sequence[str], claim_tokens: frozenset[str]
) -> float:
    value = fit(candidate, co_names, claim_tokens)
    if value is None:
        return 0.0
    title, text = title_match(topic, candidate.title), focus(topic, candidate.text)
    if not candidate.direct and not title and not text:
        return 0.0
    return TITLE_WEIGHT * title + value + FOCUS_WEIGHT * text


def pick_reference(
    topic: str,
    candidates: Sequence[Candidate],
    claim_tokens: frozenset[str],
    co_names: Sequence[str] = (),
) -> Reference:
    fits = [fit(c, co_names, claim_tokens) for c in candidates]
    best = None
    for candidate, value in zip(candidates, fits, strict=True):
        if candidate.direct and value is not None:
            top = max(v for v in fits if v is not None)
            if top < value + DIRECT_MARGIN:
                best = candidate
            break
    if best is None:
        scores = [match(topic, c, co_names, claim_tokens) for c in candidates]
        if scores and max(scores) > 0:
            best = candidates[scores.index(max(scores))]
    if best is None:
        return Reference(topic, "", "", "", 0.0)
    url = wiki_url(best.title)
    found = fits[candidates.index(best)] or 0.0
    return Reference(topic, best.title, url, best.text, found, best.description)


def queries(topic: str, co_names: Sequence[str]) -> list[tuple[str, int]]:
    out = [(topic, CANDIDATES)]
    if co_names:
        out.append((f'intitle:"{topic}" {co_names[0]}', CANDIDATES))
        out.append((f"{topic} {' '.join(co_names[:3])}", BROAD_CANDIDATES))
    out += [(f'intitle:"{topic}" {hint}', CANDIDATES) for hint in DOMAIN_HINTS]
    return out


def fetch_references(
    conn: Any,
    fetch: Getter,
    by_topic: Mapping[str, Sequence[WikiClaim]],
    now: datetime,
    log: TextIO,
    limit: int | None = None,
) -> int:
    have = references(conn)
    todo = [t for t in sorted(by_topic, key=lambda t: (-len(by_topic[t]), t)) if t not in have]
    fetched = 0
    for topic in todo[:limit]:
        co_names = co_topics(topic, by_topic[topic])
        titles = [topic]
        for query, count in queries(topic, co_names):
            seen = {t.casefold() for t in titles}
            titles += [t for t in search_titles(fetch, query, count) if t.casefold() not in seen]
            pause(PAUSE)
        candidates = []
        for title in titles:
            found, text, description = fetch_extract(fetch, title)
            candidates.append(Candidate(found, text, title == topic, description))
            pause(PAUSE)
        claim_tokens = frozenset().union(*(tokens(c.statement) for c in by_topic[topic]))
        store_reference(conn, pick_reference(topic, candidates, claim_tokens, co_names), now)
        conn.commit()
        fetched += 1
        if fetched % 25 == 0:
            log.write(f"fetched {fetched}/{len(todo[:limit])} references\n")
    return fetched


def common_tokens(claims: Iterable[WikiClaim], n: int = COMMON_TOKENS) -> frozenset[str]:
    counts = Counter(t for c in claims for t in tokens(c.statement))
    return frozenset(t for t, _ in counts.most_common(n))


def generic_share(statement: str, generic: frozenset[str]) -> float:
    words = tokens(statement)
    return len(words & generic) / len(words) if words else 1.0


def _embed_all(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        out += embedder.embed(texts[start : start + EMBED_BATCH])
    return out


def score(
    embedder: Embedder,
    claims: Sequence[WikiClaim],
    refs: Mapping[str, Reference],
    generic: frozenset[str],
) -> list[Novelty]:
    by_ref = {topic: sentences(ref.text) for topic, ref in refs.items() if ref.found}
    vectors = {topic: _embed_all(embedder, sents) for topic, sents in by_ref.items()}
    claim_vectors = _embed_all(embedder, [c.statement for c in claims])
    out = []
    for claim, vector in zip(claims, claim_vectors, strict=True):
        best: tuple[float, str, int] | None = None
        for topic in sorted(claim.topics & by_ref.keys()):
            for i, sentence_vector in enumerate(vectors[topic]):
                value = cosine(vector, sentence_vector)
                if best is None or value > best[0]:
                    best = (value, topic, i)
        share = generic_share(claim.statement, generic)
        if best is None:
            out.append(Novelty(claim.claim_id, None, None, None, None, share))
        else:
            value, topic, i = best
            nearest = by_ref[topic][i][:NEAREST_CHARS]
            out.append(Novelty(claim.claim_id, topic, refs[topic].title, value, nearest, share))
    return out


@dataclass(frozen=True)
class Report:
    topics: int
    with_reference: int
    scored: int
    unscored: int
    quantiles: tuple[float, float, float]
    established: dict[float, int]
    least_novel: list[tuple[str, str, float, str]]
    most_novel: list[tuple[str, str, float, str]]
    by_topic: list[tuple[str, int, float]]


def summarise(
    claims: Sequence[WikiClaim], scored: Sequence[Novelty], refs: Mapping[str, Reference]
) -> Report:
    statements = {c.claim_id: c for c in claims}
    with_cos = [n for n in scored if n.max_cosine is not None]
    values = sorted(n.max_cosine for n in with_cos if n.max_cosine is not None)
    quantiles = (
        tuple(statistics.quantiles(values, n=10)[i] for i in (0, 4, 8))
        if len(values) >= 2
        else (0.0, 0.0, 0.0)
    )
    ordered = sorted(with_cos, key=lambda n: (-(n.max_cosine or 0), n.claim_id))

    def row(n: Novelty) -> tuple[str, str, float, str]:
        return (
            statements[n.claim_id].statement,
            n.reference_title or "",
            n.max_cosine or 0.0,
            n.nearest or "",
        )

    per_topic: dict[str, list[float]] = {}
    for n in with_cos:
        for topic in statements[n.claim_id].topics:
            if topic in refs and refs[topic].found:
                per_topic.setdefault(topic, []).append(n.max_cosine or 0.0)
    by_topic = sorted(
        ((t, len(v), sum(x >= THRESHOLDS[1] for x in v) / len(v)) for t, v in per_topic.items()),
        key=lambda r: (-r[1], r[0]),
    )[:SHOW]
    return Report(
        topics=len(refs),
        with_reference=sum(1 for r in refs.values() if r.found),
        scored=len(with_cos),
        unscored=len(scored) - len(with_cos),
        quantiles=(quantiles[0], quantiles[1], quantiles[2]),
        established={t: sum(1 for v in values if v >= t) for t in THRESHOLDS},
        least_novel=[row(n) for n in ordered[:SHOW]],
        most_novel=[row(n) for n in reversed(ordered[-SHOW:])],
        by_topic=by_topic,
    )


def render(report: Report) -> str:
    lines = [
        f"topics: {report.topics} with a reference article: {report.with_reference}",
        f"claims scored: {report.scored} (no reference for their topics: {report.unscored})",
        "max cosine to a reference sentence: p10 {:.3f} p50 {:.3f} p90 {:.3f}".format(
            *report.quantiles
        ),
        "established at cosine >= " + ", ".join(f"{t}: {n}" for t, n in report.established.items()),
        f"share established (>= {THRESHOLDS[1]}) on the largest topics:",
    ]
    lines += [f"  {topic}: {share:.1%} of {n}" for topic, n, share in report.by_topic]
    for label, rows in (("least novel", report.least_novel), ("most novel", report.most_novel)):
        lines.append(f"{label}:")
        for statement, title, value, nearest in rows:
            lines.append(f"  {value:.3f} {statement}")
            lines.append(f"        ~ {title}: {nearest}")
    return "\n".join(lines) + "\n"


def run_novelty(context: "AppContext", args: argparse.Namespace) -> int:
    import sys

    conn = context.conn
    known = run_ids(conn)
    for run in args.run:
        if run not in known:
            raise ConfigError(f"unknown run {run}")
    if args.min_claims < 1:
        raise ConfigError("--min-claims must be at least 1")
    claims, _ = load_claims(
        conn,
        tech_only=args.claim_gate,
        salt=context.settings.pseudonym_salt,
        runs=args.run,
        tag_runs=[args.tag_run],
        verdicts=parse_verdicts(args.verdicts),
    )
    by_topic = {t: v for t, v in _by_topic(claims).items() if len(v) >= args.min_claims}
    if args.fetch:
        n = fetch_references(conn, get, by_topic, context.clock.now(), sys.stderr, args.limit)
        context.stdout.write(f"fetched: {n} references\n")
        context.stdout.write(f"described: {fetch_descriptions(conn, get, sys.stderr)} references\n")
    refs = {t: r for t, r in references(conn).items() if t in by_topic}
    if not any(r.found for r in refs.values()):
        context.stdout.write("no reference texts; run with --fetch\n")
        return 0
    found = {t for t, r in refs.items() if r.found}
    scored = score(
        load_embedder(DEFAULT_MODEL, DEFAULT_REVISION),
        [c for c in claims if c.topics & found],
        refs,
        common_tokens(claims),
    )
    if args.write:
        store_novelty(conn, scored, context.clock.now())
        conn.commit()
        context.stdout.write(f"stored: {len(scored)} novelty rows\n")
    context.stdout.write(render(summarise(claims, scored, refs)))
    return 0
