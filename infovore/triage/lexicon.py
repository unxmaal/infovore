import hashlib
import json
import math
import re
import sqlite3
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Final

from infovore.rows import MessageRow
from infovore.triage.bayes import MAX_TOKEN_LENGTH, TOKEN, TRAILING_PUNCTUATION
from infovore.triage.human import Gazetteer, load_gazetteer

TECH_CHANNELS: Final = (
    "sgi-development",
    "software",
    "hardware",
    "sun-sparc",
    "dec-compaq-hp",
    "ibm-retirement-home",
    "network-installers",
    "unix",
    "emulation",
    "itanium",
    "networking-telecom-datacenter",
    "other-development",
)
OFF_CHANNELS: Final = (
    "motor-vehicles",
    "food",
    "music-geeks",
    "trains",
    "events-in-2026",
    "film-photography-visual-arts",
)
MIN_COUNT: Final = 30
MINE_LIMIT: Final = 400
_WORD: Final = re.compile(r"[a-z][a-z0-9+#-]{2,}")


class LexiconError(ValueError):
    pass


@dataclass(frozen=True)
class Lexicon:
    version: str
    terms: frozenset[str]
    gazetteer: Gazetteer
    sources: dict[str, int]

    @property
    def size(self) -> int:
        return sum(self.sources.values())


@dataclass(frozen=True)
class LexiconScore:
    share: float
    hits: int
    messages: int


@dataclass(frozen=True)
class MinedTerm:
    term: str
    tech: int
    off: int
    log_odds: float


def _terms(data: dict[str, object], section: str) -> list[str]:
    body = data.get(section)
    terms = body.get("terms") if isinstance(body, dict) else None
    if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
        raise LexiconError(f"lexicon: [{section}] needs a terms list of strings")
    return terms


def parse_lexicon(text: str, gazetteer: Gazetteer) -> Lexicon:
    data = tomllib.loads(text)
    general, mined = _terms(data, "general"), _terms(data, "mined")
    if not general:
        raise LexiconError("lexicon: [general] terms must not be empty")
    canonical = json.dumps([sorted(general), sorted(mined), gazetteer.version])
    sources = {
        "general": len(set(general)),
        "mined": len(set(mined) - set(general)),
        "gazetteer": len(gazetteer.text),
    }
    return Lexicon(
        version="lx-" + hashlib.sha256(canonical.encode()).hexdigest()[:12],
        terms=frozenset(general) | frozenset(mined),
        gazetteer=gazetteer,
        sources=sources,
    )


@lru_cache(maxsize=1)
def load_lexicon() -> Lexicon:
    path = resources.files("infovore.triage").joinpath("tech_lexicon.toml")
    return parse_lexicon(path.read_text(encoding="utf-8"), load_gazetteer())


def words(text: str) -> set[str]:
    found = {t.rstrip(TRAILING_PUNCTUATION) for t in TOKEN.findall(text.lower())}
    return {t for t in found if t and len(t) <= MAX_TOKEN_LENGTH}


def message_hits(lexicon: Lexicon, text: str) -> list[str]:
    tokens = words(text)
    hits = sorted(t for t in tokens if t in lexicon.terms)
    hits += sorted(
        t[:-1]
        for t in tokens
        if t.endswith("s") and t not in lexicon.terms and t[:-1] in lexicon.terms
    )
    hits = sorted(set(hits))
    hits += [
        f"gaz:{name}" for name, pattern in lexicon.gazetteer.text.items() if pattern.search(text)
    ]
    return hits


def score_lexicon(lexicon: Lexicon, messages: Sequence[MessageRow]) -> LexiconScore:
    per_message = [len(message_hits(lexicon, m.content)) for m in messages]
    with_hits = sum(1 for n in per_message if n)
    return LexiconScore(
        share=with_hits / len(per_message) if per_message else 0.0,
        hits=sum(per_message),
        messages=len(per_message),
    )


def _document_frequency(
    conn: sqlite3.Connection, channels: Sequence[str]
) -> tuple[Counter[str], int]:
    marks = ", ".join("?" for _ in channels)
    counts: Counter[str] = Counter()
    total = 0
    for row in conn.execute(
        f"SELECT m.content FROM messages m JOIN channels c ON c.id = m.channel_id"
        f" WHERE c.name IN ({marks})",
        tuple(channels),
    ):
        total += 1
        counts.update(set(_WORD.findall(row["content"].lower())))
    return counts, total


def mine_terms(
    conn: sqlite3.Connection,
    tech_channels: Sequence[str],
    off_channels: Sequence[str],
    min_count: int = MIN_COUNT,
    limit: int = MINE_LIMIT,
    lexicon: Lexicon | None = None,
) -> list[MinedTerm]:
    tech, tech_total = _document_frequency(conn, tech_channels)
    off, off_total = _document_frequency(conn, off_channels)
    known = lexicon.terms if lexicon is not None else frozenset()
    mined = [
        MinedTerm(
            term,
            count,
            off[term],
            math.log((count + 0.5) / (tech_total + 1))
            - math.log((off[term] + 0.5) / (off_total + 1)),
        )
        for term, count in tech.items()
        if count >= min_count and term not in known
    ]
    mined = [m for m in mined if m.log_odds > 0]
    mined.sort(key=lambda m: (-m.log_odds, m.term))
    return mined[:limit]
