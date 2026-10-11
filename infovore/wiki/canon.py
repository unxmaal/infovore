import re
import tomllib
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from infovore.wiki.topics import Topics

VENDORS: Final = (
    "silicon graphics",
    "sun microsystems",
    "sgi",
    "sun",
    "ibm",
    "dec",
    "digital",
    "hp",
    "apple",
    "compaq",
)
GENERIC: Final = ("machine", "system", "computer", "workstation")
MIN_SHARE: Final = 0.25
_STRIP = re.compile(r"[\s\-_/]+|[\u2013\u2014]+")
_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
_SYMBOLS = re.compile(r"[\u00ae\u2122\u00a9*\"\'`\u2018\u2019\u201c\u201d]|\(\)")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_GENERIC = re.compile(r"^(?:the\s+)?(.+?)\s+(?:" + "|".join(GENERIC) + r")s?$", re.IGNORECASE)
_THE = re.compile(r"^the\s+(.+)$", re.IGNORECASE)

Spellings = Mapping[str, Counter[str]]


def canonical(tag: str) -> str:
    text = _SYMBOLS.sub("", unicodedata.normalize("NFKC", tag)).casefold().strip().lstrip("$")
    for vendor in VENDORS:
        rest = text[len(vendor) :].strip()
        if text.startswith(vendor + " ") and rest:
            text = rest
            break
    return _DOT.sub("", _STRIP.sub("", text))


def spelling_counts(tags: Iterable[str]) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = {}
    for tag in tags:
        key = canonical(tag)
        if key:
            out.setdefault(key, Counter())[tag] += 1
    return out


def _singulars(spelling: str) -> list[str]:
    word = spelling.strip()
    if len(word) < 3 or word[-1] != "s" or word[-2] in " s":
        return []
    out = [word[:-1]]
    if word.endswith("ies"):
        out.append(word[:-3] + "y")
    elif word.endswith("es"):
        out.append(word[:-2])
    return out


def _variants(key: str, spelling: str) -> list[str]:
    out = [_NON_ASCII.sub("", key)]
    if m := _GENERIC.match(spelling):
        out.append(m.group(1))
    if m := _THE.match(spelling):
        out.append(m.group(1))
    out += _singulars(spelling)
    return out


def merge_map(spellings: Spellings) -> dict[str, str]:
    counts = {key: sum(forms.values()) for key, forms in spellings.items()}
    step: dict[str, str] = {}
    for key, forms in spellings.items():
        spelling = min(forms, key=lambda s: (-forms[s], s))
        for variant in _variants(key, spelling):
            target = canonical(variant)
            if target and target != key and counts.get(target, 0) >= MIN_SHARE * counts[key]:
                step[key] = target
                break
    out: dict[str, str] = {}
    for key in step:
        seen, cursor = {key}, key
        while cursor in step and step[cursor] not in seen:
            cursor = step[cursor]
            seen.add(cursor)
        out[key] = cursor
    return out


def display_names(tags: Iterable[str], merge: Mapping[str, str] | None = None) -> dict[str, str]:
    spellings = spelling_counts(tags)
    merged: dict[str, Counter[str]] = {}
    for key, forms in spellings.items():
        target = (merge or {}).get(key, key)
        if target not in spellings or target == key:
            merged.setdefault(target, Counter()).update(forms)
    return {
        key: min(counts, key=lambda spelling: (-counts[spelling], spelling))
        for key, counts in merged.items()
    }


@dataclass(frozen=True)
class Merged:
    keys: int
    tags: int

    def __str__(self) -> str:
        return f"{self.keys} keys, {self.tags} claim-tags"


def summarise_merge(spellings: Spellings, merge: Mapping[str, str]) -> Merged:
    return Merged(len(merge), sum(sum(spellings[key].values()) for key in merge))


def alias_names(topics: Topics) -> dict[str, str]:
    data = tomllib.loads(Path(__file__).with_name("topics.toml").read_text())
    known = set(topics.names)
    return {
        canonical(alias): entry["name"]
        for entry in data["topic"]
        if entry["name"] in known
        for alias in entry.get("aliases", [])
        if canonical(alias)
    }
