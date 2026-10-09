import re
import tomllib
from collections import Counter
from collections.abc import Iterable
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
_STRIP = re.compile(r"[\s\-_/]+")
_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")


def canonical(tag: str) -> str:
    text = tag.casefold().strip()
    for vendor in VENDORS:
        rest = text[len(vendor) :].strip()
        if text.startswith(vendor + " ") and rest:
            text = rest
            break
    return _DOT.sub("", _STRIP.sub("", text))


def display_names(tags: Iterable[str]) -> dict[str, str]:
    spellings: dict[str, Counter[str]] = {}
    for tag in tags:
        key = canonical(tag)
        if key:
            spellings.setdefault(key, Counter())[tag] += 1
    return {
        key: min(counts, key=lambda spelling: (-counts[spelling], spelling))
        for key, counts in spellings.items()
    }


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
