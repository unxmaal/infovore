import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infovore.wiki.kinds import KINDS

_EDGE = r"(?<![A-Za-z0-9]){}(?![A-Za-z0-9])"


@dataclass(frozen=True)
class Topics:
    entries: tuple[tuple[str, re.Pattern[str]], ...]

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.entries]

    def assign(self, text: str) -> frozenset[str]:
        return frozenset(name for name, pattern in self.entries if pattern.search(text))


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _entries(text: str | None) -> list[dict[str, Any]]:
    source = text if text is not None else Path(__file__).with_name("topics.toml").read_text()
    seen: set[str] = set()
    entries: list[dict[str, Any]] = tomllib.loads(source)["topic"]
    for topic in entries:
        if topic["name"] in seen:
            raise ValueError(f"duplicate topic: {topic['name']}")
        seen.add(topic["name"])
    return entries


def load_topics(text: str | None = None) -> Topics:
    entries: list[tuple[str, re.Pattern[str]]] = []
    for topic in _entries(text):
        parts = [_EDGE.format(re.escape(a)) for a in topic.get("aliases", [])]
        parts += topic.get("patterns", [])
        if parts:
            entries.append((topic["name"], re.compile("|".join(parts), re.IGNORECASE)))
    return Topics(tuple(entries))


def topic_kinds(text: str | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for topic in _entries(text):
        if "kind" in topic:
            if topic["kind"] not in KINDS:
                raise ValueError(f"unknown kind for {topic['name']}: {topic['kind']}")
            out[topic["name"]] = topic["kind"]
    return out
