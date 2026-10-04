import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

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


def load_topics(text: str | None = None) -> Topics:
    source = text if text is not None else Path(__file__).with_name("topics.toml").read_text()
    entries: list[tuple[str, re.Pattern[str]]] = []
    seen: set[str] = set()
    for topic in tomllib.loads(source)["topic"]:
        name = topic["name"]
        if name in seen:
            raise ValueError(f"duplicate topic: {name}")
        seen.add(name)
        parts = [_EDGE.format(re.escape(a)) for a in topic.get("aliases", [])]
        parts += topic.get("patterns", [])
        entries.append((name, re.compile("|".join(parts), re.IGNORECASE)))
    return Topics(tuple(entries))
