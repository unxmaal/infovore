import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REMOVED: Final = "[link removed]"
_DISCORD = re.compile(r"https?://(?:[a-z0-9-]+\.)?discord(?:app)?\.(?:com|gg)/\S*", re.IGNORECASE)
_SNOWFLAKE = re.compile(r"(?<![\w.])\d{17,19}(?![\w.])")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PATTERNS: Final = (("discord-link", _DISCORD), ("snowflake", _SNOWFLAKE), ("email", _EMAIL))
SUFFIXES: Final = (".html", ".md", ".json")


@dataclass(frozen=True)
class Hit:
    file: str
    line: int
    kind: str
    text: str


def scrub(text: str) -> str:
    return _DISCORD.sub(REMOVED, text)


def scan_text(name: str, text: str) -> Iterator[Hit]:
    for number, line in enumerate(text.splitlines(), 1):
        for kind, pattern in _PATTERNS:
            for m in pattern.finditer(line):
                yield Hit(name, number, kind, m.group(0))


def scan(root: Path) -> list[Hit]:
    hits: list[Hit] = []
    for path in sorted(p for p in root.rglob("*") if p.suffix in SUFFIXES and p.is_file()):
        hits += scan_text(str(path.relative_to(root)), path.read_text(encoding="utf-8"))
    return hits
