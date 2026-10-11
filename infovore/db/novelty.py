import sqlite3
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

WIKI_PAGE: Final = "https://en.wikipedia.org/wiki/"


def wiki_url(title: str) -> str:
    return WIKI_PAGE + urllib.parse.quote(title.replace(" ", "_"))


@dataclass(frozen=True)
class Reference:
    topic: str
    title: str
    url: str
    text: str
    match: float
    description: str = ""

    @property
    def found(self) -> bool:
        return bool(self.text)


@dataclass(frozen=True)
class Novelty:
    claim_id: int
    reference_topic: str | None
    reference_title: str | None
    max_cosine: float | None
    nearest: str | None
    generic_share: float


def store_reference(conn: sqlite3.Connection, ref: Reference, now: datetime) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO reference_texts"
        " (topic, title, url, text, match, fetched_at, description) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ref.topic, ref.title, ref.url, ref.text, ref.match, now.isoformat(), ref.description),
    )


def references(conn: sqlite3.Connection) -> dict[str, Reference]:
    rows = conn.execute("SELECT topic, title, url, text, match, description FROM reference_texts")
    return {row[0]: Reference(*row) for row in rows}


def set_descriptions(conn: sqlite3.Connection, by_title: Mapping[str, str]) -> None:
    conn.executemany(
        "UPDATE reference_texts SET description = ? WHERE title = ?",
        [(description, title) for title, description in by_title.items()],
    )


def descriptions(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT topic, description FROM reference_texts WHERE description != ''")
    return dict(rows)


def store_novelty(conn: sqlite3.Connection, rows: Iterable[Novelty], now: datetime) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO claim_novelty (claim_id, reference_topic, reference_title,"
        " max_cosine, nearest, generic_share, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r.claim_id,
                r.reference_topic,
                r.reference_title,
                r.max_cosine,
                r.nearest,
                r.generic_share,
                now.isoformat(),
            )
            for r in rows
        ],
    )


def novelty_for(conn: sqlite3.Connection, claim_ids: Sequence[int]) -> dict[int, Novelty]:
    out: dict[int, Novelty] = {}
    ids = list(claim_ids)
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            "SELECT claim_id, reference_topic, reference_title, max_cosine, nearest,"
            f" generic_share FROM claim_novelty WHERE claim_id IN ({marks})",
            chunk,
        )
        for row in rows:
            out[row[0]] = Novelty(*row)
    return out
