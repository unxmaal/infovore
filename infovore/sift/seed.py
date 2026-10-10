import json
import math
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from infovore.db.channel_filter import exclude_channels_clause
from infovore.db.codec import to_db_time
from infovore.rows import LabelRegime
from infovore.sift.export import MANIFEST_NAME
from infovore.triage.lexicon import Lexicon, message_hits, tokens

SEED_FLOOR: Final = 20
SEED_TOP: Final = 50
SEED_REGIME: Final = LabelRegime.VALUE.value
SEED_SOURCE_REF_PREFIX: Final = "sift:seed-lexicon:"
SEED_DEEP_SOURCE_REF_PREFIX: Final = "sift:seed-lexicon-deep:"
REPORT_FLOORS: Final = (5, 10, 20, 40)
PERCENTILES: Final = (50, 75, 90, 95, 99)
BUCKET_EDGES: Final = (1, 3, 5, 10, 20, 40, 80, 160)
PREVIEW_CHARS: Final = 80


@dataclass(frozen=True)
class SeedRow:
    id: int
    channel: str
    words: int
    hits: int
    score: float
    text: str


def _scan(
    conn: sqlite3.Connection,
    lexicon: Lexicon,
    exclude_channels: frozenset[str],
    floor: int,
    counts: list[int] | None,
) -> list[SeedRow]:
    clause, params = exclude_channels_clause("m.channel_id", exclude_channels)
    length_clause = " AND length(m.content) >= ?" if counts is None else ""
    cursor = conn.execute(
        "SELECT m.id AS id, c.name AS channel, m.content AS content"
        " FROM messages m JOIN channels c ON c.id = m.channel_id"
        " WHERE m.deleted_at IS NULL AND m.author_is_bot = 0"
        " AND m.author_id NOT IN (SELECT user_id FROM opt_outs)"
        " AND EXISTS (SELECT 1 FROM exchange_messages e WHERE e.message_id = m.id)"
        " AND NOT EXISTS (SELECT 1 FROM message_labels ml"
        " WHERE ml.message_id = m.id AND ml.source = 'human')"
        f"{clause}{length_clause}",
        (*params, *([floor] if counts is None else [])),
    )
    rows: list[SeedRow] = []
    for row in cursor:
        words = len(tokens(row["content"]))
        if counts is not None:
            counts.append(words)
        if words < floor:
            continue
        hits = len(message_hits(lexicon, row["content"]))
        rows.append(
            SeedRow(row["id"], row["channel"], words, hits, hits * 100 / words, row["content"])
        )
    rows.sort(key=lambda r: (-r.score, -r.hits, r.id))
    return rows


def seed_rows(
    conn: sqlite3.Connection, lexicon: Lexicon, exclude_channels: frozenset[str], floor: int
) -> list[SeedRow]:
    return _scan(conn, lexicon, exclude_channels, floor, None)


def top_rows(rows: list[SeedRow], k: int) -> list[SeedRow]:
    return rows[:k]


def parse_ranks(text: str) -> tuple[int, int]:
    parts = text.split("-")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"expected A-B, got {text!r}")
    start, end = int(parts[0]), int(parts[1])
    if start < 1 or end < start:
        raise ValueError(f"expected 1 <= A <= B, got {text!r}")
    return start, end


def sample_window(rows: list[SeedRow], start: int, end: int, n: int, seed: int) -> list[SeedRow]:
    window = rows[start - 1 : end]
    picked = random.Random(seed).sample(range(len(window)), min(n, len(window)))
    return [window[i] for i in sorted(picked)]


def percentile(sorted_values: list[int], pct: int) -> int:
    if not sorted_values:
        return 0
    return sorted_values[max(0, math.ceil(pct / 100 * len(sorted_values)) - 1)]


def render_report(
    conn: sqlite3.Connection,
    lexicon: Lexicon,
    exclude_channels: frozenset[str],
    floor: int,
    top: int,
) -> str:
    counts: list[int] = []
    rows = _scan(conn, lexicon, exclude_channels, min(REPORT_FLOORS), counts)
    counts.sort()
    lines = [f"candidates: {len(counts)}", "word count percentiles:"]
    lines += [f"  p{pct}: {percentile(counts, pct)}" for pct in PERCENTILES]
    lines.append("word count buckets:")
    lines += [f"  <= {edge}: {sum(1 for c in counts if c <= edge)}" for edge in BUCKET_EDGES]
    lines.append(f"  > {BUCKET_EDGES[-1]}: {sum(1 for c in counts if c > BUCKET_EDGES[-1])}")
    lines.append(f"floor remaining top-{top} mean words")
    for candidate in REPORT_FLOORS:
        kept = [r for r in rows if r.words >= candidate]
        best = kept[:top]
        mean = sum(r.words for r in best) / len(best) if best else 0.0
        lines.append(f"{candidate:>5} {len(kept):>9} {mean:>10.1f}")
    lines.append(f"floor used: {floor}")
    return "\n".join(lines) + "\n"


def write_seed_queue(
    rows: list[SeedRow], out_dir: Path, now: datetime, prefix: str = SEED_SOURCE_REF_PREFIX
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_ref = prefix + now.date().isoformat()
    manifest = {
        "message_ids": [r.id for r in rows],
        "strategy": "seed",
        "size": len(rows),
        "created_at": to_db_time(now),
        "source_ref": source_ref,
        "regime": SEED_REGIME,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return source_ref


def format_rows(rows: list[SeedRow]) -> str:
    lines = ["id\tchannel\twords\thits\tscore\ttext"]
    lines += [
        f"{r.id}\t{r.channel}\t{r.words}\t{r.hits}\t{r.score:.1f}\t"
        f"{' '.join(r.text.split())[:PREVIEW_CHARS]}"
        for r in rows
    ]
    return "\n".join(lines) + "\n"
