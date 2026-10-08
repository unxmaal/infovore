import argparse
import functools
import hashlib
import math
import random
import re
import sqlite3
import struct
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from infovore.config import ConfigError
from infovore.db.claims_v2 import run_ids
from infovore.wiki.build import subjects
from infovore.wiki.topics import load_topics

if TYPE_CHECKING:
    from infovore.cli import AppContext

_WORDS: Final = (
    "a an the and or but if so of to in on at by for with from as is are was were be been "
    "it its this that these those i we you he she they them their his her what who which "
    "about up out get has have had"
)
STOPWORDS: Final = frozenset(_WORDS.split())
PERMS: Final = 64
BANDS: Final = 16
ROWS: Final = 4
THRESHOLD: Final = 0.6
WINDOW: Final = 1000
Z95: Final = 1.96
_WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in STOPWORDS)


@functools.lru_cache(maxsize=50_000)
def _hashes(token: str) -> tuple[int, ...]:
    raw = b"".join(
        hashlib.blake2b(token.encode(), digest_size=64, salt=bytes([k])).digest() for k in range(4)
    )
    return struct.unpack(f"<{PERMS}I", raw)


def _signature(toks: frozenset[str]) -> tuple[int, ...]:
    return tuple(map(min, zip(*(_hashes(t) for t in toks), strict=True)))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b)


class NearDuplicateIndex:
    def __init__(self) -> None:
        self._sets: list[frozenset[str]] = []
        self._buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}

    @staticmethod
    def _keys(toks: frozenset[str]) -> list[tuple[int, tuple[int, ...]]]:
        sig = _signature(toks)
        return [(b, sig[b * ROWS : (b + 1) * ROWS]) for b in range(BANDS)]

    def seen(self, toks: frozenset[str]) -> bool:
        if not toks:
            return False
        for key in self._keys(toks):
            for i in self._buckets.get(key, ()):
                if _jaccard(self._sets[i], toks) >= THRESHOLD:
                    return True
        return False

    def add(self, toks: frozenset[str]) -> None:
        if not toks:
            return
        self._sets.append(toks)
        for key in self._keys(toks):
            self._buckets.setdefault(key, []).append(len(self._sets) - 1)


def wilson(good: int, total: int) -> tuple[float, float] | None:
    if not total:
        return None
    p, z2 = good / total, Z95 * Z95
    denom = 1 + z2 / total
    centre = (p + z2 / (2 * total)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denom
    return centre - half, centre + half


def sample_ids(ids: Sequence[int], n: int, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(sorted(ids), min(n, len(ids))))


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def _rate(count: int, conversations: int) -> str:
    return f"{1000 * count / conversations:.1f}" if conversations else "n/a"


_SCAN = (
    "SELECT e.run_id, e.rowid AS pos, e.outcome, ch.name AS channel, c.statement"
    " FROM claim_run_exchanges e"
    " LEFT JOIN current_exchanges x ON x.id = e.exchange_id"
    " LEFT JOIN channels ch ON ch.id = x.channel_id"
    " LEFT JOIN claims_v2 c ON c.run_id = e.run_id AND c.exchange_id = e.exchange_id"
    " WHERE e.run_id <= ? ORDER BY e.run_id, e.rowid, c.id"
)


def _scan(conn: sqlite3.Connection, runs: frozenset[int], min_claims: int) -> list[str]:
    topics = load_topics()
    index = NearDuplicateIndex()
    counts: dict[str, int] = {}
    per_conv: list[list[int]] = []
    channels: dict[str, list[int]] = {}
    failed = claims = 0
    current = -1
    row: list[int] = []
    stats: list[int] = []
    for r in conn.execute(_SCAN, (max(runs),)):
        chosen = r["run_id"] in runs
        if chosen and r["pos"] != current:
            current = r["pos"]
            if r["outcome"] == "failed":
                failed += 1
                row, stats = [], []
            else:
                row = [0, 0, 0, 0]
                per_conv.append(row)
                stats = channels.setdefault(r["channel"], [0, 0, 0])
                stats[0] += 1
        if r["statement"] is None:
            continue
        toks = tokens(r["statement"])
        dup = chosen and index.seen(toks)
        index.add(toks)
        if not chosen:
            for name in subjects(topics, r["statement"]):
                counts[name] = counts.get(name, 0) + 1
            continue
        claims += 1
        row[0] += 1
        stats[1] += 1
        row[1] += dup
        for name in subjects(topics, r["statement"]):
            counts[name] = counts.get(name, 0) + 1
            if counts[name] == 1:
                row[2] += 1
                stats[2] += 1
            if counts[name] == min_claims:
                row[3] += 1
    return _lines(per_conv, channels, failed, claims, min_claims)


def _lines(
    per_conv: list[list[int]],
    channels: dict[str, list[int]],
    failed: int,
    claims: int,
    min_claims: int,
) -> list[str]:
    n = len(per_conv)
    per = f"{claims / n:.2f}" if n else "n/a"
    out = [f"conversations {n} (failed {failed}), claims {claims} ({per} per conversation)"]
    out.append(f"novelty (near-duplicate of an earlier claim, Jaccard >= {THRESHOLD}):")
    for label, part in (("first half", per_conv[: n // 2]), ("second half", per_conv[n // 2 :])):
        made, dups = sum(r[0] for r in part), sum(r[1] for r in part)
        out.append(
            f"  {label}: {dups} of {made} claims ({_pct(dups, made)})"
            f" over {len(part)} conversations"
        )
    new, reached = sum(r[2] for r in per_conv), sum(r[3] for r in per_conv)
    out.append("wiki growth (same subjects as wiki build, claims not filtered by check):")
    out.append(f"  new subjects: {new} ({_rate(new, n)} per 1000 conversations)")
    out.append(
        f"  reaching {min_claims} claims: {reached} ({_rate(reached, n)} per 1000 conversations)"
    )
    for start in range(0, n, WINDOW):
        part = per_conv[start : start + WINDOW]
        out.append(
            f"  conversations {start + 1}-{start + len(part)}:"
            f" new subjects {sum(r[2] for r in part)}, reaching {sum(r[3] for r in part)}"
        )
    out.append("per channel:")
    for name, (convs, made, fresh) in sorted(channels.items()):
        out.append(
            f"  {name}: conversations {convs}, claims per conversation {made / convs:.2f},"
            f" new subjects {fresh}"
        )
    return out


def _in(runs: frozenset[int]) -> tuple[str, list[int]]:
    return ",".join("?" for _ in runs), sorted(runs)


def _tail(conn: sqlite3.Connection, runs: frozenset[int]) -> list[str]:
    marks, params = _in(runs)
    total = conn.execute(
        f"SELECT COUNT(*) FROM claims_v2 WHERE run_id IN ({marks})", params
    ).fetchone()[0]
    checks = dict(
        conn.execute(
            "SELECT k.verdict, COUNT(*) FROM claims_v2 c JOIN current_claim_checks k"
            f" ON k.claim_id = c.id WHERE c.run_id IN ({marks}) GROUP BY k.verdict ORDER BY 1",
            params,
        ).fetchall()
    )
    mix = ", ".join(f"{k} {v}" for k, v in checks.items())
    if checks:
        mix += f", unchecked {total - sum(checks.values())}"
    reviews = dict(
        conn.execute(
            "SELECT r.verdict, COUNT(*) FROM claims_v2 c JOIN current_claim_reviews r"
            f" ON r.claim_id = c.id WHERE c.run_id IN ({marks}) AND r.interface = 'conversation'"
            " GROUP BY r.verdict",
            params,
        ).fetchall()
    )
    good, reviewed = reviews.get("good", 0), sum(reviews.values())
    interval = wilson(good, reviewed)
    if interval:
        low, high = interval
        sample = (
            f"reviewed sample (interface=conversation): good {good} of {reviewed}"
            f" ({_pct(good, reviewed)}), Wilson 95% {100 * low:.1f}% to {100 * high:.1f}%"
        )
    else:
        sample = "reviewed sample: none"
    return [f"claim checks: {mix or 'none'}", sample]


def run_value(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode

    conn = context.conn
    try:
        runs = frozenset(int(p) for p in args.runs.split(","))
    except ValueError as error:
        raise ConfigError("--runs must be comma-separated run ids") from error
    if unknown := sorted(runs - set(run_ids(conn))):
        raise ConfigError(f"unknown run {unknown[0]}")
    if args.min_claims < 1:
        raise ConfigError("--min-claims must be positive")
    marks, params = _in(runs)
    rejected = conn.execute(
        f"SELECT COUNT(*) FROM claim_rejections WHERE run_id IN ({marks})", params
    ).fetchone()[0]
    lines = _scan(conn, runs, args.min_claims)
    lines[0] = f"runs {','.join(map(str, sorted(runs)))}: {lines[0]}, rejected {rejected}"
    context.stdout.write("\n".join([*lines, *_tail(conn, runs)]) + "\n")
    return int(ExitCode.OK)
