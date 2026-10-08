import argparse
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from infovore.config import ConfigError
from infovore.db.batch import exchange_inputs_for_ids
from infovore.db.claims_v2 import run_ids
from infovore.rows import MessageRow
from infovore.triage.lexicon import Lexicon, load_lexicon, message_hits
from infovore.wiki.topics import Topics, load_topics

if TYPE_CHECKING:
    from infovore.cli import AppContext

DEFAULT_WORD_FLOOR: Final = 4
DEFAULT_DENSITIES: Final = "0,1,2,3,5,8"
DEFAULT_SHARES: Final = "0,0.2,0.34,0.5,0.67"
CHUNK: Final = 200
BAD: Final = ("wrong", "made_up")


@dataclass(frozen=True)
class ConversationGate:
    min_density: float = 0.0
    min_share: float = 0.0
    word_floor: int = DEFAULT_WORD_FLOOR

    @property
    def active(self) -> bool:
        return self.min_density > 0 or self.min_share > 0

    def passes(self, density: float, share: float) -> bool:
        return density >= self.min_density and share >= self.min_share


def conversation_stats(
    lexicon: Lexicon, messages: Sequence[MessageRow], word_floor: int
) -> tuple[float, float]:
    if not messages:
        return 0.0, 0.0
    total = hits = substantive = 0
    for message in messages:
        count = len(message.content.split())
        total += count
        substantive += count >= word_floor
        hits += len(message_hits(lexicon, message.content))
    return 100 * hits / total if total else 0.0, substantive / len(messages)


def claim_has_tech(lexicon: Lexicon, topics: Topics, statement: str) -> bool:
    return bool(message_hits(lexicon, statement)) or bool(topics.assign(statement))


def _stats_for(
    conn: sqlite3.Connection, ids: Sequence[int], lexicon: Lexicon, word_floor: int
) -> dict[int, tuple[float, float]]:
    found: dict[int, tuple[float, float]] = {}
    for start in range(0, len(ids), CHUNK):
        part = ids[start : start + CHUNK]
        for eid, inputs in exchange_inputs_for_ids(conn, part).items():
            found[eid] = conversation_stats(lexicon, inputs.messages, word_floor)
    return found


def gated_ids(
    conn: sqlite3.Connection, ids: Sequence[int], gate: ConversationGate, lexicon: Lexicon
) -> list[int]:
    if not gate.active:
        return list(ids)
    stats = _stats_for(conn, ids, lexicon, gate.word_floor)
    return [i for i in ids if gate.passes(*stats[i])]


@dataclass(frozen=True)
class Reduction:
    good_lost: int
    not_useful_removed: int
    bad_removed: int
    good_rate: tuple[int, int]


def reduction(before: Mapping[str, int], removed: Mapping[str, int]) -> Reduction:
    left = Counter(before)
    left.subtract(removed)
    return Reduction(
        removed.get("good", 0),
        removed.get("not_useful", 0),
        sum(removed.get(v, 0) for v in BAD),
        (left["good"], sum(left.values())),
    )


def _rate(good: int, total: int) -> str:
    from infovore.claims.value import wilson

    interval = wilson(good, total)
    if interval is None:
        return "good rate n/a"
    return (
        f"good rate {100 * good / total:.1f}%"
        f" ({100 * interval[0]:.1f}% to {100 * interval[1]:.1f}%)"
    )


def _row(label: str, dropped: str, r: Reduction) -> str:
    return (
        f"  {label}  dropped {dropped}  good lost {r.good_lost}"
        f"  not_useful removed {r.not_useful_removed}  wrong/made_up removed {r.bad_removed}"
        f"  {_rate(*r.good_rate)}"
    )


def _floats(text: str, name: str, low: float, high: float) -> list[float]:
    try:
        values = [float(p) for p in text.split(",")]
    except ValueError as error:
        raise ConfigError(f"--{name} must be comma-separated numbers") from error
    if any(not low <= v <= high for v in values):
        raise ConfigError(f"--{name} values must be between {low:g} and {high:g}")
    return values


def check_gate_args(density: float, share: float, word_floor: int) -> ConversationGate:
    if density < 0:
        raise ConfigError("--gate-density must not be negative")
    if not 0 <= share <= 1:
        raise ConfigError("--gate-share must be between 0 and 1")
    if word_floor < 1:
        raise ConfigError("--gate-word-floor must be positive")
    return ConversationGate(density, share, word_floor)


def run_gate_score(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode

    conn = context.conn
    try:
        runs = sorted({int(p) for p in args.runs.split(",")})
    except ValueError as error:
        raise ConfigError("--runs must be comma-separated run ids") from error
    if unknown := sorted(set(runs) - set(run_ids(conn))):
        raise ConfigError(f"unknown run {unknown[0]}")
    densities = _floats(args.densities, "densities", 0, 1000)
    shares = _floats(args.shares, "shares", 0, 1)
    if args.word_floor < 1:
        raise ConfigError("--word-floor must be positive")
    marks = ",".join("?" for _ in runs)
    exchanges = [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT exchange_id FROM claim_run_exchanges WHERE run_id IN ({marks})"
            " AND outcome != 'failed' ORDER BY exchange_id",
            runs,
        )
    ]
    reviews = conn.execute(
        "SELECT c.exchange_id, c.statement, r.verdict FROM claims_v2 c"
        " JOIN current_claim_reviews r ON r.claim_id = c.id"
        f" WHERE c.run_id IN ({marks})",
        runs,
    ).fetchall()
    lexicon, topics = load_lexicon(conn), load_topics()
    stats = _stats_for(conn, exchanges, lexicon, args.word_floor)
    before = Counter(r["verdict"] for r in reviews)
    mix = ", ".join(f"{k} {before[k]}" for k in sorted(before))
    out = [
        f"runs {','.join(map(str, runs))}: conversations {len(exchanges)},"
        f" reviewed claims {len(reviews)} ({mix or 'none'})",
        f"baseline {_rate(before['good'], len(reviews))}",
        f"conversation gate (word floor {args.word_floor}; density = lexicon hits per 100 words,"
        " share = messages at or above the floor):",
    ]
    for density in densities:
        for share in shares:
            gate = ConversationGate(density, share, args.word_floor)
            dropped = {e for e in exchanges if not gate.passes(*stats[e])}
            cut = Counter(r["verdict"] for r in reviews if r["exchange_id"] in dropped)
            pct = f"{100 * len(dropped) / len(exchanges):.1f}%" if exchanges else "n/a"
            label = f"density>={density:g} share>={share:g}"
            out.append(_row(label, f"{len(dropped)} ({pct})", reduction(before, cut)))
    techless = Counter(
        r["verdict"] for r in reviews if not claim_has_tech(lexicon, topics, r["statement"])
    )
    out.append("claim gate (statement has a lexicon or topic term):")
    dropped_claims = f"{sum(techless.values())} of {len(reviews)} reviewed"
    out.append(_row("no tech term", dropped_claims, reduction(before, techless)))
    context.stdout.write("\n".join(out) + "\n")
    return int(ExitCode.OK)
