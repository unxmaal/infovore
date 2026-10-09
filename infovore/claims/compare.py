import argparse
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from infovore.claims.value import tokens, wilson
from infovore.config import ConfigError
from infovore.db.claims_v2 import run_ids

if TYPE_CHECKING:
    from infovore.cli import AppContext

DEFAULT_THRESHOLD: Final = 0.5
VERDICTS: Final = ("good", "not_useful", "wrong", "made_up")


def best_match(statement: str, candidates: Sequence[frozenset[str]]) -> float:
    toks = tokens(statement)
    best = 0.0
    for other in candidates:
        union = toks | other
        if union:
            best = max(best, len(toks & other) / len(union))
    return best


@dataclass(frozen=True)
class Comparison:
    reviewed: dict[str, int]
    reproduced: dict[str, int]
    new_claims: int
    unmatched_new: int
    exchanges: int


def compare(
    conn: sqlite3.Connection, base_runs: Sequence[int], run: int, threshold: float
) -> Comparison:
    exchanges = {
        r[0]
        for r in conn.execute(
            "SELECT exchange_id FROM claim_run_exchanges WHERE run_id = ? AND outcome != 'failed'",
            (run,),
        )
    }
    new: dict[int, list[frozenset[str]]] = {}
    for eid, statement in conn.execute(
        "SELECT c.exchange_id, c.statement FROM claims_v2 c JOIN claim_run_exchanges x"
        " ON x.run_id = c.run_id AND x.exchange_id = c.exchange_id"
        " WHERE c.run_id = ? AND x.outcome != 'failed'",
        (run,),
    ):
        new.setdefault(eid, []).append(tokens(statement))
    marks = ",".join("?" for _ in base_runs)
    old: dict[int, list[frozenset[str]]] = {}
    reviewed: Counter[str] = Counter()
    reproduced: Counter[str] = Counter()
    for eid, statement, verdict in conn.execute(
        "SELECT c.exchange_id, c.statement, r.verdict FROM claims_v2 c"
        f" JOIN current_claim_reviews r ON r.claim_id = c.id WHERE c.run_id IN ({marks})",
        list(base_runs),
    ):
        if eid not in exchanges:
            continue
        old.setdefault(eid, []).append(tokens(statement))
        reviewed[verdict] += 1
        if best_match(statement, new.get(eid, [])) >= threshold:
            reproduced[verdict] += 1
    unmatched = sum(
        1
        for eid, sets in new.items()
        for toks in sets
        if best_match(" ".join(sorted(toks)), old.get(eid, [])) < threshold
    )
    return Comparison(
        dict(reviewed),
        dict(reproduced),
        sum(len(v) for v in new.values()),
        unmatched,
        len(exchanges),
    )


def format_comparison(c: Comparison) -> list[str]:
    lines = [
        f"conversations {c.exchanges}, new claims {c.new_claims}, unmatched new {c.unmatched_new}"
    ]
    for verdict in VERDICTS:
        got, of = c.reproduced.get(verdict, 0), c.reviewed.get(verdict, 0)
        lines.append(f"  {verdict}: reproduced {got} of {of}")
    good, total = c.reproduced.get("good", 0), sum(c.reproduced.values())
    interval = wilson(good, total)
    if interval is None:
        lines.append("reproduced good rate n/a")
    else:
        lines.append(
            f"reproduced good rate {100 * good / total:.1f}%"
            f" ({100 * interval[0]:.1f}% to {100 * interval[1]:.1f}%)"
        )
    return lines


def run_compare(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode

    try:
        base = sorted({int(p) for p in args.base_runs.split(",")})
    except ValueError as error:
        raise ConfigError("--base-runs must be comma-separated run ids") from error
    known = set(run_ids(context.conn))
    for rid in [*base, args.run]:
        if rid not in known:
            raise ConfigError(f"unknown run {rid}")
    if not 0 < args.threshold <= 1:
        raise ConfigError("--threshold must be in (0, 1]")
    result = compare(context.conn, base, args.run, args.threshold)
    context.stdout.write("\n".join(format_comparison(result)) + "\n")
    return int(ExitCode.OK)
