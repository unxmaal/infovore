import argparse
import random
import sqlite3
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from infovore.claims.redact import pseudonyms, require_salt
from infovore.config import ConfigError
from infovore.db.claims_v2 import run_ids
from infovore.db.speaker_drops import decisions, dropped_authors

if TYPE_CHECKING:
    from infovore.cli import AppContext

Pairs = frozenset[tuple[int, str]]


@dataclass(frozen=True)
class ClaimLine:
    claim_id: int
    statement: str
    verdict: str | None


@dataclass(frozen=True)
class SpeakerStats:
    author_id: int
    label: str
    claims: tuple[ClaimLine, ...]
    reviewed: int
    good: int


@dataclass(frozen=True)
class SpeakerGroup:
    author_id: int
    label: str
    total: int
    decision: str | None
    sample: tuple[ClaimLine, ...]


def _resolve(
    conn: sqlite3.Connection, salt: str, scope: str, params: Sequence[int]
) -> dict[tuple[int, str], int]:
    by_exchange: dict[int, set[int]] = {}
    for row in conn.execute(
        "SELECT DISTINCT em.exchange_id, m.author_id FROM all_exchange_messages em"
        f" JOIN messages m ON m.id = em.message_id WHERE em.exchange_id IN ({scope})",
        params,
    ):
        by_exchange.setdefault(row[0], set()).add(row[1])
    return {
        (exchange, name): author
        for exchange, authors in by_exchange.items()
        for author, name in pseudonyms(authors, salt).items()
    }


def _marks(values: Sequence[int]) -> str:
    return ",".join("?" for _ in values)


def dropped_pairs(conn: sqlite3.Connection, salt: str | None) -> Pairs:
    dropped = sorted(dropped_authors(conn))
    if not dropped:
        return frozenset()
    scope = (
        "SELECT x.exchange_id FROM all_exchange_messages x JOIN messages n ON n.id = x.message_id"
        f" WHERE n.author_id IN ({_marks(dropped)})"
    )
    resolved = _resolve(conn, require_salt(salt), scope, dropped)
    return frozenset(pair for pair, author in resolved.items() if author in dropped)


def exclude_dropped(conn: sqlite3.Connection, pairs: Pairs) -> None:
    conn.create_function(
        "is_dropped", 2, lambda exchange, speaker: (exchange, speaker) in pairs, deterministic=True
    )


def kept_exchange_ids(
    conn: sqlite3.Connection, ids: Sequence[int], dropped: frozenset[int]
) -> list[int]:
    if not dropped:
        return list(ids)
    gone, kept = sorted(dropped), set[int]()
    for start in range(0, len(ids), 400):
        chunk = list(ids[start : start + 400])
        kept.update(
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT x.exchange_id FROM all_exchange_messages x"
                " JOIN messages m ON m.id = x.message_id"
                f" WHERE x.exchange_id IN ({_marks(chunk)})"
                f" AND m.author_id NOT IN ({_marks(gone)})",
                [*chunk, *gone],
            )
        )
    return [i for i in ids if i in kept]


def speaker_stats(
    conn: sqlite3.Connection,
    salt: str,
    runs: Sequence[int],
    exclude: frozenset[int] = frozenset(),
) -> list[SpeakerStats]:
    marks = _marks(runs)
    authors = _resolve(
        conn,
        salt,
        f"SELECT exchange_id FROM claims_v2 WHERE run_id IN ({marks})",
        runs,
    )
    lines: dict[int, list[ClaimLine]] = {}
    names: dict[int, Counter[str]] = {}
    reviewed: Counter[int] = Counter()
    good: Counter[int] = Counter()
    for row in conn.execute(
        "SELECT c.id, c.exchange_id, c.speaker, c.statement, r.verdict, r.interface"
        " FROM claims_v2 c LEFT JOIN current_claim_reviews r ON r.claim_id = c.id"
        f" WHERE c.run_id IN ({marks}) ORDER BY c.id",
        list(runs),
    ):
        author = authors.get((row["exchange_id"], row["speaker"]))
        if author is None or author in exclude:
            continue
        lines.setdefault(author, []).append(ClaimLine(row["id"], row["statement"], row["verdict"]))
        names.setdefault(author, Counter())[row["speaker"]] += 1
        if row["interface"] == "conversation":
            reviewed[author] += 1
            good[author] += row["verdict"] == "good"
    ranked = sorted(lines, key=lambda a: (-len(lines[a]), a))
    return [
        SpeakerStats(
            a,
            min(names[a], key=lambda n: (-names[a][n], n)),
            tuple(lines[a]),
            reviewed[a],
            good[a],
        )
        for a in ranked
    ]


def speaker_groups(
    conn: sqlite3.Connection, salt: str, run_id: int, top: int, per: int, seed: int
) -> list[SpeakerGroup]:
    decided = decisions(conn)
    groups = []
    for stats in speaker_stats(conn, salt, [run_id])[:top]:
        by_id = {c.claim_id: c for c in stats.claims}
        picked = random.Random(seed).sample(sorted(by_id), min(per, len(by_id)))
        sample = tuple(by_id[i] for i in sorted(picked))
        groups.append(
            SpeakerGroup(
                stats.author_id,
                stats.label,
                len(stats.claims),
                decided.get(stats.author_id),
                sample,
            )
        )
    return groups


def run_speakers(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.claims.httpd import listening_url, shutdown_all
    from infovore.claims.speakers_httpd import start_all
    from infovore.cli import ExitCode
    from infovore.sift.httpd import block_until_interrupted

    conn = context.conn
    if args.run not in run_ids(conn):
        raise ConfigError(f"unknown run {args.run}")
    salt = require_salt(context.settings.pseudonym_salt)
    if args.top < 1:
        raise ConfigError("--top must be positive")
    if args.per < 1:
        raise ConfigError("--per must be positive")
    everyone = speaker_stats(conn, salt, [args.run])
    groups = speaker_groups(conn, salt, args.run, args.top, args.per, args.seed)
    dropped = sum(1 for g in groups if g.decision == "drop")
    context.stdout.write(f"{len(groups)} of {len(everyone)} speakers, {dropped} dropped\n")
    servers = start_all(
        args.hosts or ["127.0.0.1"], args.port, conn, context.clock, args.run, groups
    )
    try:
        for server in servers:
            context.stdout.write(f"listening on {listening_url(server)}\n")
        context.stdout.flush()
        block_until_interrupted(threading.Event())
    finally:
        shutdown_all(servers)
    return int(ExitCode.OK)
