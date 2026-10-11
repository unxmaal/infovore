import argparse
import json
import random
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from infovore.claims.check import VERDICTS, check_claim, overlap
from infovore.db.claims_v2 import ReviewRow
from infovore.triage.lexicon import Lexicon, message_hits
from infovore.wiki.support import FLOOR

if TYPE_CHECKING:
    from infovore.cli import AppContext

SEVERITY: Final = ("unsupported_fact", "low_overlap", "uncheckable", "unchecked", "supported")
LENGTH_BUCKETS: Final = ((0, 29), (30, 79), (80, 199), (200, 10**9))
QUANTILES: Final = (0.5, 0.9, 0.99)


@dataclass(frozen=True)
class Tally:
    n: int
    verdicts: dict[str, int]
    mean_overlap: float

    def rate(self, verdict: str) -> float:
        return self.verdicts.get(verdict, 0) / self.n if self.n else 0.0


def _tally(results: Sequence[tuple[str, float]]) -> Tally:
    counts = Counter(v for v, _ in results)
    mean = sum(o for _, o in results) / len(results) if results else 0.0
    return Tally(len(results), {v: counts.get(v, 0) for v in VERDICTS}, mean)


def _texts(row: ReviewRow) -> list[str]:
    return [content for _, _, content in row.sources]


def _partner(rows: Sequence[ReviewRow], index: int, rng: random.Random) -> ReviewRow:
    candidates = [i for i in range(len(rows)) if i != index]
    return rows[rng.choice(candidates)]


def shuffled_citations(
    rows: Sequence[ReviewRow], lexicon: Lexicon, threshold: float, rng: random.Random
) -> tuple[Tally, Tally, float]:
    by_exchange: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_exchange[row.exchange_id].append(i)
    real = []
    shuffled = []
    within = 0
    for i, row in enumerate(rows):
        checked = check_claim(row.statement, _texts(row), lexicon, threshold)
        real.append((checked.verdict, checked.overlap))
        siblings = [j for j in by_exchange[row.exchange_id] if j != i]
        if siblings:
            partner = rows[rng.choice(siblings)]
            within += 1
        else:
            partner = _partner(rows, i, rng)
        other = check_claim(row.statement, _texts(partner), lexicon, threshold)
        shuffled.append((other.verdict, other.overlap))
    share = within / len(rows) if rows else 0.0
    return _tally(real), _tally(shuffled), share


@dataclass(frozen=True)
class Floor:
    n: int
    quantiles: dict[str, float]
    above_floor: float
    above_threshold: float


def overlap_floor(
    rows: Sequence[ReviewRow],
    lexicon: Lexicon,
    threshold: float,
    rng: random.Random,
    floor: float = FLOOR,
) -> Floor:
    scores = sorted(
        overlap(row.statement, _texts(_partner(rows, i, rng)), lexicon)
        for i, row in enumerate(rows)
    )
    if not scores:
        return Floor(0, {f"p{int(q * 100)}": 0.0 for q in QUANTILES}, 0.0, 0.0)
    quantiles = {
        f"p{int(q * 100)}": scores[min(len(scores) - 1, int(q * len(scores)))] for q in QUANTILES
    }
    return Floor(
        len(scores),
        quantiles,
        sum(1 for s in scores if s >= floor) / len(scores),
        sum(1 for s in scores if s >= threshold) / len(scores),
    )


def worst(verdicts: Iterable[str | None]) -> str:
    present = {v or "unchecked" for v in verdicts}
    return next((v for v in SEVERITY if v in present), "unchecked")


def read_drops(path: Path) -> list[list[str]]:
    out = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if "reason" in row:
                out.append(list(row["cited"]))
    return out


def drop_rate_by_verdict(
    sections: Mapping[str, Mapping[str, Sequence[tuple[str, Sequence[int]]]]],
    drops: Sequence[Sequence[str]],
    verdict_of: Mapping[int, str | None],
    claim_of_statement: Mapping[str, int],
) -> dict[str, tuple[int, int]]:
    kept: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    for topic in sections.values():
        for sentences in topic.values():
            for _, ids in sentences:
                kept[worst(verdict_of.get(i) for i in ids)] += 1
    for cited in drops:
        ids = [claim_of_statement[s] for s in cited if s in claim_of_statement]
        dropped[worst(verdict_of.get(i) for i in ids)] += 1
    return {v: (kept[v], dropped[v]) for v in SEVERITY if kept[v] or dropped[v]}


def _length_bucket(length: int) -> str:
    for low, high in LENGTH_BUCKETS[:-1]:
        if length <= high:
            return f"{low}-{high}"
    return f"{LENGTH_BUCKETS[-1][0]}+"


@dataclass(frozen=True)
class Stratum:
    length: str
    position: str
    cited: int
    cited_hit: float
    uncited: int
    uncited_hit: float


_MESSAGES: Final = (
    "SELECT DISTINCT em.message_id AS id, m.content AS content, em.position AS position,"
    " (SELECT MAX(x.position) FROM exchange_messages x WHERE x.exchange_id = em.exchange_id)"
    " AS last"
    " FROM claim_run_exchanges e JOIN exchange_messages em ON em.exchange_id = e.exchange_id"
    " JOIN messages m ON m.id = em.message_id"
    " WHERE e.run_id IN ({marks}) AND e.outcome = 'ok' AND m.deleted_at IS NULL"
    " AND m.author_is_bot = 0 AND e.exchange_id IN ({exchanges})"
)
_CITED: Final = (
    "SELECT DISTINCT s.message_id FROM claims_v2_sources s JOIN claims_v2 c ON c.id = s.claim_id"
    " WHERE c.run_id IN ({marks})"
)


def matched_recall(
    conn: sqlite3.Connection,
    runs: Sequence[int],
    lexicon: Lexicon,
    rng: random.Random,
    sample: int,
) -> list[Stratum]:
    marks = ",".join("?" * len(runs))
    exchanges = [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT exchange_id FROM claim_run_exchanges WHERE run_id IN ({marks})"
            " AND outcome = 'ok' ORDER BY exchange_id",
            list(runs),
        )
    ]
    chosen = sorted(rng.sample(exchanges, min(sample, len(exchanges))))
    cited = {r[0] for r in conn.execute(_CITED.format(marks=marks), list(runs))}
    cells: dict[tuple[str, str], list[list[int]]] = defaultdict(lambda: [[0, 0], [0, 0]])
    for start in range(0, len(chosen), 500):
        batch = chosen[start : start + 500]
        sql = _MESSAGES.format(marks=marks, exchanges=",".join("?" * len(batch)))
        for row in conn.execute(sql, [*runs, *batch]):
            position = (
                "first"
                if row["position"] == 1
                else "last"
                if row["position"] == row["last"]
                else "middle"
            )
            cell = cells[(_length_bucket(len(row["content"].strip())), position)][
                1 if row["id"] in cited else 0
            ]
            cell[0] += 1
            cell[1] += 1 if message_hits(lexicon, row["content"]) else 0
    return [
        Stratum(
            length,
            position,
            cited[0],
            cited[1] / cited[0] if cited[0] else 0.0,
            uncited[0],
            uncited[1] / uncited[0] if uncited[0] else 0.0,
        )
        for (length, position), (uncited, cited) in sorted(cells.items())
    ]


DEFAULT_SAMPLE: Final = 5000
DEFAULT_RECALL_SAMPLE: Final = 5000


def _rows(conn: sqlite3.Connection, runs: Sequence[int]) -> list[ReviewRow]:
    from infovore.db.claims_v2 import review_rows

    return [row for run in runs for row in review_rows(conn, run)]


def analyse(
    conn: sqlite3.Connection,
    runs: Sequence[int],
    lexicon: Lexicon,
    threshold: float,
    sample: int,
    recall_sample: int,
    seed: int,
    article_run: int | None,
    drops_log: Path | None,
) -> dict[str, object]:
    from infovore.db.wiki_articles import sections_for

    rng = random.Random(seed)
    rows = _rows(conn, runs)
    chosen = rows if len(rows) <= sample else rng.sample(rows, sample)
    real, shuffled, within = shuffled_citations(chosen, lexicon, threshold, rng)
    floor = overlap_floor(chosen, lexicon, threshold, rng)
    report: dict[str, object] = {
        "runs": list(runs),
        "claims": len(rows),
        "sample": len(chosen),
        "shuffled": {
            "within_exchange_share": within,
            "real": {"n": real.n, "verdicts": real.verdicts, "mean_overlap": real.mean_overlap},
            "shuffled": {
                "n": shuffled.n,
                "verdicts": shuffled.verdicts,
                "mean_overlap": shuffled.mean_overlap,
            },
        },
        "floor": {
            "n": floor.n,
            "quantiles": floor.quantiles,
            "above_floor": floor.above_floor,
            "above_threshold": floor.above_threshold,
        },
        "recall": [s.__dict__ for s in matched_recall(conn, runs, lexicon, rng, recall_sample)],
        "drops": None,
    }
    if article_run is not None and drops_log is not None:
        verdict_of = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT k.claim_id, k.verdict FROM current_claim_checks k"
                " JOIN claims_v2 c ON c.id = k.claim_id"
                f" WHERE c.run_id IN ({','.join('?' * len(runs))})",
                list(runs),
            )
        }
        by_statement = {" ".join(r.statement.split()): r.claim_id for r in rows}
        report["drops"] = {
            v: {"kept": kept, "dropped": dropped}
            for v, (kept, dropped) in drop_rate_by_verdict(
                sections_for(conn, [article_run]), read_drops(drops_log), verdict_of, by_statement
            ).items()
        }
    return report


def render(report: Mapping[str, object]) -> str:
    shuffled, runs = report["shuffled"], report["runs"]
    assert isinstance(shuffled, dict) and isinstance(runs, list)
    real, other = shuffled["real"], shuffled["shuffled"]
    lines = [
        f"controls: runs={','.join(str(r) for r in runs)} claims={report['claims']}"
        f" sample={report['sample']}",
        "shuffled citations (each claim checked against another claim's messages,"
        f" {shuffled['within_exchange_share']:.0%} from the same exchange):",
    ]
    for verdict in VERDICTS:
        lines.append(
            f"  {verdict}: real {real['verdicts'][verdict] / real['n']:.3f}"
            f" shuffled {other['verdicts'][verdict] / other['n']:.3f}"
            if real["n"]
            else f"  {verdict}: n/a"
        )
    lines.append(
        f"  mean overlap: real {real['mean_overlap']:.3f} shuffled {other['mean_overlap']:.3f}"
    )
    floor = report["floor"]
    assert isinstance(floor, dict)
    quantiles = " ".join(f"{k} {v:.3f}" for k, v in floor["quantiles"].items())
    lines.append(
        f"random-pair overlap floor: n={floor['n']} {quantiles};"
        f" above support floor {floor['above_floor']:.3f},"
        f" above check threshold {floor['above_threshold']:.3f}"
    )
    lines.append("lexicon hit rate, cited vs uncited messages, by length and position:")
    recall = report["recall"]
    assert isinstance(recall, list)
    for s in recall:
        lines.append(
            f"  {s['length']:>8} {s['position']:>6}: cited {s['cited_hit']:.3f} (n={s['cited']})"
            f" uncited {s['uncited_hit']:.3f} (n={s['uncited']})"
        )
    drops = report["drops"]
    if drops is None:
        lines.append(
            "writer drop rate by claim verdict: skipped (needs --article-run and --drops-log)"
        )
    else:
        assert isinstance(drops, dict)
        lines.append("writer drop rate by the worst verdict among a sentence's cited claims:")
        for verdict, counts in drops.items():
            total = counts["kept"] + counts["dropped"]
            rate = counts["dropped"] / total
            lines.append(f"  {verdict}: dropped {counts['dropped']}/{total} ({rate:.3f})")
    return "\n".join(lines) + "\n"


def run_controls(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode
    from infovore.config import ConfigError
    from infovore.db.claims_v2 import run_ids
    from infovore.triage.lexicon import load_lexicon

    conn = context.conn
    known = run_ids(conn)
    for run in args.run:
        if run not in known:
            raise ConfigError(f"unknown run {run}")
    if (args.article_run is None) != (args.drops_log is None):
        raise ConfigError("--article-run and --drops-log go together")
    if args.drops_log is not None and not args.drops_log.exists():
        raise ConfigError(f"no drops log at {args.drops_log}")
    if args.sample < 2 or args.recall_sample < 1:
        raise ConfigError("--sample must be at least 2 and --recall-sample at least 1")
    report = analyse(
        conn,
        args.run,
        load_lexicon(conn),
        args.threshold,
        args.sample,
        args.recall_sample,
        args.seed,
        args.article_run,
        args.drops_log,
    )
    if args.as_json:
        context.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        context.stdout.write(render(report))
    return int(ExitCode.OK)
