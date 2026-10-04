import argparse
import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from infovore.claims.redact import COMMON
from infovore.config import ConfigError
from infovore.db.claim_checks import CheckRow, current_checks, record_checks
from infovore.db.claims_v2 import ReviewRow, review_rows, run_ids
from infovore.triage.human import Gazetteer
from infovore.triage.lexicon import Lexicon, load_lexicon

if TYPE_CHECKING:
    from infovore.cli import AppContext

RECIPE: Final = "check-v1"
DEFAULT_THRESHOLD: Final = 0.35
VERDICTS: Final = ("supported", "unsupported_fact", "low_overlap", "uncheckable")
ERIC: Final = ("good", "not_useful", "wrong", "made_up")
BAD: Final = ("wrong", "made_up")
LEXICON_WEIGHT: Final = 2.0
GRID: Final = tuple(i / 20 for i in range(21))

_NUMBER_WORDS: Final = {
    w: str(n)
    for n, w in enumerate(
        [
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        ],
        start=2,
    )
}
_UNITS: Final = {
    "mhz": "mhz",
    "megahertz": "mhz",
    "ghz": "ghz",
    "khz": "khz",
    "kb": "kb",
    "kib": "kb",
    "kilobyte": "kb",
    "kilobytes": "kb",
    "mb": "mb",
    "mib": "mb",
    "meg": "mb",
    "megs": "mb",
    "megabyte": "mb",
    "megabytes": "mb",
    "gb": "gb",
    "gib": "gb",
    "gig": "gb",
    "gigs": "gb",
    "gigabyte": "gb",
    "gigabytes": "gb",
    "tb": "tb",
    "terabyte": "tb",
    "terabytes": "tb",
    "byte": "bytes",
    "bytes": "bytes",
    "kw": "kw",
    "kilowatt": "kw",
    "kilowatts": "kw",
    "watts": "w",
    "volts": "v",
    "rpm": "rpm",
    "mips": "mips",
}
_UNIT_ALT: Final = "|".join(sorted(_UNITS, key=len, reverse=True))
_WORD_NUM: Final = re.compile(rf"\b({'|'.join(_NUMBER_WORDS)})\b(?!\s+(?:of|more|other|thing)\b)")
_ONE_UNIT: Final = re.compile(rf"\bone\s+(?=(?:{_UNIT_ALT})\b)")
_KILO: Final = re.compile(r"(?<![\w.])(\d+)k\b")
_ORIGIN: Final = re.compile(r"\bo(\d{3,4})\b")
_THOUSANDS: Final = re.compile(r"(?<=\d),(?=\d{3}\b)")
_RK: Final = re.compile(r"\br(\d{1,2})k\b")
_MENTION: Final = re.compile(r"<[@#][!&]?\d+>")
_PSEUDONYM: Final = re.compile(r"\buser-[0-9a-f]+\b")
_SPACE: Final = re.compile(r"\s+")
_BACKTICK: Final = re.compile(r"`([^`\n]{2,80})`")
_QUOTE: Final = re.compile(r'"([^"\n]{3,80})"')
_QUANTITY: Final = re.compile(rf"(?<![\w.])(\d+(?:\.\d+)?)\s*({_UNIT_ALT})\b")
_VERSION: Final = re.compile(r"(?<![\w.])\d+(?:\.\d+)+[mf]?(?![\w.]*\d)")
_CPU: Final = re.compile(r"\br\d{3,5}\b")
_YEAR: Final = re.compile(r"(?<![\w.])(?:19[6-9]\d|20[0-2]\d)(?![\w.]*\d)")
_INTEGER: Final = re.compile(r"(?<![\w.])\d+(?![\w]*\d|\.\d)")
_WORD: Final = re.compile(r"[a-z][a-z0-9]{2,}")
_STOP: Final = COMMON | frozenset(
    [
        "said",
        "says",
        "stated",
        "states",
        "mentioned",
        "mentions",
        "claimed",
        "claims",
        "reported",
        "reports",
        "according",
        "thinks",
        "believes",
        "noted",
        "notes",
    ]
)


@dataclass(frozen=True)
class Fact:
    kind: str
    value: str


@dataclass(frozen=True)
class Checked:
    verdict: str
    overlap: float
    facts: list[tuple[Fact, bool]]

    def missing(self) -> list[Fact]:
        return [fact for fact, found in self.facts if not found]


def normalize(text: str) -> str:
    text = _PSEUDONYM.sub(" ", _MENTION.sub(" ", text.lower()))
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = _THOUSANDS.sub("", text)
    text = _RK.sub(lambda m: f"r{int(m.group(1)) * 1000}", text)
    text = _KILO.sub(lambda m: str(int(m.group(1)) * 1000), text)
    text = _ORIGIN.sub(r"origin \1", text)
    text = _ONE_UNIT.sub("1 ", text)
    text = _WORD_NUM.sub(lambda m: _NUMBER_WORDS[m.group(1)], text)
    return _SPACE.sub(" ", text).strip()


def _quantity(match: re.Match[str]) -> str:
    return f"{float(match.group(1)):g} {_UNITS[match.group(2)]}"


def _version(match: re.Match[str]) -> str:
    return match.group(0).rstrip("mf")


def extract_facts(text: str, gazetteer: Gazetteer) -> list[Fact]:
    rest = normalize(text)
    found: list[Fact] = []

    def take(pattern: re.Pattern[str], kind: str, value: Callable[[re.Match[str]], str]) -> None:
        nonlocal rest
        for match in pattern.finditer(rest):
            found.append(Fact(kind, value(match)))
        rest = pattern.sub(" ", rest)

    take(_BACKTICK, "file", lambda m: m.group(1))
    take(_QUOTE, "quote", lambda m: m.group(1))
    take(gazetteer.text["part_number"], "part", lambda m: m.group(0))
    take(gazetteer.text["models"], "model", lambda m: _SPACE.sub(" ", m.group(0)))
    take(_QUANTITY, "quantity", _quantity)
    take(_VERSION, "version", _version)
    take(_CPU, "cpu", lambda m: m.group(0))
    take(_YEAR, "year", lambda m: m.group(0))
    take(_INTEGER, "number", lambda m: m.group(0))
    return list(dict.fromkeys(found))


def _source_values(texts: Sequence[str], gazetteer: Gazetteer) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for text in texts:
        for fact in extract_facts(text, gazetteer):
            values.setdefault(fact.kind, set()).add(fact.value)
        values.setdefault("digits", set()).update(re.findall(r"\d+", normalize(text)))
    return values


def _verify(fact: Fact, source: str, values: dict[str, set[str]]) -> bool:
    if fact.kind in ("file", "quote"):
        return fact.value in source
    if fact.kind == "number":
        return fact.value in values.get("digits", set())
    pool = values.get(fact.kind, set())
    if fact.kind == "version":
        return any(v == fact.value or v.startswith(fact.value + ".") for v in pool)
    if fact.kind == "model":
        return any(v.startswith(fact.value) or fact.value.startswith(v) for v in pool)
    return fact.value in pool


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _stems(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(normalize(text))}


def overlap(statement: str, sources: Sequence[str], lexicon: Lexicon) -> float:
    claim = [w for w in _WORD.findall(normalize(statement)) if w not in _STOP]
    if not claim:
        return 1.0
    seen = set().union(*(_stems(s) for s in sources)) if sources else set()
    total = got = 0.0
    for word in claim:
        weight = LEXICON_WEIGHT if word in lexicon.terms or _stem(word) in lexicon.terms else 1.0
        total += weight
        got += weight if _stem(word) in seen else 0.0
    return got / total


def check_claim(
    statement: str, sources: Sequence[str], lexicon: Lexicon, threshold: float
) -> Checked:
    gazetteer = lexicon.gazetteer
    joined = " \n ".join(normalize(s) for s in sources)
    values = _source_values(sources, gazetteer)
    facts = [(f, _verify(f, joined, values)) for f in extract_facts(statement, gazetteer)]
    score = overlap(statement, sources, lexicon)
    if any(not found for _, found in facts):
        verdict = "unsupported_fact"
    elif score < threshold:
        verdict = "low_overlap"
    else:
        verdict = "supported" if facts else "uncheckable"
    return Checked(verdict, score, facts)


def flagged(verdict: str) -> bool:
    return verdict in ("unsupported_fact", "low_overlap")


@dataclass(frozen=True)
class Scored:
    has_missing: bool
    overlap: float
    verdict: str


def flag_at(item: Scored, threshold: float) -> bool:
    return item.has_missing or item.overlap < threshold


@dataclass(frozen=True)
class Rates:
    threshold: float
    caught: int
    positives: int
    false_alarms: int
    negatives: int
    flags: int

    @property
    def precision(self) -> float:
        return self.caught / self.flags if self.flags else 0.0

    @property
    def recall(self) -> float:
        return self.caught / self.positives if self.positives else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def rates(items: Iterable[tuple[Scored, str]], threshold: float) -> Rates:
    caught = positives = false_alarms = negatives = 0
    for scored, eric in items:
        hit = flag_at(scored, threshold)
        if eric in BAD:
            positives += 1
            caught += hit
        elif eric == "good":
            negatives += 1
            false_alarms += hit
    flags = caught + false_alarms
    return Rates(threshold, caught, positives, false_alarms, negatives, flags)


def tune(items: Sequence[tuple[Scored, str]]) -> Rates:
    return max(
        (rates(items, t) for t in GRID),
        key=lambda r: (r.f1, -r.threshold),
    )


def confusion(items: Iterable[tuple[str, str]]) -> str:
    counts = Counter(items)
    head = "check\\eric".ljust(18) + "".join(v.rjust(11) for v in ERIC)
    lines = [head]
    for verdict in VERDICTS:
        row = "".join(str(counts[(verdict, e)]).rjust(11) for e in ERIC)
        lines.append(verdict.ljust(18) + row)
    return "\n".join(lines)


def _format_rates(r: Rates) -> str:
    return (
        f"threshold {r.threshold:.2f}: caught {r.caught} of {r.positives} wrong+made_up,"
        f" false alarms {r.false_alarms} of {r.negatives} good,"
        f" precision {r.precision:.2f} recall {r.recall:.2f}"
    )


def _sources(row: ReviewRow) -> list[str]:
    return [content for _, _, content in row.sources]


def _fact_text(fact: Fact) -> str:
    return f"{fact.kind}:{fact.value}"


def run_check(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode

    conn, out = context.conn, context.stdout
    known = run_ids(conn)
    for run in args.run:
        if run not in known:
            raise ConfigError(f"unknown run {run}")
    lexicon = load_lexicon(conn)
    recipe = {"recipe": RECIPE, "threshold": args.threshold, "lexicon": lexicon.version}
    items: list[tuple[Scored, str]] = []
    pairs: list[tuple[str, str]] = []
    for run in args.run:
        rows = review_rows(conn, run)
        results = [check_claim(r.statement, _sources(r), lexicon, args.threshold) for r in rows]
        tally = Counter(c.verdict for c in results)
        out.write(
            f"run {run}: {len(rows)} claims, "
            + ", ".join(f"{v} {tally[v]}" for v in VERDICTS)
            + "\n"
        )
        for row, checked in zip(rows, results, strict=True):
            if flagged(checked.verdict):
                missing = ", ".join(_fact_text(f) for f in checked.missing()) or "-"
                out.write(
                    f"  {row.claim_id}\t{checked.verdict}\toverlap {checked.overlap:.2f}"
                    f"\t{missing}\t{row.verdict or '-'}\n"
                )
            if row.verdict:
                items.append(
                    (Scored(bool(checked.missing()), checked.overlap, checked.verdict), row.verdict)
                )
                pairs.append((checked.verdict, row.verdict))
        if args.write:
            record_checks(
                conn,
                [
                    CheckRow(
                        r.claim_id,
                        c.verdict,
                        c.overlap,
                        [{"kind": f.kind, "value": f.value, "found": ok} for f, ok in c.facts],
                    )
                    for r, c in zip(rows, results, strict=True)
                ],
                recipe,
                context.clock.now(),
            )
    out.write("written\n" if args.write else "not written (use --write)\n")
    out.write(f"{len(pairs)} reviewed claims\n{confusion(pairs)}\n")
    out.write(_format_rates(rates(items, args.threshold)) + "\n")
    out.write("tuned " + _format_rates(tune(items)) + "\n")
    return int(ExitCode.OK)


def show_checks(conn: sqlite3.Connection, run_id: int, write: Callable[[str], object]) -> None:
    stored = current_checks(conn, run_id)
    for row in review_rows(conn, run_id):
        check = stored.get(row.claim_id)
        if check is None:
            write(f"{row.claim_id}\t-\t{row.speaker}: {row.statement}\n")
            continue
        failing = ", ".join(f"{f['kind']}:{f['value']}" for f in check.facts if not f["found"])
        tail = f"\tmissing {failing}" if failing else ""
        write(
            f"{row.claim_id}\t{check.verdict}\toverlap {check.overlap:.2f}\t"
            f"{row.speaker}: {row.statement}{tail}\n"
        )
