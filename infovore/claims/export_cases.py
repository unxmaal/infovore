import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from infovore.claims.extract import SCHEMA, SYSTEM, render_window, windows
from infovore.claims.redact import RenderedLine, redact_conversation, require_salt
from infovore.config import ConfigError
from infovore.db.archived import SCORER_PREFIX, STAGES
from infovore.db.batch import BATCH_SIZE, exchange_inputs_for_ids
from infovore.db.claims_v2 import ReviewedClaim, reviewed_claims, run_ids
from infovore.rows import Label
from infovore.triage.human import trainable_labels

if TYPE_CHECKING:
    from infovore.cli import AppContext


def _parse_runs(text: str) -> list[int]:
    try:
        return list(dict.fromkeys(int(part) for part in text.split(",")))
    except ValueError as error:
        raise ConfigError("--runs must be comma-separated run ids") from error


def _checked_out(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    for parent in path.parents:
        if (parent / ".git").exists():
            raise ConfigError(
                f"--out {path} is inside the git work tree {parent}; transcripts are verbatim"
                " Discord text, write them outside any repository"
            )
    return path


def _window_id(eid: int, k: int, parts: list[Any]) -> str:
    return f"{eid}/{k}" if len(parts) > 1 else str(eid)


def _case(eid: int, k: int, parts: list[Any], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": _window_id(eid, k, parts),
        "system": SYSTEM,
        "transcript": render_window(parts[k]),
        "schema": SCHEMA,
        "reviews": reviews,
    }


def exchange_windows(
    conn: sqlite3.Connection, eid: int, salt: str, size: int
) -> tuple[list[list[RenderedLine]], dict[int, int]]:
    messages = exchange_inputs_for_ids(conn, [eid])[eid].messages
    redacted = redact_conversation(messages, salt)
    return windows(redacted.lines, size), {ln.message_id: ln.ref for ln in redacted.lines}


def _exchange_cases(
    context: "AppContext", eid: int, claims: list[ReviewedClaim], salt: str, size: int
) -> list[dict[str, Any]]:
    parts, ref_of = exchange_windows(context.conn, eid, salt, size)
    window_of = {line.ref: k for k, part in enumerate(parts) for line in part}
    kept: dict[tuple[int, str, str], ReviewedClaim] = {}
    refs_of: dict[tuple[int, str, str], list[int]] = {}
    for claim in claims:
        refs = sorted(ref_of[m] for m in claim.message_ids if m in ref_of)
        if not refs:
            continue
        key = (window_of[refs[0]], claim.speaker, claim.statement)
        if key not in kept or kept[key].review_id < claim.review_id:
            kept[key], refs_of[key] = claim, refs
    cases: list[dict[str, Any]] = []
    for k in sorted({key[0] for key in kept}):
        reviews = [
            {
                "user": c.speaker,
                "claim": c.statement,
                "refs": refs_of[key],
                "verdict": c.verdict,
                "interface": c.interface,
            }
            for key, c in kept.items()
            if key[0] == k
        ]
        cases.append(_case(eid, k, parts, reviews))
    return cases


def cascade_origins(conn: sqlite3.Connection, ids: list[int]) -> dict[int, str]:
    scorers = {f"{SCORER_PREFIX}{stage}": stage for stage in STAGES}
    marks = ", ".join("?" for _ in scorers)
    latest: dict[int, tuple[str, dict[str, str]]] = {}
    for start in range(0, len(ids), BATCH_SIZE):
        chunk = ids[start : start + BATCH_SIZE]
        rows = conn.execute(
            "SELECT subject_id, scorer, label, created_at FROM annotations"
            f" WHERE subject_kind = 'exchange' AND scorer IN ({marks})"
            f" AND subject_id IN ({', '.join('?' for _ in chunk)})",
            (*scorers, *chunk),
        )
        for row in rows:
            at, decided = latest.get(row["subject_id"], ("", {}))
            if row["created_at"] > at:
                at, decided = row["created_at"], {}
            if row["created_at"] == at and row["label"] is not None:
                decided[scorers[row["scorer"]]] = row["label"]
            latest[row["subject_id"]] = (at, decided)
    return {
        eid: "undecided" if stage == "residue" else stage
        for eid, (_, decided) in latest.items()
        for stage in STAGES
        if stage in decided
    }


def _negative_cases(context: "AppContext", salt: str, size: int) -> list[dict[str, Any]]:
    labels, _ = trainable_labels(context.conn, context.settings.exclude_channels)
    ids = sorted(eid for eid, label in labels.items() if label is Label.NOISE)
    origins = cascade_origins(context.conn, ids)
    cases: list[dict[str, Any]] = []
    for eid in ids:
        messages = exchange_inputs_for_ids(context.conn, [eid])[eid].messages
        parts = windows(redact_conversation(messages, salt).lines, size)
        extra = {
            "expect_empty": True,
            "basis": "human_irrelevant",
            "origin": origins.get(eid, "unsorted"),
        }
        cases.extend({**_case(eid, k, parts, []), **extra} for k in range(len(parts)))
    return cases


def _reviewed_cases(
    context: "AppContext", args: argparse.Namespace, salt: str
) -> list[dict[str, Any]]:
    runs = _parse_runs(args.runs)
    known = run_ids(context.conn)
    if missing := [r for r in runs if r not in known]:
        raise ConfigError(f"unknown run {', '.join(map(str, missing))}")
    by_exchange: dict[int, list[ReviewedClaim]] = {}
    for claim in reviewed_claims(context.conn, runs):
        by_exchange.setdefault(claim.exchange_id, []).append(claim)
    return [
        case
        for eid, claims in sorted(by_exchange.items())
        for case in _exchange_cases(context, eid, claims, salt, args.window_chars)
    ]


def export_cases(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode

    salt = require_salt(context.settings.pseudonym_salt)
    out = _checked_out(args.out)
    if args.negatives:
        cases = _negative_cases(context, salt, args.window_chars)
    else:
        cases = _reviewed_cases(context, args, salt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8"
    )
    context.stdout.write(f"wrote {len(cases)} cases to {out}\n")
    if args.negatives:
        for origin, count in sorted(Counter(c["origin"] for c in cases).items()):
            context.stdout.write(f"{origin}: {count}\n")
    return int(ExitCode.OK)
