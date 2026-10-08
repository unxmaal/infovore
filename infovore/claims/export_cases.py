import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from infovore.claims.extract import SCHEMA, SYSTEM, render_window, windows
from infovore.claims.redact import redact_conversation, require_salt
from infovore.config import ConfigError
from infovore.db.batch import exchange_inputs_for_ids
from infovore.db.claims_v2 import ReviewedClaim, reviewed_claims, run_ids

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


def _exchange_cases(
    context: "AppContext", eid: int, claims: list[ReviewedClaim], salt: str, size: int
) -> list[dict[str, Any]]:
    messages = exchange_inputs_for_ids(context.conn, [eid])[eid].messages
    redacted = redact_conversation(messages, salt)
    parts = windows(redacted.lines, size)
    ref_of = {line.message_id: line.ref for line in redacted.lines}
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
            {"user": c.speaker, "claim": c.statement, "refs": refs_of[key], "verdict": c.verdict}
            for key, c in kept.items()
            if key[0] == k
        ]
        cases.append(
            {
                "id": f"{eid}/{k}" if len(parts) > 1 else str(eid),
                "system": SYSTEM,
                "transcript": render_window(parts[k]),
                "schema": SCHEMA,
                "reviews": reviews,
            }
        )
    return cases


def export_cases(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.cli import ExitCode

    salt = require_salt(context.settings.pseudonym_salt)
    out = _checked_out(args.out)
    runs = _parse_runs(args.runs)
    known = run_ids(context.conn)
    if missing := [r for r in runs if r not in known]:
        raise ConfigError(f"unknown run {', '.join(map(str, missing))}")
    by_exchange: dict[int, list[ReviewedClaim]] = {}
    for claim in reviewed_claims(context.conn, runs):
        by_exchange.setdefault(claim.exchange_id, []).append(claim)
    cases = [
        case
        for eid, claims in sorted(by_exchange.items())
        for case in _exchange_cases(context, eid, claims, salt, args.window_chars)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8"
    )
    context.stdout.write(f"wrote {len(cases)} cases to {out}\n")
    return int(ExitCode.OK)
