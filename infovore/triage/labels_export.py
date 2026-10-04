import argparse
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from infovore.db.batch import exchange_inputs_for_ids
from infovore.rows import Label
from infovore.triage.cascade import SCORERS
from infovore.triage.human import HUMAN_SCORER, held_out_ids, training_labels
from infovore.triage.llm_score import render

if TYPE_CHECKING:
    from infovore.cli import AppContext

CHUNK: Final = 500
STAGE_OF: Final = {scorer: stage for stage, scorer in SCORERS.items()}


def _chunks(ids: Sequence[int]) -> Iterator[Sequence[int]]:
    for start in range(0, len(ids), CHUNK):
        yield ids[start : start + CHUNK]


def label_source(ref: str | None) -> str | None:
    if not ref:
        return None
    parts = ref.split(":")
    return parts[1] if parts[0] == "judge" and len(parts) > 1 else parts[0]


def _sources(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, str | None]:
    found: dict[int, str | None] = {}
    for chunk in _chunks(ids):
        marks = ",".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT subject_id, source_ref FROM annotations WHERE scorer = ?"
            " AND subject_kind = 'exchange' AND reproducibility = 'recorded'"
            f" AND subject_id IN ({marks}) ORDER BY id",
            (HUMAN_SCORER, *chunk),
        )
        found.update((row["subject_id"], label_source(row["source_ref"])) for row in rows)
    return found


def _stages(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, str]:
    found: dict[int, str] = {}
    scorers = list(STAGE_OF)
    scorer_marks = ",".join("?" for _ in scorers)
    for chunk in _chunks(ids):
        marks = ",".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT subject_id, scorer FROM annotations WHERE subject_kind = 'exchange'"
            f" AND scorer IN ({scorer_marks}) AND label IS NOT NULL"
            f" AND subject_id IN ({marks}) ORDER BY id",
            (*scorers, *chunk),
        )
        found.update((row["subject_id"], STAGE_OF[row["scorer"]]) for row in rows)
    return found


def _slices(conn: sqlite3.Connection, held: frozenset[int]) -> dict[int, str]:
    best: dict[int, tuple[bool, str]] = {}
    for row in conn.execute("SELECT exchange_id, name FROM current_slice_members"):
        eid = row["exchange_id"]
        key = (eid not in held, row["name"])
        if eid not in best or key < best[eid]:
            best[eid] = key
    return {eid: key[1] for eid, key in best.items()}


def _channels(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, str]:
    found: dict[int, str] = {}
    for chunk in _chunks(ids):
        marks = ",".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT e.id, COALESCE(p.name, c.name) AS name FROM current_exchanges e"
            " JOIN channels c ON c.id = e.channel_id"
            f" LEFT JOIN channels p ON p.id = c.parent_id WHERE e.id IN ({marks})",
            chunk,
        )
        found.update((row["id"], row["name"]) for row in rows)
    return found


def export_rows(conn: sqlite3.Connection, exclude: frozenset[str]) -> list[dict[str, Any]]:
    labels, _ = training_labels(conn, exclude_channels=exclude)
    ids = sorted(labels)
    held = held_out_ids(conn)
    sources, stages = _sources(conn, ids), _stages(conn, ids)
    slices, channels = _slices(conn, held), _channels(conn, ids)
    rows = []
    for chunk in _chunks(ids):
        inputs = exchange_inputs_for_ids(conn, chunk)
        for eid in chunk:
            rows.append(
                {
                    "id": eid,
                    "text": "\n".join(render(inputs[eid].messages)),
                    "label": "relevant" if labels[eid] is Label.LORE else "irrelevant",
                    "held_out": eid in held,
                    "slice": slices.get(eid),
                    "channel": channels[eid],
                    "cascade_stage": stages.get(eid),
                    "label_source": sources[eid],
                }
            )
    return rows


def summary(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [f"labels: {len(rows)}"]
    for field in ("label", "held_out", "cascade_stage"):
        counts = Counter("none" if r[field] is None else str(r[field]).lower() for r in rows)
        lines.extend(f"{field} {value}: {count}" for value, count in sorted(counts.items()))
    return lines


class LabelsCommand:
    name = "labels"
    help = "export human relevance labels"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(
            dest="labels_action", required=True, parser_class=argparse.ArgumentParser
        )
        export = sub.add_parser("export", help="write labelled current exchanges as JSONL")
        export.add_argument("--jsonl", required=True, type=Path)
        export.add_argument(
            "--include-excluded-channels", action="store_true", dest="include_excluded"
        )

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        exclude = frozenset() if args.include_excluded else context.settings.exclude_channels
        rows = export_rows(context.conn, exclude)
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        context.stdout.write(f"wrote {len(rows)} rows to {args.jsonl}\n")
        for line in summary(rows):
            context.stdout.write(line + "\n")
        return int(ExitCode.OK)
