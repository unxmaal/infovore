import csv
import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

from infovore.db.codec import to_db_time
from infovore.db.message_labels import set_message_label
from infovore.rows import MessageLabel, MessageLabelSource
from infovore.sift.export import BATCH_LOG_NAME, MANIFEST_NAME

KEPT_CSV_NAME = "kept.csv"
TRASH_REGEXES_CSV_NAME = "trash-regexes.csv"
TRASH_RULES_DIR_NAME = "sift_trash_rules"

_MSG_TAG = re.compile(r"\[msg:(\d+) ex:\d+ p:[-0-9.]+\]")


class MissingManifestError(Exception):
    pass


class NoSiftResultsFoundError(Exception):
    pass


@dataclass(frozen=True)
class ChannelCounts:
    keep: int = 0
    trash: int = 0


@dataclass(frozen=True)
class ImportReport:
    path: str
    source_ref: str
    keep: int
    trash: int
    by_channel: dict[str, ChannelCounts] = field(default_factory=dict)
    patterns: tuple[str, ...] = ()


def _read_manifest(dir_: Path) -> tuple[list[int], str]:
    manifest_path = dir_ / MANIFEST_NAME
    if not manifest_path.exists():
        raise MissingManifestError(str(manifest_path))
    manifest = json.loads(manifest_path.read_text())
    message_ids = [int(value) for value in cast(list[int], manifest["message_ids"])]
    created_at = str(manifest["created_at"])
    return message_ids, created_at


def _batch_log_lines(dir_: Path) -> dict[int, str]:
    """Every batch.log line, keyed by its `msg:<id>` tag — the exact text a
    human filtered against in lnav, reused here so a saved trash regex is
    matched the same way lnav matched it (issue #128)."""
    lines: dict[int, str] = {}
    for line in (dir_ / BATCH_LOG_NAME).read_text().splitlines():
        match = _MSG_TAG.search(line)
        if match is not None:
            lines[int(match.group(1))] = line
    return lines


def _read_kept_ids(path: Path) -> frozenset[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return frozenset(int(row["msg"]) for row in csv.DictReader(handle))


def read_trash_regexes(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["pattern"] for row in csv.DictReader(handle)]


def _channel_names(conn: sqlite3.Connection, message_ids: Sequence[int]) -> dict[int, str]:
    if not message_ids:
        return {}
    placeholders = ", ".join("?" * len(message_ids))
    rows = conn.execute(
        f"SELECT m.id AS id, c.name AS name FROM messages m"
        f" JOIN channels c ON c.id = m.channel_id"
        f" WHERE m.id IN ({placeholders})",
        tuple(message_ids),
    ).fetchall()
    return {row["id"]: row["name"] for row in rows}


def import_batch(conn: sqlite3.Connection, dir_: Path, at: datetime) -> ImportReport:
    """Read back a human's lnav sift session from `dir_` (issue #128) and
    record `message_labels` rows (`source='human'`), replacing any prior
    human label for the same message. Prefers the kept-lines path (a,
    `kept.csv`, an exact enumeration) over the trash-regex path (b,
    `trash-regexes.csv`, re-matched against `batch.log`) when both are
    present, since (a) is a direct statement of the outcome rather than a
    derived one. Raises `MissingManifestError`/`NoSiftResultsFoundError`
    (left for the caller to turn into a `ConfigError`) when `dir_` isn't a
    sift batch, or holds neither result file."""
    message_ids, created_at = _read_manifest(dir_)
    kept_path = dir_ / KEPT_CSV_NAME
    regex_path = dir_ / TRASH_REGEXES_CSV_NAME

    patterns: tuple[str, ...] = ()
    if kept_path.exists():
        path_used = "a"
        kept_ids = _read_kept_ids(kept_path)
        labels = {
            message_id: (MessageLabel.KEEP if message_id in kept_ids else MessageLabel.TRASH)
            for message_id in message_ids
        }
    elif regex_path.exists():
        path_used = "b"
        patterns = tuple(read_trash_regexes(regex_path))
        compiled = [re.compile(pattern) for pattern in patterns]
        lines = _batch_log_lines(dir_)
        labels = {
            message_id: (
                MessageLabel.TRASH
                if any(pattern.search(lines[message_id]) for pattern in compiled)
                else MessageLabel.KEEP
            )
            for message_id in message_ids
        }
    else:
        raise NoSiftResultsFoundError(str(dir_))

    source_ref = f"sift:{created_at}:{path_used}"
    channels = _channel_names(conn, message_ids)
    by_channel: dict[str, ChannelCounts] = {}
    keep_total = 0
    trash_total = 0
    for message_id in message_ids:
        label = labels[message_id]
        set_message_label(conn, message_id, label, MessageLabelSource.HUMAN, source_ref, at)
        channel = channels[message_id]
        counts = by_channel.get(channel, ChannelCounts())
        if label is MessageLabel.KEEP:
            by_channel[channel] = ChannelCounts(keep=counts.keep + 1, trash=counts.trash)
            keep_total += 1
        else:
            by_channel[channel] = ChannelCounts(keep=counts.keep, trash=counts.trash + 1)
            trash_total += 1

    return ImportReport(
        path=path_used,
        source_ref=source_ref,
        keep=keep_total,
        trash=trash_total,
        by_channel=by_channel,
        patterns=patterns,
    )


def save_trash_rules(scratch_dir: Path, name: str, patterns: Sequence[str], at: datetime) -> Path:
    """Store a named set of trash regexes (issue #128, `sift import
    --save-rules NAME`) for later corpus-wide use — this PR only stores
    them; corpus-wide application is a later issue."""
    rules_dir = scratch_dir / TRASH_RULES_DIR_NAME
    rules_dir.mkdir(parents=True, exist_ok=True)
    path = rules_dir / f"{name}.json"
    path.write_text(
        json.dumps({"name": name, "patterns": list(patterns), "saved_at": to_db_time(at)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path
