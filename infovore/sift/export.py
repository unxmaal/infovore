import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from infovore.db.codec import from_db_time, to_db_time
from infovore.sift.lnav_format import LNAV_FORMAT_JSON
from infovore.sift.sampling import SiftStrategy, select_sift_sample

BATCH_TEXT_LIMIT = 300
BATCH_LOG_NAME = "batch.log"
FORMAT_NAME = "infovore-sift.json"
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class SiftBatchMessage:
    id: int
    exchange_id: int
    channel_name: str
    author_name: str
    created_at: datetime
    content: str
    p_trash: float | None


@dataclass(frozen=True)
class ExportReport:
    count: int
    out_dir: Path
    batch_log_path: Path
    format_path: Path
    manifest_path: Path


def fetch_batch_messages(
    conn: sqlite3.Connection, message_ids: Sequence[int]
) -> list[SiftBatchMessage]:
    """Full row detail for a batch's messages, in chronological order — the
    order `batch.log` and `manifest.json`'s `message_ids` both use, so a
    human sifting in lnav reads the batch like a log, oldest first."""
    if not message_ids:
        return []
    placeholders = ", ".join("?" * len(message_ids))
    rows = conn.execute(
        f"SELECT m.id AS id, em.exchange_id AS exchange_id, c.name AS channel_name,"
        f" m.author_name_at_time AS author_name, m.created_at AS created_at,"
        f" m.content AS content, m.p_trash AS p_trash"
        f" FROM messages m"
        f" JOIN exchange_messages em ON em.message_id = m.id"
        f" JOIN channels c ON c.id = m.channel_id"
        f" WHERE m.id IN ({placeholders})"
        f" ORDER BY m.created_at, m.id",
        tuple(message_ids),
    ).fetchall()
    return [
        SiftBatchMessage(
            id=row["id"],
            exchange_id=row["exchange_id"],
            channel_name=row["channel_name"],
            author_name=row["author_name"],
            created_at=from_db_time(row["created_at"]),
            content=row["content"],
            p_trash=row["p_trash"],
        )
        for row in rows
    ]


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, limit: int = BATCH_TEXT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def render_batch_line(row: SiftBatchMessage) -> str:
    """One `batch.log` line (issue #128), matching
    `infovore.sift.lnav_format.LNAV_FORMAT_JSON`'s regex: ISO timestamp,
    `#channel`, author display name, a `[msg:<id> ex:<exchange_id>
    p:<p_trash or ->]` tag, then the message text collapsed to one line and
    truncated to `BATCH_TEXT_LIMIT` characters with an ellipsis."""
    p_text = f"{row.p_trash:.2f}" if row.p_trash is not None else "-"
    body = _truncate(_collapse(row.content))
    return (
        f"{to_db_time(row.created_at)} #{row.channel_name} {row.author_name}"
        f" [msg:{row.id} ex:{row.exchange_id} p:{p_text}] {body}"
    )


def export_batch(
    conn: sqlite3.Connection,
    *,
    size: int,
    strategy: SiftStrategy,
    seed: int,
    mix: float,
    out_dir: Path,
    now: datetime,
    exclude_channels: frozenset[str] = frozenset(),
    include_channels: frozenset[str] = frozenset(),
    repeat: int = 0,
) -> ExportReport:
    """Write one sift batch (issue #128) to `out_dir`: `batch.log` (the
    lnav-friendly log), `infovore-sift.json` (the lnav format file — a copy
    of the shipped, lnav-verified format, so the batch is self-contained
    and doesn't depend on the format already being installed), and
    `manifest.json` (message ids in batch order, strategy, seed,
    `created_at`). Raises `infovore.sift.sampling.NoScoredMessagesError`
    (propagated, uncaught) before writing anything if `strategy` is
    `uncertain` and no message has a `p_trash` yet. `exclude_channels` (the
    denylist, issue #138) and `include_channels` (`--channels`) narrow the
    sampling pool by channel name; the denylist always wins."""
    message_ids = select_sift_sample(
        conn,
        size,
        seed,
        strategy,
        mix=mix,
        exclude_channels=exclude_channels,
        include_channels=include_channels,
        repeat=repeat,
    )
    batch = fetch_batch_messages(conn, message_ids)

    out_dir.mkdir(parents=True, exist_ok=True)

    batch_log_path = out_dir / BATCH_LOG_NAME
    batch_log_path.write_text(
        "".join(render_batch_line(row) + "\n" for row in batch), encoding="utf-8"
    )

    format_path = out_dir / FORMAT_NAME
    format_path.write_text(LNAV_FORMAT_JSON, encoding="utf-8")

    manifest_path = out_dir / MANIFEST_NAME
    manifest = {
        "message_ids": [row.id for row in batch],
        "strategy": strategy.value,
        "seed": seed,
        "size": size,
        "created_at": to_db_time(now),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return ExportReport(
        count=len(batch),
        out_dir=out_dir,
        batch_log_path=batch_log_path,
        format_path=format_path,
        manifest_path=manifest_path,
    )
