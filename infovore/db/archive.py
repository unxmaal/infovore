import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from infovore.db.channel_filter import exclude_channels_clause
from infovore.db.fts import TOKENIZE
from infovore.privacy.optout import REDACTED_CONTENT
from infovore.triage.gate import gate_sql

_SCHEMA = f"""
CREATE TABLE channels (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE exchanges (
    id INTEGER PRIMARY KEY,
    channel_id INTEGER NOT NULL REFERENCES channels (id),
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    exchange_id INTEGER NOT NULL REFERENCES exchanges (id),
    position INTEGER NOT NULL,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL REFERENCES channels (id),
    author_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX messages_exchange ON messages (exchange_id, position);
CREATE VIRTUAL TABLE messages_fts USING fts5 (
    content, content = 'messages', content_rowid = 'id', tokenize = "{TOKENIZE}"
);
"""


@dataclass(frozen=True)
class ArchiveReport:
    dest: Path
    exchanges: int
    messages: int
    size_bytes: int


def export_archive(
    conn: sqlite3.Connection,
    dest: Path | str,
    min_score: float,
    min_p_lore: float,
    exclude_channels: frozenset[str] = frozenset(),
    force: bool = False,
) -> ArchiveReport:
    """A shareable copy holding only what search needs: gate-passing exchanges
    and their messages, minus opted-out authors, already-redacted rows and
    deleted rows. Author ids, raw JSON, revisions, attachments, claims and
    every other table stay behind (issue #191)."""
    dest_path = Path(dest)
    if dest_path.exists() and not force:
        raise FileExistsError(f"{dest_path} already exists; pass force=True to overwrite")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=dest_path.parent, prefix=f".{dest_path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        out = sqlite3.connect(tmp_path)
        try:
            exchanges, messages = _write(conn, out, min_score, min_p_lore, exclude_channels)
        finally:
            out.close()
        os.replace(tmp_path, dest_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return ArchiveReport(
        dest=dest_path, exchanges=exchanges, messages=messages, size_bytes=dest_path.stat().st_size
    )


def _write(
    conn: sqlite3.Connection,
    out: sqlite3.Connection,
    min_score: float,
    min_p_lore: float,
    exclude_channels: frozenset[str],
) -> tuple[int, int]:
    gate_clause, gate_params = gate_sql(min_score, min_p_lore)
    excl_clause, excl_params = exclude_channels_clause("e.channel_id", exclude_channels)
    out.executescript(_SCHEMA)
    rows = conn.execute(
        "SELECT m.id, em.exchange_id, em.position, m.guild_id, m.channel_id,"
        " m.author_name_at_time, m.created_at, m.content"
        " FROM exchanges e"
        " JOIN exchange_messages em ON em.exchange_id = e.id"
        " JOIN messages m ON m.id = em.message_id"
        f" WHERE {gate_clause}{excl_clause}"
        "   AND m.author_id NOT IN (SELECT user_id FROM opt_outs)"
        "   AND m.content != ? AND m.deleted_at IS NULL"
        " ORDER BY em.exchange_id, em.position",
        (*gate_params, *excl_params, REDACTED_CONTENT),
    ).fetchall()
    exchange_ids = sorted({row[1] for row in rows})
    for start in range(0, len(exchange_ids), 500):
        chunk = exchange_ids[start : start + 500]
        marks = ",".join("?" for _ in chunk)
        out.executemany(
            "INSERT INTO exchanges (id, channel_id, started_at, ended_at) VALUES (?, ?, ?, ?)",
            conn.execute(
                f"SELECT id, channel_id, started_at, ended_at FROM exchanges WHERE id IN ({marks})",
                chunk,
            ).fetchall(),
        )
    channel_ids = sorted(
        {row[4] for row in rows} | {r[0] for r in out.execute("SELECT channel_id FROM exchanges")}
    )
    for channel_id in channel_ids:
        name = conn.execute("SELECT name FROM channels WHERE id = ?", (channel_id,)).fetchone()
        out.execute("INSERT INTO channels (id, name) VALUES (?, ?)", (channel_id, name[0]))
    out.executemany(
        "INSERT INTO messages (id, exchange_id, position, guild_id, channel_id, author_name,"
        " created_at, content) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [tuple(row) for row in rows],
    )
    out.execute(
        "UPDATE exchanges SET message_count ="
        " (SELECT COUNT(*) FROM messages WHERE messages.exchange_id = exchanges.id)"
    )
    out.execute("INSERT INTO messages_fts (messages_fts) VALUES ('rebuild')")
    out.commit()
    return len(exchange_ids), len(rows)
