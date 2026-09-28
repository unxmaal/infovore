"""Server-side state for `sift serve` (issue #131): resolving which batch to
serve, loading its messages, and `ServeApp` — labeling, undo, bulk-rule
preview/apply and progress, all in plain, fully unit-tested Python.
`infovore.sift.httpd` only translates HTTP <-> these calls, and the page
itself only calls that HTTP layer.

**Connection strategy.** `ServeApp` is handed the same `sqlite3.Connection`
the rest of the CLI already opens via `infovore.db.connection.open_database`
(WAL, `busy_timeout`, `check_same_thread=False`), rather than opening one
per request. That connection is safe to share with the extract/probe
processes' *own* connections because of WAL + `busy_timeout`; what it is
*not* safe against is two of `sift serve`'s own HTTP worker threads
interleaving statements on the one connection object at the same time
(`ThreadingHTTPServer` dispatches each request on its own thread, and one
process runs one `ServeApp` shared across every bound address). A single
`threading.Lock` around each write path (`label`, `undo`,
`trash_rest_of_channel`, `apply_rule`) serializes exactly that, and nothing
more: every write happens inside `infovore.db.connection.transaction`
(`BEGIN IMMEDIATE` ... `COMMIT`), held only for the duration of one small
write, never across a request boundary. Reads (`state`, `progress`,
`preview_rule`) take no lock, matching the read-heavy, mostly-append
workload described in the issue.
"""

import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from infovore.db.connection import transaction
from infovore.db.message_labels import set_message_label
from infovore.rows import MessageLabel, MessageLabelSource
from infovore.sift.export import MANIFEST_NAME, SiftBatchMessage, export_batch, fetch_batch_messages
from infovore.sift.importer import MissingManifestError
from infovore.sift.rules import BulkRule, RulePreview, preview_rule, save_bulk_rule
from infovore.sift.sampling import SiftStrategy
from infovore.timing import Clock

_LabelRow = tuple[str, "str | None", str]


class UnknownMessageError(Exception):
    pass


@dataclass(frozen=True)
class Progress:
    total: int
    labeled: int
    keep: int
    trash: int
    remaining_by_channel: dict[str, int]


@dataclass(frozen=True)
class BatchSource:
    dir: Path
    message_ids: list[int]


def _opted_out_ids(conn: sqlite3.Connection, message_ids: Sequence[int]) -> frozenset[int]:
    if not message_ids:
        return frozenset()
    placeholders = ", ".join("?" * len(message_ids))
    rows = conn.execute(
        f"SELECT m.id AS id FROM messages m"
        f" JOIN opt_outs o ON o.user_id = m.author_id"
        f" WHERE m.id IN ({placeholders})",
        tuple(message_ids),
    ).fetchall()
    return frozenset(row["id"] for row in rows)


def load_batch_messages(
    conn: sqlite3.Connection, message_ids: Sequence[int]
) -> list[SiftBatchMessage]:
    """Batch messages ready to serve (issue #131): reuses
    `infovore.sift.export.fetch_batch_messages` for row detail and
    chronological ordering, then re-excludes any author who has opted out
    since the batch was exported. Sampling already excludes opted-out
    authors (`infovore.sift.sampling.eligible_message_pool`), but a batch
    can be exported once and served much later, so this is a second,
    defense-in-depth check at serve time — an opted-out author's messages
    must never be served, full stop."""
    excluded = _opted_out_ids(conn, message_ids)
    ids = message_ids if not excluded else [mid for mid in message_ids if mid not in excluded]
    return fetch_batch_messages(conn, ids)


def resolve_batch(
    conn: sqlite3.Connection,
    *,
    dir_: Path | None,
    new: bool,
    size: int,
    strategy: SiftStrategy,
    seed: int,
    mix: float,
    out_dir: Path | None,
    now: datetime,
    exclude_channels: frozenset[str] = frozenset(),
    include_channels: frozenset[str] = frozenset(),
) -> BatchSource:
    """Which batch `sift serve` shows (issue #131): either an existing
    export dir (`dir_`, read as-is), or a freshly sampled one (`new=True`,
    reusing `infovore.sift.export.export_batch` — the same sampling,
    `batch.log` and `manifest.json` a plain `sift export` writes, so the
    batch is resumable and importable the ordinary way too). Raises
    `infovore.sift.importer.MissingManifestError` if `dir_` isn't a sift
    batch, and propagates `infovore.sift.sampling.NoScoredMessagesError`
    uncaught (`new=True`, `strategy=uncertain`, nothing scored yet) for the
    caller to turn into a `ConfigError`, same as plain `sift export`."""
    if new:
        assert out_dir is not None
        export_batch(
            conn,
            size=size,
            strategy=strategy,
            seed=seed,
            mix=mix,
            out_dir=out_dir,
            now=now,
            exclude_channels=exclude_channels,
            include_channels=include_channels,
        )
        target_dir = out_dir
    else:
        assert dir_ is not None
        target_dir = dir_

    manifest_path = target_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise MissingManifestError(str(manifest_path))
    manifest = json.loads(manifest_path.read_text())
    message_ids = [int(value) for value in manifest["message_ids"]]
    return BatchSource(dir=target_dir, message_ids=message_ids)


class _Action:
    def revert(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError  # pragma: no cover - interface only, always overridden


@dataclass
class _SingleAction(_Action):
    message_id: int
    prior: _LabelRow | None

    def revert(self, conn: sqlite3.Connection) -> None:
        _restore_label_row(conn, self.message_id, self.prior)


@dataclass
class _BulkAction(_Action):
    priors: dict[int, _LabelRow | None]

    def revert(self, conn: sqlite3.Connection) -> None:
        for message_id, prior in self.priors.items():
            _restore_label_row(conn, message_id, prior)


def _read_label_row(conn: sqlite3.Connection, message_id: int) -> _LabelRow | None:
    row = conn.execute(
        "SELECT label, source_ref, labeled_at FROM message_labels"
        " WHERE message_id = ? AND source = ?",
        (message_id, MessageLabelSource.HUMAN.value),
    ).fetchone()
    if row is None:
        return None
    return (row["label"], row["source_ref"], row["labeled_at"])


def _restore_label_row(conn: sqlite3.Connection, message_id: int, prior: _LabelRow | None) -> None:
    """Undo's core primitive: put a message's `human` `message_labels` row
    back exactly how it was before the action being undone — deleting it
    if there wasn't one, or restoring the previous label/source_ref/
    labeled_at verbatim if there was (so undoing a re-label doesn't just
    clear it, it restores the earlier human call)."""
    if prior is None:
        conn.execute(
            "DELETE FROM message_labels WHERE message_id = ? AND source = ?",
            (message_id, MessageLabelSource.HUMAN.value),
        )
        return
    label, source_ref, labeled_at = prior
    conn.execute(
        "INSERT INTO message_labels (message_id, label, source, source_ref, labeled_at)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT (message_id, source) DO UPDATE SET"
        " label = excluded.label, source_ref = excluded.source_ref,"
        " labeled_at = excluded.labeled_at",
        (message_id, label, MessageLabelSource.HUMAN.value, source_ref, labeled_at),
    )


def _progress_dict(progress: Progress) -> dict[str, object]:
    return {
        "total": progress.total,
        "labeled": progress.labeled,
        "keep": progress.keep,
        "trash": progress.trash,
        "remaining_by_channel": progress.remaining_by_channel,
    }


class ServeApp:
    """One running `sift serve` batch: its messages, a lock-guarded write
    path over the shared connection (see the module docstring), and an
    in-memory undo log. The undo log is process-memory only — a maintainer
    restarting the server loses the ability to undo past actions, but
    never loses the labels themselves (those are in `message_labels`,
    which is what makes reloading the page, or resuming after a restart,
    show already-labeled messages correctly: `state`/`progress` always
    re-read the database rather than trusting any cached label)."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        messages: list[SiftBatchMessage],
        batch_name: str,
        scratch_dir: Path,
        clock: Clock,
    ) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        self._messages = messages
        self._by_id = {message.id: message for message in messages}
        self.batch_name = batch_name
        self._scratch_dir = scratch_dir
        self._clock = clock
        self._actions: list[_Action] = []

    @property
    def source_ref(self) -> str:
        return f"sift-serve:{self.batch_name}"

    def messages(self) -> list[SiftBatchMessage]:
        return list(self._messages)

    def _current_labels(self) -> dict[int, MessageLabel]:
        if not self._messages:
            return {}
        placeholders = ", ".join("?" * len(self._messages))
        rows = self._conn.execute(
            f"SELECT message_id, label FROM message_labels"
            f" WHERE source = ? AND message_id IN ({placeholders})",
            (MessageLabelSource.HUMAN.value, *(message.id for message in self._messages)),
        ).fetchall()
        return {row["message_id"]: MessageLabel(row["label"]) for row in rows}

    def progress(self) -> Progress:
        labels = self._current_labels()
        keep = sum(1 for label in labels.values() if label is MessageLabel.KEEP)
        trash = sum(1 for label in labels.values() if label is MessageLabel.TRASH)
        remaining: dict[str, int] = {}
        for message in self._messages:
            if message.id not in labels:
                remaining[message.channel_name] = remaining.get(message.channel_name, 0) + 1
        return Progress(
            total=len(self._messages),
            labeled=len(labels),
            keep=keep,
            trash=trash,
            remaining_by_channel=remaining,
        )

    def state(self) -> dict[str, object]:
        labels = self._current_labels()
        return {
            "batch": self.batch_name,
            "messages": [
                {
                    "id": str(message.id),
                    "exchange_id": message.exchange_id,
                    "channel": message.channel_name,
                    "author": message.author_name,
                    "created_at": message.created_at.isoformat(),
                    "content": message.content,
                    "p_trash": message.p_trash,
                    "label": labels[message.id].value if message.id in labels else None,
                }
                for message in self._messages
            ],
            "progress": _progress_dict(self.progress()),
        }

    def label(self, message_id: int, label: MessageLabel) -> None:
        if message_id not in self._by_id:
            raise UnknownMessageError(message_id)
        with self._lock:
            prior = _read_label_row(self._conn, message_id)
            with transaction(self._conn):
                set_message_label(
                    self._conn,
                    message_id,
                    label,
                    MessageLabelSource.HUMAN,
                    self.source_ref,
                    self._clock.now(),
                )
            self._actions.append(_SingleAction(message_id, prior))

    def undo(self) -> bool:
        with self._lock:
            if not self._actions:
                return False
            action = self._actions.pop()
            with transaction(self._conn):
                action.revert(self._conn)
            return True

    def trash_rest_of_channel(self, channel: str) -> int:
        with self._lock:
            labels = self._current_labels()
            targets = [
                message.id
                for message in self._messages
                if message.channel_name == channel and message.id not in labels
            ]
            if not targets:
                return 0
            priors = {message_id: _read_label_row(self._conn, message_id) for message_id in targets}
            with transaction(self._conn):
                for message_id in targets:
                    set_message_label(
                        self._conn,
                        message_id,
                        MessageLabel.TRASH,
                        MessageLabelSource.HUMAN,
                        self.source_ref,
                        self._clock.now(),
                    )
            self._actions.append(_BulkAction(priors))
            return len(targets)

    def preview_rule(self, rule: BulkRule) -> RulePreview:
        return preview_rule(self._messages, self._current_labels(), rule)

    def apply_rule(self, rule: BulkRule, name: str) -> tuple[RulePreview, Path]:
        """Trash every batch message the rule matches (issue #131:
        "every match becomes trash", overriding an existing `keep` — the
        preview already warned about those conflicts before the maintainer
        confirmed), as one undoable bulk action, then save the rule
        (`infovore.sift.rules.save_bulk_rule`)."""
        with self._lock:
            preview = preview_rule(self._messages, self._current_labels(), rule)
            if preview.matched:
                priors = {
                    message_id: _read_label_row(self._conn, message_id)
                    for message_id in preview.matched
                }
                with transaction(self._conn):
                    for message_id in preview.matched:
                        set_message_label(
                            self._conn,
                            message_id,
                            MessageLabel.TRASH,
                            MessageLabelSource.HUMAN,
                            self.source_ref,
                            self._clock.now(),
                        )
                self._actions.append(_BulkAction(priors))
            saved_path = save_bulk_rule(self._scratch_dir, name, rule, self._clock.now())
            return preview, saved_path


def build_serve_app(
    conn: sqlite3.Connection,
    *,
    dir_: Path | None,
    new: bool,
    size: int,
    strategy: SiftStrategy,
    seed: int,
    mix: float,
    out_dir: Path | None,
    scratch_dir: Path,
    clock: Clock,
    exclude_channels: frozenset[str] = frozenset(),
    include_channels: frozenset[str] = frozenset(),
) -> ServeApp:
    batch = resolve_batch(
        conn,
        dir_=dir_,
        new=new,
        size=size,
        strategy=strategy,
        seed=seed,
        mix=mix,
        out_dir=out_dir,
        now=clock.now(),
        exclude_channels=exclude_channels,
        include_channels=include_channels,
    )
    messages = load_batch_messages(conn, batch.message_ids)
    return ServeApp(conn, messages, batch.dir.name, scratch_dir, clock)
