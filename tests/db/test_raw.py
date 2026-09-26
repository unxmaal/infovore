import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.raw import (
    get_backfill_checkpoint,
    get_channel,
    set_backfill_checkpoint,
    upsert_channel,
)
from infovore.rows import ChannelKind, ChannelRow

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "test.db")
    migrate(connection)
    return connection


def make_channel(
    channel_id: int = 1,
    guild_id: int = 1,
    parent_id: int | None = None,
    name: str = "general",
    kind: ChannelKind = ChannelKind.TEXT,
    archived: bool = False,
    last_backfilled_message_id: int | None = None,
) -> ChannelRow:
    return ChannelRow(
        id=channel_id,
        guild_id=guild_id,
        parent_id=parent_id,
        name=name,
        kind=kind,
        archived=archived,
        last_backfilled_message_id=last_backfilled_message_id,
    )


def test_get_channel_returns_none_when_missing(conn: sqlite3.Connection) -> None:
    assert get_channel(conn, 1) is None


def test_upsert_channel_inserts_then_get_channel_returns_it(conn: sqlite3.Connection) -> None:
    upsert_channel(conn, make_channel())
    assert get_channel(conn, 1) == make_channel()


def test_upsert_channel_inserts_thread_with_parent(conn: sqlite3.Connection) -> None:
    upsert_channel(conn, make_channel(channel_id=2, parent_id=1, kind=ChannelKind.THREAD))
    fetched = get_channel(conn, 2)
    assert fetched is not None
    assert fetched.parent_id == 1
    assert fetched.kind is ChannelKind.THREAD


def test_upsert_channel_updates_mutable_fields_but_not_checkpoint(
    conn: sqlite3.Connection,
) -> None:
    upsert_channel(conn, make_channel(name="general", archived=False))
    set_backfill_checkpoint(conn, 1, 42)
    upsert_channel(conn, make_channel(name="renamed", archived=True))
    fetched = get_channel(conn, 1)
    assert fetched is not None
    assert fetched.name == "renamed"
    assert fetched.archived is True
    assert fetched.last_backfilled_message_id == 42


def test_get_backfill_checkpoint_returns_none_for_unknown_channel(
    conn: sqlite3.Connection,
) -> None:
    assert get_backfill_checkpoint(conn, 1) is None


def test_get_backfill_checkpoint_returns_none_when_not_set(conn: sqlite3.Connection) -> None:
    upsert_channel(conn, make_channel())
    assert get_backfill_checkpoint(conn, 1) is None


def test_set_backfill_checkpoint_then_get_returns_it(conn: sqlite3.Connection) -> None:
    upsert_channel(conn, make_channel())
    set_backfill_checkpoint(conn, 1, 99)
    assert get_backfill_checkpoint(conn, 1) == 99
