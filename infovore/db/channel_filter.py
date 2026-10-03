"""Shared SQL for filtering exchanges/messages by channel name (issue #138).

The maintainer's channel denylist (`INFOVORE_EXCLUDE_CHANNELS`) and `sift
export`/`sift serve --new`'s `--channels` include filter both name channels
by their human-readable name, not id, and both need a denylisted (or
allowlisted) channel to also cover every Discord thread hanging off it. A
thread is its own row in the `channels` table (`kind='thread'`), with
`parent_id` pointing at the text channel it belongs to, so "channel X is
excluded" must match either the row's own name or its parent's.

Both helpers below build a correlated `EXISTS`/`NOT EXISTS` clause against
whatever `channel_id_column` the caller's query already exposes (e.g.
`exchanges.channel_id`, or an aliased `m.channel_id`), rather than joining —
so they drop into an existing `WHERE` clause without changing the outer
query's row shape. An empty name set means "no filter": both return `("",
[])`, safe to always call.
"""

import sqlite3
from collections.abc import Iterable


def _match_clause(channel_id_column: str, names: Iterable[str]) -> tuple[str, list[str]]:
    ordered = sorted(names)
    placeholders = ", ".join("?" * len(ordered))
    clause = (
        "SELECT 1 FROM channels __cf_c LEFT JOIN channels __cf_p ON __cf_p.id = __cf_c.parent_id"
        f" WHERE __cf_c.id = {channel_id_column}"
        f" AND (LOWER(__cf_c.name) IN ({placeholders}) OR LOWER(__cf_p.name) IN ({placeholders}))"
    )
    return clause, [*ordered, *ordered]


def exclude_channels_clause(
    channel_id_column: str, exclude_channels: frozenset[str]
) -> tuple[str, list[str]]:
    """A ` AND NOT EXISTS (...)` clause excluding rows whose channel (or, for
    a thread, whose parent channel) name is in `exclude_channels`."""
    if not exclude_channels:
        return "", []
    inner, params = _match_clause(channel_id_column, exclude_channels)
    return f" AND NOT EXISTS ({inner})", params


def include_channels_clause(
    channel_id_column: str, include_channels: frozenset[str]
) -> tuple[str, list[str]]:
    """A ` AND EXISTS (...)` clause restricting to rows whose channel (or,
    for a thread, whose parent channel) name is in `include_channels`."""
    if not include_channels:
        return "", []
    inner, params = _match_clause(channel_id_column, include_channels)
    return f" AND EXISTS ({inner})", params


def excluded_exchange_ids(
    conn: sqlite3.Connection, exclude_channels: frozenset[str]
) -> frozenset[int]:
    """Ids of exchanges whose channel (or thread parent) is in `exclude_channels`."""
    if not exclude_channels:
        return frozenset()
    clause, params = include_channels_clause("e.channel_id", exclude_channels)
    rows = conn.execute("SELECT e.id FROM current_exchanges e WHERE 1 = 1" + clause, params)
    return frozenset(row[0] for row in rows)


def known_channel_names(conn: sqlite3.Connection) -> frozenset[str]:
    """Every distinct channel name (lowercased) in the `channels` table —
    what a `--channels`/`INFOVORE_EXCLUDE_CHANNELS` name is validated
    against."""
    rows = conn.execute("SELECT DISTINCT name FROM channels").fetchall()
    return frozenset(str(row[0]).lower() for row in rows)
