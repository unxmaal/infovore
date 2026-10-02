import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.db.messages_fts import search_messages

AT = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    return connection


def _message(conn: sqlite3.Connection, message_id: int, content: str) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, 1, 1, 1, 'author', ?, ?, ?, '{}')",
        (message_id, AT.isoformat(), content, AT.isoformat()),
    )


def test_a_message_is_findable_by_a_distinctive_term(conn: sqlite3.Connection) -> None:
    _message(conn, 1, "the Octane2 PSU is rated at 330W")
    _message(conn, 2, "unrelated chatter about lunch")

    assert [hit.message_id for hit in search_messages(conn, "Octane2")] == [1]


def test_a_version_string_is_one_token(conn: sqlite3.Connection) -> None:
    """`tokenize = unicode61 tokenchars '-./_'` is what makes this corpus
    searchable at all: without it `6.5.22m` splits into three useless
    numbers. The claims index already uses it; this mirrors it."""
    _message(conn, 1, "you need IRIX 6.5.22m for that")

    assert [hit.message_id for hit in search_messages(conn, "6.5.22m")] == [1]


def test_a_part_number_is_one_token(conn: sqlite3.Connection) -> None:
    _message(conn, 1, "the board is 030-1234-567 rev B")

    assert [hit.message_id for hit in search_messages(conn, "030-1234-567")] == [1]


def test_a_path_is_one_token(conn: sqlite3.Connection) -> None:
    _message(conn, 1, "it lives under /usr/people/eric")

    assert [hit.message_id for hit in search_messages(conn, "/usr/people/eric")] == [1]


def test_redacting_a_message_removes_it_from_the_index(conn: sqlite3.Connection) -> None:
    """THE PRIVACY SEAM. `privacy.optout.redact_stored` overwrites
    `messages.content` with '[redacted]'. An external-content FTS5 index keeps
    its own tokens, so without an update trigger the old words stay searchable
    and a query could prove what a redacted message said."""
    _message(conn, 1, "my serial number is ABCDEF123456")
    assert [hit.message_id for hit in search_messages(conn, "ABCDEF123456")] == [1]

    conn.execute("UPDATE messages SET content = '[redacted]' WHERE id = 1")

    assert search_messages(conn, "ABCDEF123456") == []


def test_an_opt_out_sync_leaves_nothing_searchable(conn: sqlite3.Connection) -> None:
    """End to end against the real redaction path rather than a hand-written
    UPDATE, because the trigger has to fire for the code that actually runs."""
    from infovore.privacy.optout import redact_stored

    _message(conn, 1, "contact me at averydistinctivehandle")
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (1, ?)", (AT.isoformat(),))

    redact_stored(conn, frozenset({1}))

    assert search_messages(conn, "averydistinctivehandle") == []


def test_deleting_a_message_removes_it_from_the_index(conn: sqlite3.Connection) -> None:
    _message(conn, 1, "transient Zephyrus content")
    conn.execute("DELETE FROM messages WHERE id = 1")

    assert search_messages(conn, "Zephyrus") == []


def test_the_index_is_backfilled_for_rows_that_predate_it(tmp_path: Path) -> None:
    """The 1,381,473 existing messages were written long before this index, so
    a migration that only installs triggers indexes nothing."""
    from infovore.db.connection import load_migrations

    connection = open_database(tmp_path / "b.db")
    all_migrations = load_migrations()
    migrate(connection, [m for m in all_migrations if m.version <= 20])
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    _message(connection, 1, "an Indigo2 Impact with a Galileo board")

    migrate(connection, [m for m in all_migrations if m.version == 21])

    assert [hit.message_id for hit in search_messages(connection, "Galileo")] == [1]


def test_search_reports_the_channel_and_exchange_for_each_hit(conn: sqlite3.Connection) -> None:
    """A hit nobody can place in a conversation is not useful: the point of
    this index is recovering a fact extraction missed, which means getting
    back to the thread."""
    _message(conn, 1, "the Tezro front panel button can be disabled")
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (9, 1, 1, 1, ?, ?, 1, 'quiet_gap', 'h9')",
        (AT.isoformat(), AT.isoformat()),
    )
    conn.execute(
        "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (9, 1, 1)"
    )

    hit = search_messages(conn, "Tezro")[0]

    assert hit.channel_name == "general"
    assert hit.exchange_id == 9


def test_an_ungrouped_message_is_still_findable(conn: sqlite3.Connection) -> None:
    """54% of messages belong to no exchange the gate ever admitted. Those are
    precisely the ones this index exists to reach."""
    _message(conn, 1, "a stray Crimson reference")

    hit = search_messages(conn, "Crimson")[0]

    assert hit.exchange_id is None


def test_a_query_matching_nothing_returns_nothing(conn: sqlite3.Connection) -> None:
    _message(conn, 1, "ordinary text")

    assert search_messages(conn, "Nonexistentterm") == []


def test_the_result_limit_is_honoured(conn: sqlite3.Connection) -> None:
    for message_id in range(1, 11):
        _message(conn, message_id, "repeated Onyx mention")

    assert len(search_messages(conn, "Onyx", limit=3)) == 3


def test_a_query_cannot_escape_its_own_quoting(conn: sqlite3.Connection) -> None:
    """FTS5 query syntax is a language, so an unescaped term is an injection
    surface as well as a crash."""
    _message(conn, 1, "harmless text")

    assert search_messages(conn, 'oct" OR messages_fts MATCH "a') == []
    assert search_messages(conn, '" OR "') == []


def test_multiple_terms_still_mean_and(conn: sqlite3.Connection) -> None:
    _message(conn, 1, "the Octane2 takes V12 graphics")
    _message(conn, 2, "the Octane2 alone")

    assert [hit.message_id for hit in search_messages(conn, "Octane2 V12")] == [1]


def test_an_empty_query_matches_nothing_rather_than_raising(conn: sqlite3.Connection) -> None:
    _message(conn, 1, "anything")

    assert search_messages(conn, "   ") == []


def test_a_trailing_tokenchar_does_not_zero_the_search(conn: sqlite3.Connection) -> None:
    """`.`, `-`, `/` and `_` are tokenchars, so a term ending in one produces a
    token that matches nothing, and FTS5 ANDs terms: one such term zeroes the
    whole result set. A trailing full stop then reads as "the archive does not
    contain this", which is the worst failure for an index whose purpose is
    proving an extraction miss is still recoverable (issue #179)."""
    _message(conn, 1, "the Octane2 PSU is rated at 330W")

    assert [hit.message_id for hit in search_messages(conn, "Octane2.")] == [1]
    assert [hit.message_id for hit in search_messages(conn, "the Octane2 PSU.")] == [1]
    assert [hit.message_id for hit in search_messages(conn, "Octane2 -")] == [1]


def test_a_trailing_underscore_does_not_zero_the_search(conn: sqlite3.Connection) -> None:
    """`_` is a tokenchar too, and the strip set that predates this was the
    hand-written literal `-./`, which misses it."""
    _message(conn, 1, "the irix_build is fine")

    assert [hit.message_id for hit in search_messages(conn, "irix_build_")] == [1]


def test_an_interior_tokenchar_is_still_significant(conn: sqlite3.Connection) -> None:
    """The negative control: stripping must only apply at the END of a term.
    If it stripped everywhere, `6.5.22m` would collapse to `6522m` and the
    tokenchars would have been pointless."""
    _message(conn, 1, "you need IRIX 6.5.22m for that")

    assert search_messages(conn, "6522m") == []
    assert [hit.message_id for hit in search_messages(conn, "6.5.22m.")] == [1]


def test_a_term_that_is_only_tokenchars_is_dropped_not_matched(conn: sqlite3.Connection) -> None:
    _message(conn, 1, "ordinary text here")

    assert [hit.message_id for hit in search_messages(conn, "ordinary -")] == [1]
    assert search_messages(conn, "-") == []
