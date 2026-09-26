import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.db.claims import NewClaim, get_claim, record_run
from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database
from infovore.db.raw import (
    attachments_for_messages,
    get_message,
    message_revisions,
    upsert_attachment,
    upsert_message,
)
from infovore.ingest.normalize import normalize_message
from infovore.privacy.optout import (
    REDACTED_AUTHOR,
    REDACTED_CONTENT,
    SyncReport,
    opted_out_user_ids,
    redact_normalized,
    redact_stored,
    sync_opt_outs,
)
from infovore.rows import AttachmentRow, ClaimKind, ExtractionRunRow, RunMode, RunOutcome
from infovore.source.fake import FakeDiscordSource
from infovore.source.protocol import SourceAttachment, SourceMessage, SourceReaction
from infovore.timing import FixedClock

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def make_source_message(**overrides: object) -> SourceMessage:
    fields: dict[str, object] = {
        "id": 1,
        "channel_id": 10,
        "guild_id": 100,
        "author_id": 5,
        "author_name": "alice",
        "author_is_bot": False,
        "is_system": False,
        "created_at": NOW,
        "edited_at": None,
        "content": "hello",
        "reply_to_id": None,
        "thread_id": None,
        "attachments": (),
        "reactions": (),
        "raw": {"id": "1"},
    }
    fields.update(overrides)
    return SourceMessage(**fields)  # type: ignore[arg-type]


def insert_message(
    conn: sqlite3.Connection,
    message_id: int,
    author_id: int = 1,
    content: str = "hi",
    author_name: str = "alice",
    raw_json: str = '{"a":1}',
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, 1, 1, ?, ?, ?, ?, ?, ?)",
        (message_id, author_id, author_name, to_db_time(NOW), content, to_db_time(NOW), raw_json),
    )


def insert_exchange(conn: sqlite3.Connection, exchange_id: int = 1, message_id: int = 1) -> None:
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, 1, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
        (exchange_id, message_id, message_id, to_db_time(NOW), to_db_time(NOW), f"h{exchange_id}"),
    )


def a_run(exchange_id: int = 1) -> ExtractionRunRow:
    return ExtractionRunRow(
        id=None,
        exchange_id=exchange_id,
        model="m",
        prompt_version="v1",
        started_at=NOW,
        finished_at=NOW,
        input_tokens=1,
        output_tokens=1,
        mode=RunMode.LIVE,
        outcome=RunOutcome.OK,
        error=None,
    )


def a_claim(exchange_id: int = 1, source_message_ids: tuple[int, ...] = (1,)) -> NewClaim:
    return NewClaim(
        exchange_id=exchange_id,
        statement="Octane2 needs PROM 6.5",
        subject="IP30",
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question="what prom does the octane2 need?",
        permalink="https://discord.com/channels/1/1/1",
        supersedes_claim_id=None,
        source_message_ids=source_message_ids,
    )


def test_opted_out_user_ids_empty(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert opted_out_user_ids(conn) == frozenset()


def test_opted_out_user_ids_returns_all_rows(tmp_path: Path) -> None:
    conn = db(tmp_path)
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (?, ?)", (1, to_db_time(NOW)))
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (?, ?)", (2, to_db_time(NOW)))
    assert opted_out_user_ids(conn) == frozenset({1, 2})


def test_redact_normalized_leaves_non_opted_out_untouched() -> None:
    message = make_source_message(author_id=5)
    normalized = normalize_message(message, NOW, include_bots=False)
    assert normalized is not None
    assert redact_normalized(normalized, frozenset({999})) == normalized


def test_redact_normalized_redacts_content_author_and_raw_json_drops_attachments() -> None:
    attachment = SourceAttachment(id=1, filename="f", content_type=None, size=1, url="u")
    reaction = SourceReaction(emoji="thumbsup", count=2)
    message = make_source_message(author_id=5, attachments=(attachment,), reactions=(reaction,))
    normalized = normalize_message(message, NOW, include_bots=False)
    assert normalized is not None

    result = redact_normalized(normalized, frozenset({5}))

    assert result.message.id == normalized.message.id
    assert result.message.content == REDACTED_CONTENT
    assert result.message.author_name_at_time == REDACTED_AUTHOR
    assert result.message.raw_json == "{}"
    assert result.attachments == ()
    assert result.reactions == normalized.reactions


def test_redact_stored_redacts_messages_revisions_and_drops_attachments(tmp_path: Path) -> None:
    conn = db(tmp_path)
    insert_message(conn, 1, author_id=5, content="hello")
    conn.execute(
        "INSERT INTO message_revisions (message_id, revision, content, edited_at, raw_json)"
        " VALUES (1, 0, 'original', ?, '{\"a\":1}')",
        (to_db_time(NOW),),
    )
    upsert_attachment(
        conn,
        AttachmentRow(
            id=1,
            message_id=1,
            filename="f",
            content_type=None,
            size=1,
            url="u",
            sha256=None,
            local_path=None,
        ),
    )
    insert_message(conn, 2, author_id=9, content="other")

    count = redact_stored(conn, frozenset({5}))

    assert count == 1
    message = get_message(conn, 1)
    assert message is not None
    assert message.content == REDACTED_CONTENT
    assert message.author_name_at_time == REDACTED_AUTHOR
    assert message.raw_json == "{}"

    revisions = message_revisions(conn, 1)
    assert len(revisions) == 1
    assert revisions[0].content == REDACTED_CONTENT
    assert revisions[0].raw_json == "{}"

    assert attachments_for_messages(conn, [1]) == []

    other = get_message(conn, 2)
    assert other is not None
    assert other.content == "other"


def test_redact_stored_handles_multiple_messages_same_author(tmp_path: Path) -> None:
    conn = db(tmp_path)
    insert_message(conn, 1, author_id=5, content="a")
    insert_message(conn, 2, author_id=5, content="b")

    count = redact_stored(conn, frozenset({5}))

    assert count == 2
    first = get_message(conn, 1)
    second = get_message(conn, 2)
    assert first is not None and first.content == REDACTED_CONTENT
    assert second is not None and second.content == REDACTED_CONTENT


def test_redact_stored_is_idempotent_and_writes_no_revision(tmp_path: Path) -> None:
    conn = db(tmp_path)
    insert_message(conn, 1, author_id=5, content="hello")

    redact_stored(conn, frozenset({5}))
    after_first = message_revisions(conn, 1)
    redact_stored(conn, frozenset({5}))
    after_second = message_revisions(conn, 1)

    assert after_first == after_second == []
    message = get_message(conn, 1)
    assert message is not None
    assert message.content == REDACTED_CONTENT


def test_redact_stored_with_empty_user_ids_is_noop(tmp_path: Path) -> None:
    conn = db(tmp_path)
    insert_message(conn, 1, author_id=5, content="hello")

    assert redact_stored(conn, frozenset()) == 0
    message = get_message(conn, 1)
    assert message is not None
    assert message.content == "hello"


def test_backfill_rerun_after_opt_out_never_unredacts_row_or_revisions(tmp_path: Path) -> None:
    conn = db(tmp_path)
    source_message = make_source_message(author_id=5, content="original secret")

    first_pass = normalize_message(source_message, NOW, include_bots=False)
    assert first_pass is not None
    upsert_message(conn, redact_normalized(first_pass, frozenset({5})).message)

    edited = make_source_message(
        author_id=5, content="edited secret", edited_at=NOW + timedelta(minutes=1)
    )
    edited_normalized = normalize_message(edited, NOW, include_bots=False)
    assert edited_normalized is not None
    upsert_message(conn, redact_normalized(edited_normalized, frozenset({5})).message)

    redact_stored(conn, frozenset({5}))

    rerun_normalized = normalize_message(source_message, NOW, include_bots=False)
    assert rerun_normalized is not None
    outcome = upsert_message(conn, redact_normalized(rerun_normalized, frozenset({5})).message)

    message = get_message(conn, source_message.id)
    assert message is not None
    assert message.content == REDACTED_CONTENT
    assert message.author_name_at_time == REDACTED_AUTHOR
    revisions = message_revisions(conn, source_message.id)
    assert len(revisions) == 1
    assert revisions[0].content == REDACTED_CONTENT
    assert revisions[0].raw_json == "{}"
    from infovore.db.raw import UpsertOutcome

    assert outcome == UpsertOutcome.UNCHANGED


async def test_sync_opt_outs_adds_removes_redacts_and_retracts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    conn = db(tmp_path)
    stale_since = NOW - timedelta(days=30)
    conn.execute(
        "INSERT INTO opt_outs (user_id, since) VALUES (?, ?)", (7, to_db_time(stale_since))
    )
    insert_message(conn, 1, author_id=42, content="new opt out message")
    insert_message(conn, 2, author_id=3, content="other author")
    insert_exchange(conn, 1, 1)
    insert_exchange(conn, 2, 2)
    all_opted_out = record_run(conn, a_run(1), [a_claim(1, source_message_ids=(1,))])
    mixed = record_run(conn, a_run(2), [a_claim(2, source_message_ids=(2,))])

    source = FakeDiscordSource(role_members={100: {"no-archive": [42, 7]}})
    clock = FixedClock(NOW)

    with caplog.at_level(logging.INFO, logger="infovore.privacy.optout"):
        result: SyncReport = await sync_opt_outs(conn, source, 100, "no-archive", clock)

    assert result.added == frozenset({42})
    assert result.removed == frozenset()
    assert result.redacted_messages == 1
    assert result.retracted_claims == (all_opted_out.claim_ids[0],)

    message = get_message(conn, 1)
    assert message is not None
    assert message.content == REDACTED_CONTENT

    claim = get_claim(conn, all_opted_out.claim_ids[0])
    assert claim is not None
    assert claim.retracted_at is not None

    mixed_claim = get_claim(conn, mixed.claim_ids[0])
    assert mixed_claim is not None
    assert mixed_claim.retracted_at is None

    assert opted_out_user_ids(conn) == frozenset({7, 42})
    row = conn.execute("SELECT since FROM opt_outs WHERE user_id = 42").fetchone()
    assert row["since"] == to_db_time(NOW)
    stale_row = conn.execute("SELECT since FROM opt_outs WHERE user_id = 7").fetchone()
    assert stale_row["since"] == to_db_time(stale_since)

    assert any("42" in record.message for record in caplog.records)
    assert not any("new opt out message" in record.message for record in caplog.records)


async def test_sync_opt_outs_removes_users_who_opted_back_in_without_restoring_history(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (?, ?)", (7, to_db_time(NOW)))
    insert_message(conn, 1, author_id=7, content="already redacted")
    redact_stored(conn, frozenset({7}))

    source = FakeDiscordSource(role_members={100: {"no-archive": []}})
    clock = FixedClock(NOW)

    result = await sync_opt_outs(conn, source, 100, "no-archive", clock)

    assert result.added == frozenset()
    assert result.removed == frozenset({7})
    assert result.redacted_messages == 0
    assert result.retracted_claims == ()
    assert opted_out_user_ids(conn) == frozenset()

    message = get_message(conn, 1)
    assert message is not None
    assert message.content == REDACTED_CONTENT


async def test_sync_opt_outs_with_no_changes_does_not_touch_opt_outs(tmp_path: Path) -> None:
    conn = db(tmp_path)
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (?, ?)", (7, to_db_time(NOW)))

    source = FakeDiscordSource(role_members={100: {"no-archive": [7]}})
    clock = FixedClock(NOW + timedelta(days=1))

    result = await sync_opt_outs(conn, source, 100, "no-archive", clock)

    assert result.added == frozenset()
    assert result.removed == frozenset()
    assert result.redacted_messages == 0
    assert result.retracted_claims == ()
    row = conn.execute("SELECT since FROM opt_outs WHERE user_id = 7").fetchone()
    assert row["since"] == to_db_time(NOW)
