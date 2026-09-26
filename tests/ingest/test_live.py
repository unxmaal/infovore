import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.db.claims import NewClaim, record_run
from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import get_exchange, insert_exchange, set_status
from infovore.db.raw import get_message, message_revisions, reactions_for_messages
from infovore.ingest.live import ConsumeReport, EventOutcome, consume, handle_event
from infovore.rows import (
    ClaimKind,
    ExchangeRow,
    ExtractionRunRow,
    ExtractionStatus,
    GroupingRule,
    RunMode,
    RunOutcome,
)
from infovore.source.fake import FakeDiscordSource
from infovore.source.protocol import (
    MessageCreated,
    MessageDeleted,
    MessageEdited,
    ReactionChanged,
    SourceAttachment,
    SourceChannel,
    SourceMessage,
    SourceReaction,
    ThreadCreated,
)
from infovore.rows import ChannelKind
from infovore.timing import FixedClock

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "live.db")
    migrate(connection)
    return connection


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(NOW)


def make_message(**overrides: object) -> SourceMessage:
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


async def test_handle_message_created_upserts_message(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    outcome = await handle_event(conn, MessageCreated(make_message()), clock, include_bots=False)
    assert outcome is EventOutcome.MESSAGE_CREATED
    row = get_message(conn, 1)
    assert row is not None
    assert row.content == "hello"


async def test_handle_message_created_with_attachments_and_reactions(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    message = make_message(
        attachments=(
            SourceAttachment(
                id=1, filename="a.png", content_type="image/png", size=10, url="http://x/a.png"
            ),
        ),
        reactions=(SourceReaction(emoji="👍", count=2),),
    )
    outcome = await handle_event(conn, MessageCreated(message), clock, include_bots=False)
    assert outcome is EventOutcome.MESSAGE_CREATED
    reactions = reactions_for_messages(conn, [1])
    assert len(reactions) == 1
    assert reactions[0].count == 2


async def test_handle_message_created_skips_bot_message(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    message = make_message(author_is_bot=True)
    outcome = await handle_event(conn, MessageCreated(message), clock, include_bots=False)
    assert outcome is EventOutcome.MESSAGE_SKIPPED
    assert get_message(conn, 1) is None


async def test_handle_message_edited_known_message_marks_revision(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    await handle_event(conn, MessageCreated(make_message()), clock, include_bots=False)
    edited = make_message(content="edited", edited_at=NOW + timedelta(minutes=1))
    outcome = await handle_event(conn, MessageEdited(edited), clock, include_bots=False)
    assert outcome is EventOutcome.MESSAGE_EDITED
    row = get_message(conn, 1)
    assert row is not None
    assert row.content == "edited"
    revisions = message_revisions(conn, 1)
    assert len(revisions) == 1
    assert revisions[0].content == "hello"


async def test_handle_message_edited_skips_bot_message(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    message = make_message(author_is_bot=True)
    outcome = await handle_event(conn, MessageEdited(message), clock, include_bots=False)
    assert outcome is EventOutcome.MESSAGE_SKIPPED
    assert get_message(conn, 1) is None


async def test_handle_message_edited_unknown_message_treated_as_create(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    outcome = await handle_event(conn, MessageEdited(make_message()), clock, include_bots=False)
    assert outcome is EventOutcome.MESSAGE_CREATED
    row = get_message(conn, 1)
    assert row is not None
    assert row.content == "hello"


async def test_handle_message_deleted_unknown_message_is_noop(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    outcome = await handle_event(
        conn, MessageDeleted(message_id=999, channel_id=10), clock, include_bots=False
    )
    assert outcome is EventOutcome.MESSAGE_DELETED
    assert get_message(conn, 999) is None


async def test_handle_message_deleted_marks_deleted_at(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    await handle_event(conn, MessageCreated(make_message()), clock, include_bots=False)
    outcome = await handle_event(
        conn, MessageDeleted(message_id=1, channel_id=10), clock, include_bots=False
    )
    assert outcome is EventOutcome.MESSAGE_DELETED
    row = get_message(conn, 1)
    assert row is not None
    assert row.deleted_at == NOW


async def test_handle_reaction_changed_known_message(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    await handle_event(conn, MessageCreated(make_message()), clock, include_bots=False)
    outcome = await handle_event(
        conn, ReactionChanged(message_id=1, emoji="👍", count=3), clock, include_bots=False
    )
    assert outcome is EventOutcome.REACTION_UPDATED
    reactions = reactions_for_messages(conn, [1])
    assert reactions[0].count == 3


async def test_handle_reaction_changed_unknown_message_is_noop(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    outcome = await handle_event(
        conn, ReactionChanged(message_id=999, emoji="👍", count=3), clock, include_bots=False
    )
    assert outcome is EventOutcome.REACTION_SKIPPED
    assert reactions_for_messages(conn, [999]) == []


async def test_handle_thread_created_upserts_channel(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    channel = SourceChannel(
        id=42, guild_id=100, parent_id=10, name="side-quest", kind=ChannelKind.THREAD, archived=False
    )
    outcome = await handle_event(conn, ThreadCreated(channel), clock, include_bots=False)
    assert outcome is EventOutcome.THREAD_CREATED
    row = conn.execute("SELECT name FROM channels WHERE id = 42").fetchone()
    assert row["name"] == "side-quest"


async def test_handle_event_unknown_event_type_raises(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    with pytest.raises(TypeError):
        await handle_event(conn, object(), clock, include_bots=False)  # type: ignore[arg-type]


def _make_exchange(message_ids: list[int], content_hash: str) -> ExchangeRow:
    return ExchangeRow(
        id=None,
        channel_id=10,
        thread_id=None,
        first_message_id=message_ids[0],
        last_message_id=message_ids[-1],
        started_at=NOW,
        ended_at=NOW,
        message_count=len(message_ids),
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=content_hash,
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )


async def test_edit_of_message_in_done_exchange_marks_it_stale(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    await handle_event(conn, MessageCreated(make_message(id=1)), clock, include_bots=False)
    exchange_id = insert_exchange(conn, _make_exchange([1], "hash-done"), [1])
    set_status(conn, exchange_id, ExtractionStatus.DONE)
    edited = make_message(id=1, content="edited")
    await handle_event(conn, MessageEdited(edited), clock, include_bots=False)
    exchange = get_exchange(conn, exchange_id)
    assert exchange is not None
    assert exchange.extraction_status is ExtractionStatus.STALE


async def test_edit_of_message_in_pending_exchange_stays_pending(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    await handle_event(conn, MessageCreated(make_message(id=1)), clock, include_bots=False)
    exchange_id = insert_exchange(conn, _make_exchange([1], "hash-pending"), [1])
    edited = make_message(id=1, content="edited")
    await handle_event(conn, MessageEdited(edited), clock, include_bots=False)
    exchange = get_exchange(conn, exchange_id)
    assert exchange is not None
    assert exchange.extraction_status is ExtractionStatus.PENDING


def _record_claim_with_sources(
    conn: sqlite3.Connection, exchange_id: int, source_message_ids: tuple[int, ...]
) -> int:
    run = ExtractionRunRow(
        id=None,
        exchange_id=exchange_id,
        model="fake",
        prompt_version="v1",
        started_at=NOW,
        finished_at=NOW,
        input_tokens=1,
        output_tokens=1,
        mode=RunMode.LIVE,
        outcome=RunOutcome.OK,
        error=None,
    )
    claim = NewClaim(
        exchange_id=exchange_id,
        statement="stmt",
        subject="subj",
        kind=ClaimKind.FACT,
        confidence=0.5,
        probe_question="q?",
        permalink="http://x",
        supersedes_claim_id=None,
        source_message_ids=source_message_ids,
    )
    recorded = record_run(conn, run, [claim])
    return recorded.claim_ids[0]


async def test_delete_of_only_source_retracts_claim(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    await handle_event(conn, MessageCreated(make_message(id=1)), clock, include_bots=False)
    exchange_id = insert_exchange(conn, _make_exchange([1], "hash-single-source"), [1])
    claim_id = _record_claim_with_sources(conn, exchange_id, (1,))
    await handle_event(
        conn, MessageDeleted(message_id=1, channel_id=10), clock, include_bots=False
    )
    row = conn.execute(
        "SELECT retracted_at, retraction_reason FROM claims WHERE id = ?", (claim_id,)
    ).fetchone()
    assert row["retracted_at"] is not None
    assert row["retraction_reason"] == "sources_deleted"


async def test_delete_of_one_of_two_sources_does_not_retract(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    await handle_event(conn, MessageCreated(make_message(id=1)), clock, include_bots=False)
    await handle_event(conn, MessageCreated(make_message(id=2)), clock, include_bots=False)
    exchange_id = insert_exchange(conn, _make_exchange([1, 2], "hash-two-sources"), [1, 2])
    claim_id = _record_claim_with_sources(conn, exchange_id, (1, 2))
    await handle_event(
        conn, MessageDeleted(message_id=1, channel_id=10), clock, include_bots=False
    )
    row = conn.execute(
        "SELECT retracted_at FROM claims WHERE id = ?", (claim_id,)
    ).fetchone()
    assert row["retracted_at"] is None


async def test_replay_of_event_sequence_is_idempotent(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    events = [
        MessageCreated(make_message(id=1)),
        MessageEdited(make_message(id=1, content="edited")),
        ReactionChanged(message_id=1, emoji="👍", count=2),
    ]
    for event in events:
        await handle_event(conn, event, clock, include_bots=False)
    first_message = get_message(conn, 1)
    first_reactions = reactions_for_messages(conn, [1])
    first_revisions = message_revisions(conn, 1)
    for event in events:
        await handle_event(conn, event, clock, include_bots=False)
    second_message = get_message(conn, 1)
    second_reactions = reactions_for_messages(conn, [1])
    second_revisions = message_revisions(conn, 1)
    assert first_message == second_message
    assert first_reactions == second_reactions
    assert first_revisions == second_revisions


async def test_consume_processes_all_events_from_source(
    conn: sqlite3.Connection, clock: FixedClock
) -> None:
    source = FakeDiscordSource()
    source.push(MessageCreated(make_message(id=1)))
    source.push(MessageCreated(make_message(id=2)))
    source.close()
    report = await consume(conn, source, clock, include_bots=False)
    assert report.processed == 2
    assert report.failed == 0
    assert isinstance(report, ConsumeReport)
    assert get_message(conn, 1) is not None
    assert get_message(conn, 2) is not None


async def test_consume_counts_failure_and_continues(
    conn: sqlite3.Connection, clock: FixedClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    import infovore.ingest.live as live_module

    original = live_module.upsert_message
    calls = {"n": 0}

    def flaky(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("boom")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(live_module, "upsert_message", flaky)
    source = FakeDiscordSource()
    source.push(MessageCreated(make_message(id=1)))
    source.push(MessageCreated(make_message(id=2)))
    source.close()
    report = await consume(conn, source, clock, include_bots=False)
    assert report.processed == 1
    assert report.failed == 1
    assert get_message(conn, 1) is None
    assert get_message(conn, 2) is not None
