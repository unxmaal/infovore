import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from infovore.db.connection import migrate, open_database
from infovore.db.raw import (
    attachments_for_messages,
    get_backfill_checkpoint,
    get_channel,
    get_message,
    reactions_for_messages,
)
from infovore.ingest.backfill import BackfillReport, backfill
from infovore.rows import ChannelKind
from infovore.source.fake import FakeDiscordSource
from infovore.source.protocol import (
    SourceAttachment,
    SourceChannel,
    SourceMessage,
    SourceRateLimitedError,
    SourceReaction,
    SourceUnavailableError,
)
from infovore.timing import FixedClock, RecordingSleeper

GUILD_ID = 100
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_conn(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "test.db")
    migrate(conn)
    return conn


def make_channel(
    channel_id: int,
    guild_id: int = GUILD_ID,
    parent_id: int | None = None,
    kind: ChannelKind = ChannelKind.TEXT,
    archived: bool = False,
) -> SourceChannel:
    return SourceChannel(
        id=channel_id,
        guild_id=guild_id,
        parent_id=parent_id,
        name=f"channel-{channel_id}",
        kind=kind,
        archived=archived,
    )


def make_message(
    msg_id: int,
    channel_id: int = 1,
    author_id: int = 5,
    author_is_bot: bool = False,
    is_system: bool = False,
    content: str = "hi",
    attachments: tuple[SourceAttachment, ...] = (),
    reactions: tuple[SourceReaction, ...] = (),
) -> SourceMessage:
    return SourceMessage(
        id=msg_id,
        channel_id=channel_id,
        guild_id=GUILD_ID,
        author_id=author_id,
        author_name="alice",
        author_is_bot=author_is_bot,
        is_system=is_system,
        created_at=NOW + timedelta(minutes=msg_id),
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        attachments=attachments,
        reactions=reactions,
        raw={},
    )


def dump_tables(conn: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
    tables = ["channels", "messages", "attachments", "reactions", "message_revisions"]
    return {
        table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2")]
        for table in tables
    }


async def test_full_walk_produces_expected_rows(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[make_channel(1)],
        messages=[make_message(i, channel_id=1) for i in range(1, 6)],
    )
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        channel_ids=[1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
        page_size=2,
    )
    assert isinstance(report, BackfillReport)
    assert report.failed == []
    channel_report = report.channels[1]
    assert channel_report.pages == 3
    assert channel_report.inserted == 5
    assert channel_report.updated == 0
    assert channel_report.unchanged == 0
    assert channel_report.skipped == 0
    assert get_channel(conn, 1) is not None
    for i in range(1, 6):
        message = get_message(conn, i)
        assert message is not None
        assert message.content == "hi"
    assert get_backfill_checkpoint(conn, 1) == 5


async def test_running_twice_changes_nothing(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[make_channel(1)],
        messages=[make_message(i, channel_id=1) for i in range(1, 4)],
    )
    args = (
        GUILD_ID,
        [1],
    )
    await backfill(
        conn, source, *args, clock=FixedClock(NOW), sleeper=RecordingSleeper(), include_bots=False
    )
    before = dump_tables(conn)
    second_report = await backfill(
        conn, source, *args, clock=FixedClock(NOW), sleeper=RecordingSleeper(), include_bots=False
    )
    after = dump_tables(conn)
    assert before == after
    assert second_report.channels[1].pages == 0
    assert second_report.channels[1].inserted == 0
    assert second_report.channels[1].updated == 0
    assert second_report.channels[1].unchanged == 0


async def test_interrupt_then_resume_matches_uninterrupted_walk(tmp_path: Path) -> None:
    baseline_conn = open_database(tmp_path / "baseline.db")
    migrate(baseline_conn)
    baseline_source = FakeDiscordSource(
        channels=[make_channel(1)],
        messages=[make_message(i, channel_id=1) for i in range(1, 6)],
    )
    await backfill(
        baseline_conn,
        baseline_source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
        page_size=2,
    )
    baseline = dump_tables(baseline_conn)

    conn = open_database(tmp_path / "resumed.db")
    migrate(conn)
    source = FakeDiscordSource(
        channels=[make_channel(1)],
        messages=[make_message(i, channel_id=1) for i in range(1, 6)],
    )
    source.interrupt_history_after(1, pages=1, error=SourceUnavailableError("boom"))
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
        page_size=2,
    )
    assert report.failed == []
    assert dump_tables(conn) == baseline


async def test_rate_limit_sleeps_retry_after_then_succeeds(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[make_channel(1)],
        messages=[make_message(1, channel_id=1)],
    )
    source.fail_next_history_call(SourceRateLimitedError(2.5), channel_id=1)
    sleeper = RecordingSleeper()
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=sleeper,
        include_bots=False,
    )
    assert sleeper.slept == [2.5]
    assert report.failed == []
    assert report.channels[1].inserted == 1


async def test_unavailable_backs_off_then_succeeds(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[make_channel(1)],
        messages=[make_message(1, channel_id=1)],
    )
    source.fail_next_history_call(SourceUnavailableError("down"), channel_id=1)
    source.fail_next_history_call(SourceUnavailableError("down"), channel_id=1)
    sleeper = RecordingSleeper()
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=sleeper,
        include_bots=False,
        max_attempts=5,
    )
    assert sleeper.slept == [1.0, 2.0]
    assert report.failed == []
    assert report.channels[1].inserted == 1


async def test_exhausting_attempts_fails_channel_but_others_complete(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[make_channel(1), make_channel(2)],
        messages=[make_message(1, channel_id=1), make_message(101, channel_id=2)],
    )
    source.fail_next_history_call(SourceUnavailableError("down"), channel_id=1)
    source.fail_next_history_call(SourceUnavailableError("down"), channel_id=1)
    sleeper = RecordingSleeper()
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1, 2],
        clock=FixedClock(NOW),
        sleeper=sleeper,
        include_bots=False,
        max_attempts=2,
    )
    assert len(report.failed) == 1
    assert report.failed[0].channel_id == 1
    assert "down" in report.failed[0].reason
    assert sleeper.slept == [1.0]
    assert report.channels[2].inserted == 1
    assert get_channel(conn, 1) is not None
    assert get_message(conn, 1) is None
    message_2 = get_message(conn, 101)
    assert message_2 is not None
    assert message_2.channel_id == 2


async def test_empty_channel_allowlist_selects_all_channels(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[
            make_channel(1),
            make_channel(2, parent_id=1, kind=ChannelKind.THREAD),
            make_channel(4),
        ],
        messages=[
            make_message(10, channel_id=1),
            make_message(11, channel_id=2),
            make_message(12, channel_id=4),
        ],
    )
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        channel_ids=(),
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    assert set(report.channels.keys()) == {1, 2, 4}
    assert get_message(conn, 10) is not None
    assert get_message(conn, 11) is not None
    assert get_message(conn, 12) is not None


async def test_threads_under_allowlisted_parent_included_others_excluded(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[
            make_channel(1),
            make_channel(2, parent_id=1, kind=ChannelKind.THREAD, archived=False),
            make_channel(3, parent_id=1, kind=ChannelKind.THREAD, archived=True),
            make_channel(4),
            make_channel(5, parent_id=4, kind=ChannelKind.THREAD),
        ],
        messages=[
            make_message(10, channel_id=2),
            make_message(11, channel_id=3),
            make_message(12, channel_id=4),
            make_message(13, channel_id=5),
        ],
    )
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    assert set(report.channels.keys()) == {1, 2, 3}
    assert get_channel(conn, 1) is not None
    assert get_channel(conn, 2) is not None
    assert get_channel(conn, 3) is not None
    assert get_channel(conn, 4) is None
    assert get_channel(conn, 5) is None
    assert get_message(conn, 10) is not None
    assert get_message(conn, 11) is not None
    assert get_message(conn, 12) is None
    assert get_message(conn, 13) is None


async def test_bot_and_system_messages_skipped_per_include_bots(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[make_channel(1)],
        messages=[
            make_message(1, channel_id=1),
            make_message(2, channel_id=1, author_is_bot=True),
            make_message(3, channel_id=1, is_system=True),
        ],
    )
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    assert report.channels[1].inserted == 1
    assert report.channels[1].skipped == 2
    assert get_message(conn, 1) is not None
    assert get_message(conn, 2) is None
    assert get_message(conn, 3) is None
    assert get_backfill_checkpoint(conn, 1) == 3


async def test_bot_messages_kept_when_include_bots_true(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[make_channel(1)],
        messages=[
            make_message(1, channel_id=1),
            make_message(2, channel_id=1, author_is_bot=True),
            make_message(3, channel_id=1, is_system=True),
        ],
    )
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=True,
    )
    assert report.channels[1].inserted == 2
    assert report.channels[1].skipped == 1
    assert get_message(conn, 2) is not None
    assert get_message(conn, 3) is None


async def test_no_channels_match_allowlist_produces_empty_report(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(channels=[make_channel(9)])
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    assert report.channels == {}
    assert report.failed == []


async def test_attachments_and_reactions_are_persisted(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    attachment = SourceAttachment(
        id=1, filename="a.png", content_type="image/png", size=10, url="https://x/a.png"
    )
    message = make_message(
        1,
        channel_id=1,
        attachments=(attachment,),
        reactions=(SourceReaction("👍", 3),),
    )
    source = FakeDiscordSource(channels=[make_channel(1)], messages=[message])
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    assert report.channels[1].inserted == 1
    attachments = attachments_for_messages(conn, [1])
    assert len(attachments) == 1
    assert attachments[0].filename == "a.png"
    reactions = reactions_for_messages(conn, [1])
    assert len(reactions) == 1
    assert reactions[0].emoji == "👍"
    assert reactions[0].count == 3


async def test_edited_message_reprocessed_after_checkpoint_reset_is_updated(
    tmp_path: Path,
) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[make_channel(1)], messages=[make_message(1, channel_id=1, content="original")]
    )
    await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    source.edit_message(make_message(1, channel_id=1, content="edited"))
    from infovore.db.raw import set_backfill_checkpoint

    set_backfill_checkpoint(conn, 1, 0)
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    assert report.channels[1].updated == 1
    message = get_message(conn, 1)
    assert message is not None
    assert message.content == "edited"


async def test_reprocessing_same_content_after_checkpoint_reset_is_unchanged(
    tmp_path: Path,
) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[make_channel(1)], messages=[make_message(1, channel_id=1, content="same")]
    )
    await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    from infovore.db.raw import set_backfill_checkpoint

    set_backfill_checkpoint(conn, 1, 0)
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    assert report.channels[1].unchanged == 1
    assert report.channels[1].inserted == 0


def opt_out_user(conn: sqlite3.Connection, user_id: int) -> None:
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (?, ?)", (user_id, NOW.isoformat()))


async def test_opted_out_author_messages_are_stored_redacted(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    opt_out_user(conn, 5)
    attachment = SourceAttachment(
        id=1, filename="a.png", content_type="image/png", size=10, url="https://x/a.png"
    )
    source = FakeDiscordSource(
        channels=[make_channel(1)],
        messages=[
            SourceMessage(
                id=1,
                channel_id=1,
                guild_id=GUILD_ID,
                author_id=5,
                author_name="alice",
                author_is_bot=False,
                is_system=False,
                created_at=NOW + timedelta(minutes=1),
                edited_at=None,
                content="secret plans",
                reply_to_id=None,
                thread_id=None,
                attachments=(attachment,),
                reactions=(SourceReaction("👍", 2),),
                raw={"content": "secret plans"},
            ),
            make_message(2, channel_id=1, author_id=6, content="not opted out"),
        ],
    )
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    assert report.channels[1].inserted == 2
    redacted = get_message(conn, 1)
    assert redacted is not None
    assert redacted.content == "[redacted]"
    assert redacted.author_name_at_time == "[redacted]"
    assert redacted.raw_json == "{}"
    assert attachments_for_messages(conn, [1]) == []
    assert reactions_for_messages(conn, [1])[0].count == 2
    kept = get_message(conn, 2)
    assert kept is not None
    assert kept.content == "not opted out"


async def test_rerun_never_unredacts_opted_out_messages(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    opt_out_user(conn, 5)
    source = FakeDiscordSource(
        channels=[make_channel(1)],
        messages=[
            SourceMessage(
                id=1,
                channel_id=1,
                guild_id=GUILD_ID,
                author_id=5,
                author_name="alice",
                author_is_bot=False,
                is_system=False,
                created_at=NOW + timedelta(minutes=1),
                edited_at=None,
                content="secret plans",
                reply_to_id=None,
                thread_id=None,
                attachments=(),
                reactions=(),
                raw={},
            ),
        ],
    )
    await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    from infovore.db.raw import set_backfill_checkpoint

    set_backfill_checkpoint(conn, 1, 0)
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=RecordingSleeper(),
        include_bots=False,
    )
    assert report.channels[1].unchanged == 1
    redacted = get_message(conn, 1)
    assert redacted is not None
    assert redacted.content == "[redacted]"


async def test_rate_limit_exhausting_attempts_fails_channel(tmp_path: Path) -> None:
    conn = make_conn(tmp_path)
    source = FakeDiscordSource(
        channels=[make_channel(1)],
        messages=[make_message(1, channel_id=1)],
    )
    source.fail_next_history_call(SourceRateLimitedError(1.0), channel_id=1)
    source.fail_next_history_call(SourceRateLimitedError(1.0), channel_id=1)
    sleeper = RecordingSleeper()
    report = await backfill(
        conn,
        source,
        GUILD_ID,
        [1],
        clock=FixedClock(NOW),
        sleeper=sleeper,
        include_bots=False,
        max_attempts=2,
    )
    assert len(report.failed) == 1
    assert "rate limited" in report.failed[0].reason
    assert sleeper.slept == [1.0]
    assert get_message(conn, 1) is None
