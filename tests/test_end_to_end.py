import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from infovore.chunk.grouper import group_pending
from infovore.db.claims import promote_prompt_version, register_prompt_version
from infovore.db.connection import migrate, open_database
from infovore.extract.fake import MarkerExtractor, MarkerProbe
from infovore.extract.novelty import run_probe
from infovore.extract.prompt import PROMPT_SHA256, PROMPT_VERSION
from infovore.extract.runner import run_extraction
from infovore.ingest.backfill import backfill
from infovore.privacy.optout import REDACTED_CONTENT, sync_opt_outs
from infovore.rows import ChannelKind, RunMode
from infovore.source.fake import FakeDiscordSource
from infovore.source.protocol import SourceChannel, SourceMessage
from infovore.timing import FixedClock, RecordingSleeper

GUILD = 9
CHANNEL = 1
START = datetime(2026, 1, 1, tzinfo=UTC)
ALICE, BOB, CAROL, DAVE = 101, 102, 103, 104


def message(
    message_id: int,
    author_id: int,
    minutes: int,
    content: str,
    reply_to_id: int | None = None,
) -> SourceMessage:
    return SourceMessage(
        id=message_id,
        channel_id=CHANNEL,
        guild_id=GUILD,
        author_id=author_id,
        author_name=f"user{author_id}",
        author_is_bot=False,
        is_system=False,
        created_at=START + timedelta(minutes=minutes),
        edited_at=None,
        content=content,
        reply_to_id=reply_to_id,
        thread_id=None,
        attachments=(),
        reactions=(),
        raw={"id": str(message_id)},
    )


def fixture_source() -> FakeDiscordSource:
    return FakeDiscordSource(
        channels=[SourceChannel(CHANNEL, GUILD, None, "hardware", ChannelKind.TEXT, False)],
        messages=[
            message(1, ALICE, 0, "FACT: Octane2 PSU :: Octane2 PSU part is 060-0035-003 [unknown]"),
            message(2, BOB, 1, "thanks, that's useful"),
            message(3, CAROL, 2, "FACT: Carol secret :: something carol never wants kept"),
            message(4, ALICE, 3, "FACT: IRIX 6.5.22 :: 6.5.22 was the final IRIX release [known]"),
            message(
                5,
                DAVE,
                4,
                "CORRECTION: Octane2 PSU :: the V12 board needs the 060-0035-004 PSU [contradicts]",
                reply_to_id=1,
            ),
            message(6, ALICE, 120, "PROCEDURE: Octane PROM reset :: hold the NMI button [partial]"),
        ],
        role_members={GUILD: {"no-archive": [CAROL]}},
    )


async def run_pipeline(conn: sqlite3.Connection, source: FakeDiscordSource) -> None:
    clock = FixedClock(START + timedelta(hours=5))
    sleeper = RecordingSleeper(clock)
    await sync_opt_outs(conn, source, GUILD, "no-archive", clock)
    await backfill(conn, source, GUILD, (CHANNEL,), clock, sleeper, include_bots=False)
    group_pending(conn, clock)
    register_prompt_version(conn, PROMPT_VERSION, PROMPT_SHA256, clock.now())
    promote_prompt_version(conn, PROMPT_VERSION, clock.now())
    await run_extraction(
        conn,
        MarkerExtractor(),
        clock,
        sleeper,
        mode=RunMode.LIVE,
        model_label="fake-marker",
        batch_size=10,
        max_retries=3,
        concurrency=2,
    )
    await run_probe(conn, MarkerProbe(), clock, sleeper, probe_model=None, limit=10, concurrency=2)


def lore(conn: sqlite3.Connection) -> list[tuple[str, str, str, str]]:
    return [
        (row["subject"], row["kind"], row["novelty"], row["source_message_ids"])
        for row in conn.execute(
            "SELECT subject, kind, novelty, source_message_ids FROM lore ORDER BY claim_id"
        )
    ]


async def test_backfill_to_lore_with_fakes_only(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "infovore.db")
    migrate(conn)
    source = fixture_source()

    await run_pipeline(conn, source)

    assert lore(conn) == [
        ("Octane2 PSU", "fact", "unknown", "1"),
        ("Octane2 PSU", "correction", "contradicts", "5"),
        ("Octane PROM reset", "procedure", "partial", "6"),
    ]
    known = conn.execute("SELECT novelty FROM claims WHERE subject = 'IRIX 6.5.22'").fetchone()
    assert known["novelty"] == "known"
    assert (
        conn.execute("SELECT COUNT(*) FROM claims WHERE subject LIKE 'Carol%'").fetchone()[0] == 0
    )
    carol = conn.execute(
        "SELECT content, author_name_at_time FROM messages WHERE id = 3"
    ).fetchone()
    assert carol["content"] == REDACTED_CONTENT
    assert carol["author_name_at_time"] != "user103"
    statuses = [row[0] for row in conn.execute("SELECT extraction_status FROM exchanges")]
    assert statuses == ["done", "done", "done"]


async def test_rerunning_the_whole_pipeline_changes_nothing(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "infovore.db")
    migrate(conn)
    source = fixture_source()
    await run_pipeline(conn, source)
    before = (
        lore(conn),
        conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM message_revisions").fetchone()[0],
    )

    await run_pipeline(conn, source)

    after = (
        lore(conn),
        conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM message_revisions").fetchone()[0],
    )
    assert after == before
