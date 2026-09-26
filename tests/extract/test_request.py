import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.claims import NewClaim, record_run, register_prompt_version
from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import insert_exchange
from infovore.db.raw import (
    set_reaction_count,
    upsert_attachment,
    upsert_channel,
    upsert_message,
)
from infovore.extract.protocol import ExtractionRequest
from infovore.extract.request import build_request
from infovore.rows import (
    AttachmentRow,
    ChannelKind,
    ChannelRow,
    ClaimKind,
    ExchangeRow,
    ExtractionRunRow,
    ExtractionStatus,
    GroupingRule,
    MessageRow,
    RunMode,
    RunOutcome,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def a_message(
    message_id: int,
    channel_id: int = 1,
    author_id: int = 1,
    author_name: str = "alice",
    content: str = "hello world",
    created_at: datetime = NOW,
) -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=channel_id,
        guild_id=500,
        author_id=author_id,
        author_name_at_time=author_name,
        author_is_bot=False,
        created_at=created_at,
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        deleted_at=None,
        ingested_at=NOW,
        raw_json="{}",
    )


def an_exchange_row(
    exchange_id: int | None,
    channel_id: int = 1,
    first_message_id: int = 1,
    last_message_id: int = 1,
    message_count: int = 1,
    parent_exchange_id: int | None = None,
) -> ExchangeRow:
    return ExchangeRow(
        id=exchange_id,
        channel_id=channel_id,
        thread_id=None,
        first_message_id=first_message_id,
        last_message_id=last_message_id,
        started_at=NOW,
        ended_at=NOW,
        message_count=message_count,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=f"hash-{exchange_id}-{first_message_id}",
        parent_exchange_id=parent_exchange_id,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )


def test_build_request_uses_channel_name_and_orders_messages(tmp_path: Path) -> None:
    conn = db(tmp_path)
    upsert_channel(
        conn,
        ChannelRow(
            id=1,
            guild_id=500,
            parent_id=None,
            name="hardware",
            kind=ChannelKind.TEXT,
            archived=False,
            last_backfilled_message_id=None,
        ),
    )
    upsert_message(conn, a_message(2, content="second"))
    upsert_message(conn, a_message(1, content="first"))
    exchange_id = insert_exchange(
        conn, an_exchange_row(None, first_message_id=1, last_message_id=2, message_count=2), [1, 2]
    )
    exchange = an_exchange_row(exchange_id, first_message_id=1, last_message_id=2, message_count=2)

    request = build_request(conn, exchange)

    assert isinstance(request, ExtractionRequest)
    assert request.channel_name == "hardware"
    assert [m.id for m in request.messages] == [1, 2]
    assert [m.content for m in request.messages] == ["first", "second"]


def test_build_request_falls_back_to_channel_id_when_unknown(tmp_path: Path) -> None:
    conn = db(tmp_path)
    upsert_message(conn, a_message(1, channel_id=42))
    exchange_id = insert_exchange(conn, an_exchange_row(None, channel_id=42), [1])
    exchange = an_exchange_row(exchange_id, channel_id=42)

    request = build_request(conn, exchange)

    assert request.channel_name == "42"


def test_build_request_no_context_when_no_parent(tmp_path: Path) -> None:
    conn = db(tmp_path)
    upsert_message(conn, a_message(1))
    exchange_id = insert_exchange(conn, an_exchange_row(None), [1])
    exchange = an_exchange_row(exchange_id)

    request = build_request(conn, exchange)

    assert request.context_messages == ()


def test_build_request_context_is_last_n_messages_of_parent_exchange(tmp_path: Path) -> None:
    conn = db(tmp_path)
    for message_id in range(1, 6):
        upsert_message(conn, a_message(message_id, content=f"parent {message_id}"))
    parent_id = insert_exchange(
        conn,
        an_exchange_row(None, first_message_id=1, last_message_id=5, message_count=5),
        [1, 2, 3, 4, 5],
    )
    upsert_message(conn, a_message(6, content="child"))
    child_id = insert_exchange(
        conn,
        an_exchange_row(
            None,
            first_message_id=6,
            last_message_id=6,
            message_count=1,
            parent_exchange_id=parent_id,
        ),
        [6],
    )
    exchange = an_exchange_row(
        child_id,
        first_message_id=6,
        last_message_id=6,
        message_count=1,
        parent_exchange_id=parent_id,
    )

    request = build_request(conn, exchange, context_size=3)

    assert [m.id for m in request.context_messages] == [3, 4, 5]
    assert [m.id for m in request.messages] == [6]


def test_build_request_attachments_and_reactions_for_exchange_messages(tmp_path: Path) -> None:
    conn = db(tmp_path)
    upsert_message(conn, a_message(1))
    upsert_attachment(
        conn,
        AttachmentRow(
            id=1,
            message_id=1,
            filename="jumpers.png",
            content_type="image/png",
            size=10,
            url="https://example.com/jumpers.png",
            sha256=None,
            local_path=None,
        ),
    )
    set_reaction_count(conn, 1, "\U0001f44d", 2)
    exchange_id = insert_exchange(conn, an_exchange_row(None), [1])
    exchange = an_exchange_row(exchange_id)

    request = build_request(conn, exchange)

    assert [a.filename for a in request.attachments] == ["jumpers.png"]
    assert [(r.emoji, r.count) for r in request.reactions] == [("\U0001f44d", 2)]


def test_build_request_opted_out_user_ids_from_opt_outs_table(tmp_path: Path) -> None:
    conn = db(tmp_path)
    upsert_message(conn, a_message(1, author_id=1))
    upsert_message(conn, a_message(2, channel_id=1, author_id=7))
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (7, ?)", (to_db_time(NOW),))
    exchange_id = insert_exchange(
        conn, an_exchange_row(None, first_message_id=1, last_message_id=2, message_count=2), [1, 2]
    )
    exchange = an_exchange_row(exchange_id, first_message_id=1, last_message_id=2, message_count=2)

    request = build_request(conn, exchange)

    assert request.opted_out_user_ids == frozenset({7})


def test_build_request_related_claims_from_message_contents(tmp_path: Path) -> None:
    conn = db(tmp_path)
    upsert_message(conn, a_message(100, content="Octane2 PROM jumper settings"))
    prior_exchange_id = insert_exchange(
        conn, an_exchange_row(None, first_message_id=100, last_message_id=100), [100]
    )
    register_prompt_version(conn, "v1", "sha", NOW)
    record_run(
        conn,
        ExtractionRunRow(
            id=None,
            exchange_id=prior_exchange_id,
            model="m",
            prompt_version="v1",
            started_at=NOW,
            finished_at=NOW,
            input_tokens=1,
            output_tokens=1,
            mode=RunMode.LIVE,
            outcome=RunOutcome.OK,
            error=None,
        ),
        [
            NewClaim(
                exchange_id=prior_exchange_id,
                statement="Octane2 needs a jumper for the PROM socket",
                subject="Octane2",
                kind=ClaimKind.FACT,
                confidence=0.9,
                probe_question="what does the octane2 need?",
                permalink="https://discord.com/channels/1/1/100",
                supersedes_claim_id=None,
                source_message_ids=(100,),
            )
        ],
    )

    upsert_message(conn, a_message(200, content="Does the Octane2 need a jumper?"))
    exchange_id = insert_exchange(
        conn, an_exchange_row(None, first_message_id=200, last_message_id=200), [200]
    )
    exchange = an_exchange_row(exchange_id, first_message_id=200, last_message_id=200)

    request = build_request(conn, exchange, related_limit=10)

    assert len(request.related_claims) == 1
    assert request.related_claims[0].subject == "Octane2"


def test_build_request_related_claims_respects_related_limit(tmp_path: Path) -> None:
    conn = db(tmp_path)
    register_prompt_version(conn, "v1", "sha", NOW)
    for message_id in range(1, 4):
        upsert_message(conn, a_message(message_id, content=f"widget {message_id}"))
        prior_exchange_id = insert_exchange(
            conn,
            an_exchange_row(None, first_message_id=message_id, last_message_id=message_id),
            [message_id],
        )
        record_run(
            conn,
            ExtractionRunRow(
                id=None,
                exchange_id=prior_exchange_id,
                model="m",
                prompt_version="v1",
                started_at=NOW,
                finished_at=NOW,
                input_tokens=1,
                output_tokens=1,
                mode=RunMode.LIVE,
                outcome=RunOutcome.OK,
                error=None,
            ),
            [
                NewClaim(
                    exchange_id=prior_exchange_id,
                    statement=f"Widget model {message_id} needs a bracket",
                    subject="Widget",
                    kind=ClaimKind.FACT,
                    confidence=0.9,
                    probe_question=f"what does widget {message_id} need?",
                    permalink="https://discord.com/channels/1/1/1",
                    supersedes_claim_id=None,
                    source_message_ids=(message_id,),
                )
            ],
        )

    upsert_message(conn, a_message(100, content="widget compatibility question"))
    exchange_id = insert_exchange(
        conn, an_exchange_row(None, first_message_id=100, last_message_id=100), [100]
    )
    exchange = an_exchange_row(exchange_id, first_message_id=100, last_message_id=100)

    request = build_request(conn, exchange, related_limit=2)

    assert len(request.related_claims) == 2


def test_build_request_related_claims_ignore_opted_out_authors_content(tmp_path: Path) -> None:
    conn = db(tmp_path)
    register_prompt_version(conn, "v1", "sha", NOW)
    upsert_message(conn, a_message(1, content="Zorblatt jumper settings"))
    prior_exchange_id = insert_exchange(
        conn, an_exchange_row(None, first_message_id=1, last_message_id=1), [1]
    )
    record_run(
        conn,
        ExtractionRunRow(
            id=None,
            exchange_id=prior_exchange_id,
            model="m",
            prompt_version="v1",
            started_at=NOW,
            finished_at=NOW,
            input_tokens=1,
            output_tokens=1,
            mode=RunMode.LIVE,
            outcome=RunOutcome.OK,
            error=None,
        ),
        [
            NewClaim(
                exchange_id=prior_exchange_id,
                statement="Zorblatt needs a jumper on pin 3",
                subject="Zorblatt",
                kind=ClaimKind.FACT,
                confidence=0.9,
                probe_question="what does the zorblatt need?",
                permalink="https://discord.com/channels/1/1/1",
                supersedes_claim_id=None,
                source_message_ids=(1,),
            )
        ],
    )

    upsert_message(conn, a_message(200, author_id=1, content="totally unrelated chatter today"))
    upsert_message(conn, a_message(201, author_id=99, content="Zorblatt Zorblatt Zorblatt"))
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (99, ?)", (to_db_time(NOW),))
    exchange_id = insert_exchange(
        conn,
        an_exchange_row(None, first_message_id=200, last_message_id=201, message_count=2),
        [200, 201],
    )
    exchange = an_exchange_row(
        exchange_id, first_message_id=200, last_message_id=201, message_count=2
    )

    request = build_request(conn, exchange, related_limit=10)

    assert request.related_claims == ()
