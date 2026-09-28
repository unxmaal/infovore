import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.claims import record_run
from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import insert_exchange
from infovore.db.message_labels import effective_message_labels_with_source
from infovore.rows import (
    ClaimKind,
    ExchangeRow,
    ExtractionRunRow,
    ExtractionStatus,
    GroupingRule,
    MessageLabel,
    MessageLabelSource,
    RunMode,
    RunOutcome,
)
from infovore.sift.citations import CitationLabelReport, derive_citation_labels

NOW = datetime(2026, 1, 1, tzinfo=UTC)
NOW_TEXT = NOW.isoformat()


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute("INSERT INTO channels (id, guild_id, name, kind) VALUES (1, 1, 'general', 'text')")
    return conn


def seed_message(conn: sqlite3.Connection, message_id: int, channel_id: int = 1) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 1, 1, 'alice', ?, 'hi', ?, '{}')",
        (message_id, channel_id, NOW_TEXT, NOW_TEXT),
    )


def seed_exchange(conn: sqlite3.Connection, exchange_id: int, message_ids: list[int]) -> int:
    for message_id in message_ids:
        seed_message(conn, message_id)
    row = ExchangeRow(
        id=None,
        channel_id=1,
        thread_id=None,
        first_message_id=message_ids[0],
        last_message_id=message_ids[-1],
        started_at=NOW,
        ended_at=NOW,
        message_count=len(message_ids),
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=f"hash-{exchange_id}",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    return insert_exchange(conn, row, message_ids)


def record_live_ok_run(
    conn: sqlite3.Connection, exchange_id: int, cited_message_ids: tuple[int, ...]
) -> int:
    from infovore.db.claims import NewClaim

    run = ExtractionRunRow(
        id=None,
        exchange_id=exchange_id,
        model="claude",
        prompt_version="v1",
        started_at=NOW,
        finished_at=NOW,
        input_tokens=10,
        output_tokens=10,
        mode=RunMode.LIVE,
        outcome=RunOutcome.OK,
        error=None,
    )
    conn.execute(
        "INSERT OR IGNORE INTO prompt_versions (version, text_sha256, created_at, promoted_at)"
        " VALUES ('v1', 'sha', ?, NULL)",
        (NOW_TEXT,),
    )
    claims = (
        [
            NewClaim(
                exchange_id=exchange_id,
                statement="a fact",
                subject="thing",
                kind=ClaimKind.FACT,
                confidence=0.9,
                probe_question="q?",
                permalink="https://example.com",
                supersedes_claim_id=None,
                source_message_ids=cited_message_ids,
            )
        ]
        if cited_message_ids
        else []
    )
    recorded = record_run(conn, run, claims)
    return recorded.run_id


def test_derive_citation_labels_keeps_cited_and_trashes_uncited(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, [1, 2, 3])
    record_live_ok_run(conn, exchange_id, (1, 2))

    report = derive_citation_labels(conn, NOW)

    assert report == CitationLabelReport(exchanges_considered=1, keep=2, trash=1)
    labels = effective_message_labels_with_source(conn)
    assert labels[1] == (MessageLabel.KEEP, MessageLabelSource.CITATION)
    assert labels[2] == (MessageLabel.KEEP, MessageLabelSource.CITATION)
    assert labels[3] == (MessageLabel.TRASH, MessageLabelSource.CITATION)


def test_derive_citation_labels_skips_exchanges_without_a_successful_live_run(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, [1, 2])
    run = ExtractionRunRow(
        id=None,
        exchange_id=exchange_id,
        model="claude",
        prompt_version="v1",
        started_at=NOW,
        finished_at=NOW,
        input_tokens=1,
        output_tokens=1,
        mode=RunMode.TRIAL,
        outcome=RunOutcome.OK,
        error=None,
    )
    conn.execute(
        "INSERT OR IGNORE INTO prompt_versions (version, text_sha256, created_at, promoted_at)"
        " VALUES ('v1', 'sha', ?, NULL)",
        (NOW_TEXT,),
    )
    record_run(conn, run, [])

    report = derive_citation_labels(conn, NOW)

    assert report == CitationLabelReport(exchanges_considered=0, keep=0, trash=0)
    assert effective_message_labels_with_source(conn) == {}


def test_derive_citation_labels_skips_a_failed_live_run(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, [1, 2])
    conn.execute(
        "INSERT OR IGNORE INTO prompt_versions (version, text_sha256, created_at, promoted_at)"
        " VALUES ('v1', 'sha', ?, NULL)",
        (NOW_TEXT,),
    )
    failed_run = ExtractionRunRow(
        id=None,
        exchange_id=exchange_id,
        model="claude",
        prompt_version="v1",
        started_at=NOW,
        finished_at=NOW,
        input_tokens=None,
        output_tokens=None,
        mode=RunMode.LIVE,
        outcome=RunOutcome.FAILED,
        error="boom",
    )
    record_run(conn, failed_run, [])

    report = derive_citation_labels(conn, NOW)

    assert report.exchanges_considered == 0
    assert effective_message_labels_with_source(conn) == {}


def test_derive_citation_labels_ignores_retracted_claims(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, [1, 2])
    record_live_ok_run(conn, exchange_id, (1,))
    claim_id = conn.execute("SELECT id FROM claims LIMIT 1").fetchone()["id"]
    conn.execute(
        "UPDATE claims SET retracted_at = ?, retraction_reason = 'test' WHERE id = ?",
        (NOW_TEXT, claim_id),
    )

    report = derive_citation_labels(conn, NOW)

    assert report == CitationLabelReport(exchanges_considered=1, keep=0, trash=2)
    labels = effective_message_labels_with_source(conn)
    assert labels[1] == (MessageLabel.TRASH, MessageLabelSource.CITATION)


def test_derive_citation_labels_is_idempotent_on_rerun(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, [1, 2])
    record_live_ok_run(conn, exchange_id, (1,))

    derive_citation_labels(conn, NOW)
    derive_citation_labels(conn, NOW)

    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM message_labels WHERE source = 'citation'"
    ).fetchone()
    assert rows["n"] == 2


def test_derive_citation_labels_does_not_override_a_human_label_at_read_time(
    tmp_path: Path,
) -> None:
    from infovore.db.message_labels import set_message_label

    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, [1, 2])
    set_message_label(conn, 1, MessageLabel.KEEP, MessageLabelSource.HUMAN, None, NOW)
    record_live_ok_run(conn, exchange_id, ())

    derive_citation_labels(conn, NOW)

    labels = effective_message_labels_with_source(conn)
    assert labels[1] == (MessageLabel.KEEP, MessageLabelSource.HUMAN)
    assert labels[2] == (MessageLabel.TRASH, MessageLabelSource.CITATION)
