import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.annotations import Annotation, record_annotation
from infovore.db.archive import export_archive
from infovore.db.connection import migrate, open_database
from infovore.db.exchange_search import search_exchanges
from infovore.db.exchanges import claimable_exchanges, has_untriaged_claimable
from infovore.db.status import collect_status
from infovore.eval.channel_report import channel_report
from infovore.eval.judge import (
    c1_queue,
    frozen_queue,
    likely_irrelevant_queue,
    uncertain_queue,
)
from infovore.eval.slices import slice_ids, slice_summary
from infovore.extract.prompt_compare import queue_size_mix, sample_extracted_exchanges
from infovore.extract.runner import select_trial_sample
from infovore.triage.human import held_out_ids, trainable_labels
from infovore.triage.report import compute_triage_stats
from infovore.triage.rules import DEFAULT_RULES
from tests.cascade_marks import mark_all

AT = "2026-01-01T00:00:00+00:00"
CURRENT, OLD = 1, 2


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "c.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO channels (id, guild_id, name, kind) VALUES (1, 9, 'general', 'text')"
    )
    for exchange_id in (CURRENT, OLD):
        first = exchange_id * 10
        for offset in range(3):
            connection.execute(
                "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
                " created_at, content, ingested_at, raw_json)"
                " VALUES (?, 1, 9, 1, 'a', ?, 'octane prom flashing', ?, '{}')",
                (first + offset, AT, AT),
            )
        connection.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
            " started_at, ended_at, message_count, grouping_rule, content_hash, triage_score,"
            " triage_version, p_lore, extraction_status)"
            " VALUES (?, 1, ?, ?, ?, ?, 3, 'quiet_gap', ?, 0.9, ?, 0.99, 'pending')",
            (exchange_id, first, first + 2, AT, AT, f"h{exchange_id}", DEFAULT_RULES.version),
        )
        for offset in range(3):
            connection.execute(
                "INSERT INTO exchange_messages (exchange_id, message_id, position)"
                " VALUES (?, ?, ?)",
                (exchange_id, first + offset, offset),
            )
    connection.execute("UPDATE exchanges SET superseded_by_recipe = 1 WHERE id = ?", (OLD,))
    mark_all(connection, [CURRENT, OLD], "residue")
    connection.commit()
    return connection


def run_for(conn: sqlite3.Connection, exchange_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO prompt_versions (version, text_sha256, created_at, promoted_at)"
        " VALUES ('v5', 'x', ?, ?)",
        (AT, AT),
    )
    conn.execute(
        "INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at,"
        " input_tokens, output_tokens, mode, outcome)"
        " VALUES (?, 'm', 'v5', ?, 5000, 500, 'live', 'ok')",
        (exchange_id, AT),
    )


def freeze(conn: sqlite3.Connection, name: str, ids: list[int]) -> None:
    for position, exchange_id in enumerate(ids, start=1):
        conn.execute(
            "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
            " VALUES (?, ?, ?, 'queue', 1, ?)",
            (name, exchange_id, position, AT),
        )


def label(conn: sqlite3.Connection, exchange_id: int, value: str) -> None:
    record_annotation(
        conn,
        Annotation(
            subject_kind="exchange",
            subject_id=exchange_id,
            scorer="human_exchange",
            scorer_version=1,
            reproducibility="recorded",
            label=value,
            source_ref=None,
        ),
        datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_the_view_is_the_definition(conn: sqlite3.Connection) -> None:
    ids = [row[0] for row in conn.execute("SELECT id FROM current_exchanges")]
    assert ids == [CURRENT]


def test_status_counts_only_current_exchanges(conn: sqlite3.Connection) -> None:
    report = collect_status(conn, triage_min_score=0.3)
    assert report.exchanges_by_status == {"pending": 1}
    assert report.triaged_exchanges == 1
    assert report.above_threshold_exchanges == 1
    assert report.archived_exchanges == 1
    assert report.pending_archived == 1
    assert report.pending_exchanges == 1
    denied = collect_status(conn, exclude_channels=frozenset({"general"}))
    assert denied.excluded_by_denylist == 1


def test_triage_readers_ignore_superseded_exchanges(conn: sqlite3.Connection) -> None:
    run_for(conn, CURRENT)
    run_for(conn, OLD)
    stats = compute_triage_stats(conn, 0.3)
    assert stats.above_threshold + stats.below_threshold == 1


def test_judge_queues_never_serve_superseded_exchanges(conn: sqlite3.Connection) -> None:
    served = [
        item.exchange_id
        for item in uncertain_queue(3, frozenset())(conn)
        + c1_queue(frozenset())(conn)
        + likely_irrelevant_queue(frozenset())(conn)
    ]
    assert served
    assert OLD not in served
    assert len(channel_report(conn, frozenset(), min_labels=0)) <= 1


def test_frozen_queue_and_slices_resolve_to_current_exchanges(conn: sqlite3.Connection) -> None:
    freeze(conn, "build", [CURRENT, OLD])
    freeze(conn, "build@2", [CURRENT, OLD])
    assert slice_ids(conn, "build") == [CURRENT]
    assert sum(b.exchanges for b in slice_summary(conn, "build")) == 1
    assert OLD not in [item.exchange_id for item in frozen_queue(conn)]
    assert held_out_ids(conn) == frozenset()


def test_labels_on_superseded_exchanges_do_not_train(conn: sqlite3.Connection) -> None:
    label(conn, CURRENT, "relevant")
    label(conn, OLD, "irrelevant")
    labels, _ = trainable_labels(conn)
    assert set(labels) == {CURRENT}


def test_extraction_queue_never_claims_superseded_exchanges(conn: sqlite3.Connection) -> None:
    assert [e.id for e in claimable_exchanges(conn, 10, 3)] == [CURRENT]
    conn.execute("UPDATE exchanges SET triage_version = 'old' WHERE id = ?", (CURRENT,))
    assert has_untriaged_claimable(conn, DEFAULT_RULES.version, 3) is True
    conn.execute(
        "UPDATE exchanges SET triage_version = ? WHERE id = ?", (DEFAULT_RULES.version, CURRENT)
    )
    conn.execute("UPDATE exchanges SET triage_version = 'old' WHERE id = ?", (OLD,))
    assert has_untriaged_claimable(conn, DEFAULT_RULES.version, 3) is False
    assert select_trial_sample(conn, 10, 0) == [CURRENT]
    assert sum(queue_size_mix(conn).values()) == 1


def test_prompt_compare_samples_only_current_exchanges(conn: sqlite3.Connection) -> None:
    run_for(conn, CURRENT)
    run_for(conn, OLD)
    conn.execute("UPDATE exchanges SET extraction_status = 'done'")
    assert sample_extracted_exchanges(conn, 10) == [CURRENT]


def test_search_and_export_never_serve_superseded_exchanges(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    hits = search_exchanges(conn, "octane")
    assert [hit.exchange_id for hit in hits] == [CURRENT]
    rejected = search_exchanges(conn, "octane", include_rejected=True)
    assert [hit.exchange_id for hit in rejected] == [CURRENT]
    report = export_archive(conn, tmp_path / "a.db")
    assert report.exchanges == 1
    assert report.messages == 3
    archived = sqlite3.connect(tmp_path / "a.db")
    assert [row[0] for row in archived.execute("SELECT id FROM exchanges")] == [CURRENT]


ALLOWED_RAW_EXCHANGES = {
    "infovore/chunk/rechunk.py",
    "infovore/db/archive.py",
    "infovore/db/exchanges.py",
    "infovore/db/connection.py",
}
RAW_EXCHANGES = re.compile(r"\b(?:FROM|JOIN)\s+exchanges\b", re.IGNORECASE)


def test_no_reader_queries_the_exchanges_table_directly() -> None:
    root = Path(__file__).resolve().parent.parent / "infovore"
    offenders = sorted(
        str(path.relative_to(root.parent))
        for path in root.rglob("*.py")
        if str(path.relative_to(root.parent)) not in ALLOWED_RAW_EXCHANGES
        and RAW_EXCHANGES.search(path.read_text())
    )
    assert offenders == [], "read exchanges through current_exchanges"
