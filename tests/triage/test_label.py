import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.claims import NewClaim, record_run, register_prompt_version, set_novelty
from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database
from infovore.db.labels import effective_labels
from infovore.rows import ClaimKind, ExtractionRunRow, Label, Novelty, RunMode, RunOutcome
from infovore.triage.label import DeriveReport, UnknownRunError, derive_labels_from_runs

NOW = datetime(2026, 1, 1, tzinfo=UTC)
NOW_TEXT = to_db_time(NOW)


def seeded(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    register_prompt_version(conn, "v1", "sha", NOW)
    return conn


def make_exchange(conn: sqlite3.Connection, exchange_id: int, message_id: int) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, 1, 1, 1, 'a', ?, 'x', ?, '{}')",
        (message_id, NOW_TEXT, NOW_TEXT),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, 1, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
        (exchange_id, message_id, message_id, NOW_TEXT, NOW_TEXT, f"h{exchange_id}"),
    )


def a_claim(exchange_id: int, statement: str = "s") -> NewClaim:
    return NewClaim(
        exchange_id=exchange_id,
        statement=statement,
        subject="subj",
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question="q?",
        permalink="https://discord.com/channels/1/1/1",
        supersedes_claim_id=None,
        source_message_ids=(exchange_id,),
    )


def record(
    conn: sqlite3.Connection,
    exchange_id: int,
    message_id: int,
    claims: list[NewClaim],
    outcome: RunOutcome = RunOutcome.OK,
    new_exchange: bool = True,
) -> int:
    if new_exchange:
        make_exchange(conn, exchange_id, message_id)
    result = record_run(
        conn,
        ExtractionRunRow(
            id=None,
            exchange_id=exchange_id,
            model="m",
            prompt_version="v1",
            started_at=NOW,
            finished_at=NOW,
            input_tokens=1,
            output_tokens=1,
            mode=RunMode.TRIAL,
            outcome=outcome,
            error="boom" if outcome is RunOutcome.FAILED else None,
        ),
        claims,
    )
    for claim_id, verdict in zip(result.claim_ids, _verdicts_for(claims), strict=True):
        if verdict is not None:
            set_novelty(conn, claim_id, verdict, "probe-model", "answer", NOW)
    return result.run_id


_VERDICT_MARKERS = {
    "unknown": Novelty.UNKNOWN,
    "partial": Novelty.PARTIAL,
    "contradicts": Novelty.CONTRADICTS,
    "known": Novelty.KNOWN,
}


def _verdicts_for(claims: list[NewClaim]) -> list[Novelty | None]:
    verdicts: list[Novelty | None] = []
    for claim in claims:
        verdicts.append(_VERDICT_MARKERS.get(claim.statement))
    return verdicts


def test_derive_labels_lore_when_a_claim_was_probed_unknown(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    run_id = record(conn, 1, 1, [a_claim(1, "unknown")])
    report = derive_labels_from_runs(conn, [run_id], NOW)
    assert report == DeriveReport(lore=1, noise=0, skipped_reasons={})
    assert effective_labels(conn) == {1: Label.LORE}


def test_derive_labels_lore_when_a_claim_was_probed_partial(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    run_id = record(conn, 1, 1, [a_claim(1, "known"), a_claim(1, "partial")])
    report = derive_labels_from_runs(conn, [run_id], NOW)
    assert report.lore == 1
    assert effective_labels(conn) == {1: Label.LORE}


def test_derive_labels_lore_when_a_claim_was_probed_contradicts(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    run_id = record(conn, 1, 1, [a_claim(1, "contradicts")])
    report = derive_labels_from_runs(conn, [run_id], NOW)
    assert report.lore == 1
    assert effective_labels(conn) == {1: Label.LORE}


def test_derive_labels_noise_for_a_successful_run_with_zero_claims(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    run_id = record(conn, 1, 1, [])
    report = derive_labels_from_runs(conn, [run_id], NOW)
    assert report == DeriveReport(lore=0, noise=1, skipped_reasons={})
    assert effective_labels(conn) == {1: Label.NOISE}


def test_derive_labels_noise_when_every_claim_is_known(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    run_id = record(conn, 1, 1, [a_claim(1, "known"), a_claim(1, "known")])
    report = derive_labels_from_runs(conn, [run_id], NOW)
    assert report.noise == 1
    assert effective_labels(conn) == {1: Label.NOISE}


def test_derive_labels_skips_a_failed_run(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    run_id = record(conn, 1, 1, [], outcome=RunOutcome.FAILED)
    report = derive_labels_from_runs(conn, [run_id], NOW)
    assert report == DeriveReport(lore=0, noise=0, skipped_reasons={run_id: "failed run"})
    assert effective_labels(conn) == {}


def test_derive_labels_skips_a_run_with_a_still_unprobed_claim(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    run_id = record(conn, 1, 1, [a_claim(1, "known"), a_claim(1, "unprobed-marker")])
    report = derive_labels_from_runs(conn, [run_id], NOW)
    assert report == DeriveReport(lore=0, noise=0, skipped_reasons={run_id: "unprobed claims"})
    assert effective_labels(conn) == {}


def test_derive_labels_raises_for_an_unknown_run_id(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    with pytest.raises(UnknownRunError, match="999"):
        derive_labels_from_runs(conn, [999], NOW)


def test_derive_labels_processes_several_runs_and_calls_progress(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    lore_run = record(conn, 1, 1, [a_claim(1, "unknown")])
    noise_run = record(conn, 2, 2, [])
    failed_run = record(conn, 3, 3, [], outcome=RunOutcome.FAILED)
    unprobed_run = record(conn, 4, 4, [a_claim(4, "known"), a_claim(4, "unprobed-marker")])
    events: list[tuple[int, str]] = []

    def on_progress(run_id: int, outcome: str) -> None:
        events.append((run_id, outcome))

    report = derive_labels_from_runs(
        conn, [lore_run, noise_run, failed_run, unprobed_run], NOW, progress=on_progress
    )
    assert report.lore == 1
    assert report.noise == 1
    assert report.skipped == 2
    assert events == [
        (lore_run, "lore"),
        (noise_run, "noise"),
        (failed_run, "skipped (failed run)"),
        (unprobed_run, "skipped (unprobed claims)"),
    ]


def test_derive_labels_is_idempotent_and_relabels_the_llm_source(tmp_path: Path) -> None:
    conn = seeded(tmp_path)
    run_id = record(conn, 1, 1, [a_claim(1, "unknown")])
    derive_labels_from_runs(conn, [run_id], NOW)
    other_run = record(conn, 2, 2, [])
    another_run_over_exchange_1 = record(conn, 1, 1, [], new_exchange=False)
    derive_labels_from_runs(conn, [other_run, another_run_over_exchange_1], NOW)
    assert effective_labels(conn) == {1: Label.NOISE, 2: Label.NOISE}
