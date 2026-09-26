import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.claims import (
    NewClaim,
    get_claim,
    record_run,
    register_prompt_version,
    retract_claim,
    set_novelty,
)
from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database
from infovore.extract.fake import MarkerProbe
from infovore.extract.novelty import ProbeReport, run_probe
from infovore.extract.protocol import Failure, FailureKind, ProbeOutcome
from infovore.rows import ClaimKind, ClaimRow, ExtractionRunRow, Novelty, RunMode, RunOutcome
from infovore.timing import FixedClock, RecordingSleeper

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def insert_message(conn: sqlite3.Connection, message_id: int) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, 1, 1, 1, 'a', ?, 'x', ?, '{}')",
        (message_id, to_db_time(NOW), to_db_time(NOW)),
    )


def insert_exchange(conn: sqlite3.Connection, exchange_id: int, message_id: int) -> None:
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, 1, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
        (exchange_id, message_id, message_id, to_db_time(NOW), to_db_time(NOW), f"h{exchange_id}"),
    )


def a_run(exchange_id: int = 1, mode: RunMode = RunMode.LIVE) -> ExtractionRunRow:
    return ExtractionRunRow(
        id=None,
        exchange_id=exchange_id,
        model="m",
        prompt_version="v1",
        started_at=NOW,
        finished_at=NOW,
        input_tokens=1,
        output_tokens=1,
        mode=mode,
        outcome=RunOutcome.OK,
        error=None,
    )


def seed_claim(
    conn: sqlite3.Connection,
    statement: str,
    exchange_id: int = 1,
    message_id: int = 1,
    mode: RunMode = RunMode.LIVE,
) -> int:
    if conn.execute("SELECT 1 FROM exchanges WHERE id = ?", (exchange_id,)).fetchone() is None:
        insert_message(conn, message_id)
        insert_exchange(conn, exchange_id, message_id)
    result = record_run(
        conn,
        a_run(exchange_id=exchange_id, mode=mode),
        [
            _new_claim(exchange_id, statement, message_id),
        ],
    )
    return result.claim_ids[0]


def _new_claim(exchange_id: int, statement: str, message_id: int) -> NewClaim:
    return NewClaim(
        exchange_id=exchange_id,
        statement=statement,
        subject="s",
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question=f"what about {statement}?",
        permalink="https://discord.com/channels/1/1/1",
        supersedes_claim_id=None,
        source_message_ids=(message_id,),
    )


def setup_db(tmp_path: Path) -> sqlite3.Connection:
    conn = db(tmp_path)
    register_prompt_version(conn, "v1", "sha", NOW)
    return conn


class ScriptedProbe:
    def __init__(self, outcomes_by_claim_id: dict[int, list[ProbeOutcome]]) -> None:
        self._outcomes = outcomes_by_claim_id
        self.calls: list[int] = []
        self.max_concurrent = 0
        self._in_flight = 0

    async def probe(self, claim: ClaimRow) -> ProbeOutcome:
        assert claim.id is not None
        self._in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self._in_flight)
        self.calls.append(claim.id)
        await asyncio.sleep(0)
        outcomes = self._outcomes[claim.id]
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        self._in_flight -= 1
        return outcome


async def test_run_probe_each_verdict_path(tmp_path: Path) -> None:
    conn = setup_db(tmp_path)
    known_id = seed_claim(conn, "widget A [known]", exchange_id=1, message_id=1)
    partial_id = seed_claim(conn, "widget B [partial]", exchange_id=2, message_id=2)
    contradicts_id = seed_claim(conn, "widget C [contradicts]", exchange_id=3, message_id=3)
    unknown_id = seed_claim(conn, "widget D [unknown]", exchange_id=4, message_id=4)

    report = await run_probe(
        conn,
        MarkerProbe(),
        FixedClock(NOW),
        RecordingSleeper(),
        probe_model=None,
        limit=10,
        concurrency=2,
    )

    assert isinstance(report, ProbeReport)
    assert report.probed == 4
    assert report.failed == 0
    assert report.pauses == 0
    assert report.by_verdict == {
        Novelty.KNOWN: 1,
        Novelty.PARTIAL: 1,
        Novelty.CONTRADICTS: 1,
        Novelty.UNKNOWN: 1,
    }

    for claim_id, verdict in (
        (known_id, Novelty.KNOWN),
        (partial_id, Novelty.PARTIAL),
        (contradicts_id, Novelty.CONTRADICTS),
        (unknown_id, Novelty.UNKNOWN),
    ):
        claim = get_claim(conn, claim_id)
        assert claim is not None
        assert claim.novelty is verdict
        assert claim.probe_model == "fake-probe"
        assert claim.probe_answer == "fake answer"
        assert claim.probed_at == NOW
        assert claim.probe_error is None


async def test_run_probe_failure_leaves_claim_unprobed_with_error(tmp_path: Path) -> None:
    conn = setup_db(tmp_path)
    fail_id = seed_claim(conn, "widget E [probe-fail]")

    report = await run_probe(
        conn,
        MarkerProbe(),
        FixedClock(NOW),
        RecordingSleeper(),
        probe_model=None,
        limit=10,
        concurrency=2,
    )

    assert report.probed == 0
    assert report.failed == 1
    assert report.pauses == 0
    claim = get_claim(conn, fail_id)
    assert claim is not None
    assert claim.novelty is Novelty.UNPROBED
    assert claim.probe_error == "fake probe failure"


async def test_run_probe_usage_limit_pauses_then_retries_same_claim(tmp_path: Path) -> None:
    conn = setup_db(tmp_path)
    claim_id = seed_claim(conn, "widget F [known]")
    probe = ScriptedProbe(
        {
            claim_id: [
                ProbeOutcome(None, None, None, Failure(FailureKind.USAGE_LIMIT, "limited", 12.0)),
                ProbeOutcome(Novelty.KNOWN, "fake-model", "answer", None),
            ]
        }
    )
    sleeper = RecordingSleeper()

    report = await run_probe(
        conn,
        probe,
        FixedClock(NOW),
        sleeper,
        probe_model=None,
        limit=10,
        concurrency=2,
    )

    assert report.probed == 1
    assert report.pauses == 1
    assert sleeper.slept == [12.0]
    claim = get_claim(conn, claim_id)
    assert claim is not None
    assert claim.novelty is Novelty.KNOWN


async def test_run_probe_usage_limit_defaults_retry_after(tmp_path: Path) -> None:
    conn = setup_db(tmp_path)
    claim_id = seed_claim(conn, "widget G [known]")
    probe = ScriptedProbe(
        {
            claim_id: [
                ProbeOutcome(None, None, None, Failure(FailureKind.USAGE_LIMIT, "limited", None)),
                ProbeOutcome(Novelty.KNOWN, "fake-model", "answer", None),
            ]
        }
    )
    sleeper = RecordingSleeper()

    await run_probe(
        conn, probe, FixedClock(NOW), sleeper, probe_model=None, limit=10, concurrency=2
    )
    assert sleeper.slept == [300.0]


async def test_run_probe_idempotent_per_model_second_run_probes_nothing(tmp_path: Path) -> None:
    conn = setup_db(tmp_path)
    seed_claim(conn, "widget H [known]")

    first = await run_probe(
        conn,
        MarkerProbe(),
        FixedClock(NOW),
        RecordingSleeper(),
        probe_model="fake-probe",
        limit=10,
        concurrency=2,
    )
    assert first.probed == 1

    second = await run_probe(
        conn,
        MarkerProbe(),
        FixedClock(NOW),
        RecordingSleeper(),
        probe_model="fake-probe",
        limit=10,
        concurrency=2,
    )
    assert second.probed == 0
    assert second.failed == 0


async def test_run_probe_reprobes_on_probe_model_change(tmp_path: Path) -> None:
    conn = setup_db(tmp_path)
    claim_id = seed_claim(conn, "widget I [known]")
    set_novelty(conn, claim_id, Novelty.KNOWN, "old-model", "old answer", NOW)

    unchanged = await run_probe(
        conn,
        MarkerProbe(),
        FixedClock(NOW),
        RecordingSleeper(),
        probe_model="old-model",
        limit=10,
        concurrency=2,
    )
    assert unchanged.probed == 0

    reprobed = await run_probe(
        conn,
        MarkerProbe(),
        FixedClock(NOW),
        RecordingSleeper(),
        probe_model="fake-probe",
        limit=10,
        concurrency=2,
    )
    assert reprobed.probed == 1
    claim = get_claim(conn, claim_id)
    assert claim is not None
    assert claim.probe_model == "fake-probe"

    settled = await run_probe(
        conn,
        MarkerProbe(),
        FixedClock(NOW),
        RecordingSleeper(),
        probe_model="fake-probe",
        limit=10,
        concurrency=2,
    )
    assert settled.probed == 0


async def test_run_probe_run_ids_scopes_to_those_runs_claims(tmp_path: Path) -> None:
    conn = setup_db(tmp_path)
    insert_message(conn, 1)
    insert_exchange(conn, 1, 1)
    insert_message(conn, 2)
    insert_exchange(conn, 2, 2)
    in_scope = record_run(conn, a_run(exchange_id=1), [_new_claim(1, "widget J [known]", 1)])
    out_of_scope = record_run(conn, a_run(exchange_id=2), [_new_claim(2, "widget K [known]", 2)])

    report = await run_probe(
        conn,
        MarkerProbe(),
        FixedClock(NOW),
        RecordingSleeper(),
        probe_model=None,
        limit=10,
        concurrency=2,
        run_ids=(in_scope.run_id,),
    )

    assert report.probed == 1
    in_scope_claim = get_claim(conn, in_scope.claim_ids[0])
    assert in_scope_claim is not None
    assert in_scope_claim.novelty is Novelty.KNOWN
    out_of_scope_claim = get_claim(conn, out_of_scope.claim_ids[0])
    assert out_of_scope_claim is not None
    assert out_of_scope_claim.novelty is Novelty.UNPROBED


async def test_run_probe_skips_retracted_claims(tmp_path: Path) -> None:
    conn = setup_db(tmp_path)
    claim_id = seed_claim(conn, "widget L [known]")
    retract_claim(conn, claim_id, "sources_deleted", NOW)

    report = await run_probe(
        conn,
        MarkerProbe(),
        FixedClock(NOW),
        RecordingSleeper(),
        probe_model=None,
        limit=10,
        concurrency=2,
    )
    assert report.probed == 0
    assert report.failed == 0


async def test_run_probe_bounds_concurrency(tmp_path: Path) -> None:
    conn = setup_db(tmp_path)
    ids = [
        seed_claim(conn, f"widget M{i} [known]", exchange_id=i, message_id=i) for i in range(1, 6)
    ]
    probe = ScriptedProbe(
        {claim_id: [ProbeOutcome(Novelty.KNOWN, "m", "a", None)] for claim_id in ids}
    )

    report = await run_probe(
        conn,
        probe,
        FixedClock(NOW),
        RecordingSleeper(),
        probe_model=None,
        limit=10,
        concurrency=2,
    )

    assert report.probed == 5
    assert probe.max_concurrent == 2
