import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.extract.prompt_compare import (
    SIZE_BUCKETS,
    ArmLabelMismatchError,
    ArmResult,
    PromptComparison,
    format_comparison,
    queue_size_mix,
    run_arm,
    sample_extracted_exchanges,
)
from infovore.extract.protocol import (
    ExtractedClaim,
    ExtractionOutcome,
    ExtractionRequest,
    Failure,
    FailureKind,
)
from infovore.rows import ClaimKind

AT = datetime(2026, 1, 1, tzinfo=UTC)


class _Stub:
    """Returns a fixed claim set per call, so the harness's arithmetic is
    tested without any model."""

    def __init__(self, claims: list[ExtractedClaim], failure: Failure | None = None):
        self._claims = claims
        self._failure = failure
        self.calls = 0

    async def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        self.calls += 1
        return ExtractionOutcome(
            claims=() if self._failure else tuple(self._claims),
            model="stub",
            input_tokens=1000,
            output_tokens=500,
            failure=self._failure,
            cost_usd=None,
        )


def _claim(statement: str, probe: str = "") -> ExtractedClaim:
    return ExtractedClaim(
        statement=statement,
        subject="SGI O2",
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question=probe,
        source_message_ids=(1,),
        supersedes_claim_id=None,
    )


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    connection.execute(
        "INSERT INTO prompt_versions (version, text_sha256, created_at) VALUES ('v5', 's', ?)",
        (AT.isoformat(),),
    )
    for exchange_id in (1, 2, 3):
        connection.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 1, 1, 'a', ?, 'the quick brown fox jumped', ?, '{}')",
            (exchange_id, AT.isoformat(), AT.isoformat()),
        )
        connection.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
            " started_at, ended_at, message_count, grouping_rule, content_hash,"
            " extraction_status) VALUES (?, 1, ?, ?, ?, ?, 1, 'quiet_gap', ?, 'done')",
            (
                exchange_id,
                exchange_id,
                exchange_id,
                AT.isoformat(),
                AT.isoformat(),
                f"h{exchange_id}",
            ),
        )
        connection.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
            (exchange_id, exchange_id),
        )
        connection.execute(
            "INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at,"
            " mode, outcome, input_tokens, output_tokens)"
            " VALUES (?, 'm', 'v5', ?, 'live', 'ok', 100, 100)",
            (exchange_id, AT.isoformat()),
        )
    return connection


def test_the_sample_is_stable_across_calls(conn: sqlite3.Connection) -> None:
    """A comparison whose sample moves cannot be re-run against its own
    earlier result."""
    first = sample_extracted_exchanges(conn, 2)

    assert first == sample_extracted_exchanges(conn, 2)
    assert first == sorted(first)


def test_the_sample_only_offers_already_extracted_exchanges(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE exchanges SET extraction_status = 'pending' WHERE id = 1")

    assert sample_extracted_exchanges(conn, 10) == [2, 3]


@pytest.mark.asyncio
async def test_an_arm_counts_claims_and_tokens(conn: sqlite3.Connection) -> None:
    stub = _Stub([_claim("The O2 PSU is 150W."), _claim("The Indy takes 256MB.")])

    arm = await run_arm(conn, stub, "v6", [1, 2])

    assert stub.calls == 2
    assert arm.claims == 4
    assert arm.input_tokens == 2000
    assert arm.output_tokens == 1000
    assert arm.claims_per_exchange == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_an_arm_writes_nothing_to_claims(conn: sqlite3.Connection) -> None:
    """A comparison that mutates `claims` destroys its own baseline."""
    before = conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"]

    await run_arm(conn, _Stub([_claim("The O2 PSU is 150W.")]), "v6", [1, 2, 3])

    assert conn.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"] == before


@pytest.mark.asyncio
async def test_an_arm_measures_compression_against_the_source(conn: sqlite3.Connection) -> None:
    """v5 emits 1.075 chars of claim per char of source across the corpus. The
    comparison has to be able to see that number move."""
    stub = _Stub([_claim("x" * 100)])

    arm = await run_arm(conn, stub, "v6", [1])

    assert arm.source_chars == len("the quick brown fox jumped")
    assert arm.compression > 1.0


@pytest.mark.asyncio
async def test_an_arm_measures_the_person_subject_share(conn: sqlite3.Connection) -> None:
    stub = _Stub(
        [
            _claim("A community member reported the PSU died."),
            _claim("The O2 PSU is rated 150W."),
        ]
    )

    arm = await run_arm(conn, stub, "v6", [1])

    assert arm.claims == 2
    assert arm.person_subject == 1
    assert arm.person_subject_share == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_a_failed_exchange_is_counted_and_not_charged_claims(
    conn: sqlite3.Connection,
) -> None:
    stub = _Stub([], failure=Failure(kind=FailureKind.TRANSIENT, message="boom", retry_after=None))

    arm = await run_arm(conn, stub, "v6", [1, 2])

    assert arm.failures == 2
    assert arm.claims == 0
    assert arm.input_tokens == 2000


def test_an_arm_with_no_claims_reports_zero_rather_than_dividing(conn: sqlite3.Connection) -> None:
    arm = ArmResult(
        version="v6",
        exchanges=0,
        failures=0,
        claims=0,
        input_tokens=0,
        output_tokens=0,
        claim_chars=0,
        source_chars=0,
        person_subject=0,
    )

    assert arm.claims_per_exchange == 0.0
    assert arm.compression == 0.0
    assert arm.person_subject_share == 0.0
    assert arm.tokens_per_claim == 0.0


def test_the_comparison_formats_both_arms_side_by_side() -> None:
    arms = tuple(
        ArmResult(
            version=version,
            exchanges=2,
            failures=0,
            claims=claims,
            input_tokens=2000,
            output_tokens=1000,
            claim_chars=100,
            source_chars=200,
            person_subject=0,
        )
        for version, claims in (("v5", 12), ("v6", 4))
    )

    lines = format_comparison(PromptComparison(exchange_ids=(1, 2), arms=arms))

    assert "over 2 already-extracted exchanges" in lines[0]
    assert any(line.strip().startswith("v5") for line in lines)
    assert any(line.strip().startswith("v6") for line in lines)


def _pending(conn: sqlite3.Connection, exchange_id: int, message_count: int) -> None:
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash,"
        " extraction_status) VALUES (?, 1, 1, 1, ?, ?, ?, 'quiet_gap', ?, 'pending')",
        (exchange_id, AT.isoformat(), AT.isoformat(), message_count, f"p{exchange_id}"),
    )


def _done(conn: sqlite3.Connection, exchange_id: int, message_count: int) -> None:
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash,"
        " extraction_status) VALUES (?, 1, 1, 1, ?, ?, ?, 'quiet_gap', ?, 'done')",
        (exchange_id, AT.isoformat(), AT.isoformat(), message_count, f"d{exchange_id}"),
    )
    conn.execute(
        "INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at,"
        " mode, outcome) VALUES (?, 'm', 'v5', ?, 'live', 'ok')",
        (exchange_id, AT.isoformat()),
    )


def test_the_sample_matches_the_queue_size_mix_not_the_extracted_mix(
    conn: sqlite3.Connection,
) -> None:
    """The already-extracted population is 90% large exchanges while the queue
    is 76% small ones. Sampling the former measures the wrong population, which
    is the defect that made the gate's figures untransferable (#173)."""
    for index in range(100, 190):
        _pending(conn, index, 2)
    for index in range(190, 200):
        _pending(conn, index, 30)
    for index in range(200, 240):
        _done(conn, index, 2)
    for index in range(240, 280):
        _done(conn, index, 30)

    picked = sample_extracted_exchanges(conn, 10, seed=0)
    placeholders = ",".join("?" * len(picked))
    counts = dict(
        conn.execute(
            "SELECT CASE WHEN message_count <= 2 THEN 'small' ELSE 'large' END AS bucket,"
            f" COUNT(*) FROM exchanges WHERE id IN ({placeholders}) GROUP BY 1",
            picked,
        ).fetchall()
    )

    assert counts["small"] == 9
    assert counts["large"] == 1


def test_the_sample_is_reproducible_for_a_seed(conn: sqlite3.Connection) -> None:
    for index in range(100, 140):
        _pending(conn, index, 2)
    for index in range(200, 260):
        _done(conn, index, 2)

    assert sample_extracted_exchanges(conn, 8, seed=3) == sample_extracted_exchanges(
        conn, 8, seed=3
    )


def test_a_different_seed_draws_a_different_sample(conn: sqlite3.Connection) -> None:
    for index in range(100, 140):
        _pending(conn, index, 2)
    for index in range(200, 260):
        _done(conn, index, 2)

    assert sample_extracted_exchanges(conn, 8, seed=1) != sample_extracted_exchanges(
        conn, 8, seed=2
    )


def test_a_bucket_with_too_few_extracted_exchanges_takes_what_exists(
    conn: sqlite3.Connection,
) -> None:
    """The queue is mostly tiny exchanges but only 303 tiny ones were ever
    extracted, so a request can exceed what a bucket can supply."""
    for index in range(100, 200):
        _pending(conn, index, 2)
    _done(conn, 200, 2)
    for index in range(210, 260):
        _done(conn, index, 30)

    picked = sample_extracted_exchanges(conn, 10, seed=0)

    assert len(picked) == 10
    assert 200 in picked


def test_the_queue_mix_ignores_already_extracted_exchanges(conn: sqlite3.Connection) -> None:
    _pending(conn, 100, 2)
    _done(conn, 200, 30)

    mix = queue_size_mix(conn)

    assert mix[(1, 2)] == 1
    assert mix[(16, 49)] == 0


class _VersionedStub(_Stub):
    def __init__(self, version: str) -> None:
        super().__init__([_claim("The O2 PSU is 150W.")])
        self.prompt_version = version


@pytest.mark.asyncio
async def test_an_arm_refuses_a_label_that_disagrees_with_the_run(
    conn: sqlite3.Connection,
) -> None:
    """An experiment labelled by hand can report two differently named arms
    that ran the same configuration (RULE #215). The label must come from the
    thing that actually rendered the prompt."""
    with pytest.raises(ArmLabelMismatchError):
        await run_arm(conn, _VersionedStub("v5"), "v6", [1])


@pytest.mark.asyncio
async def test_an_arm_accepts_a_label_that_matches_the_run(conn: sqlite3.Connection) -> None:
    arm = await run_arm(conn, _VersionedStub("v6"), "v6", [1])

    assert arm.version == "v6"


def test_the_sample_returns_fewer_than_asked_when_the_corpus_cannot_fill_it(
    conn: sqlite3.Connection,
) -> None:
    """Silent truncation in the other direction: asking for 40 when only 3
    exist must return 3, not pad or loop."""
    _pending(conn, 100, 2)

    picked = sample_extracted_exchanges(conn, 40, seed=0)

    assert len(picked) == 3


def test_the_two_size_bucketings_in_this_repo_are_known_to_differ() -> None:
    """An inventory, not a gate. `runner._size_bucket` strata trial samples by
    1 / 2-5 / 6-20 / 21+, while the prompt comparison matches the pending
    queue's own mix with 1-2 / 3-5 / 6-15 / 16-49 / 50+. They answer the same
    question with different boundaries, so a trial sample and a comparison
    sample are NOT comparable by size. Pinned here so a change to either is
    visible rather than discovered."""
    from infovore.extract.runner import _size_bucket

    assert [_size_bucket(n) for n in (1, 2, 5, 6, 20, 21)] == [
        "1",
        "2-5",
        "2-5",
        "6-20",
        "6-20",
        "21+",
    ]
    assert SIZE_BUCKETS == ((1, 2), (3, 5), (6, 15), (16, 49), (50, 10**9))


@pytest.mark.asyncio
async def test_an_arm_keeps_the_claims_it_measured(
    tmp_path: Path, conn: sqlite3.Connection
) -> None:
    """The aggregate is not the evidence. Discarding the claims makes every
    follow-up question cost another full run: the coverage check on v6 was
    identified as decisive BEFORE the comparison ran and was still lost."""
    out = tmp_path / "arm.jsonl"

    await run_arm(conn, _Stub([_claim("The O2 PSU is 150W.")]), "v6", [1, 2], dump=out)

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [record["exchange_id"] for record in records] == [1, 2]
    assert records[0]["version"] == "v6"
    assert records[0]["claims"][0]["statement"] == "The O2 PSU is 150W."
    assert records[0]["claims"][0]["sources"] == [1]


@pytest.mark.asyncio
async def test_a_dump_records_an_exchange_that_yielded_nothing(
    tmp_path: Path, conn: sqlite3.Connection
) -> None:
    """A zero-claim exchange is a result, not an absence: without the row,
    coverage cannot tell 'nothing found' from 'never attempted'."""
    out = tmp_path / "arm.jsonl"

    await run_arm(conn, _Stub([]), "v6", [1], dump=out)

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records == [{"exchange_id": 1, "version": "v6", "claims": []}]


@pytest.mark.asyncio
async def test_no_dump_is_written_when_none_is_asked_for(conn: sqlite3.Connection) -> None:
    arm = await run_arm(conn, _Stub([_claim("x" * 20)]), "v6", [1])

    assert arm.claims == 1
