import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from infovore.db.claims import (
    NewClaim,
    PromptVersionConflictError,
    RecordedRun,
    UnknownPromptVersionError,
    claim_source_ids,
    claims_for_run,
    claims_for_runs_needing_probe,
    claims_needing_probe,
    get_claim,
    live_prompt_version,
    promote_prompt_version,
    record_run,
    register_prompt_version,
    related_claims,
    retract_claim,
    retract_claims_with_all_sources_deleted,
    retract_claims_with_all_sources_opted_out,
    set_novelty,
    set_probe_error,
    unprobed_claims,
)
from infovore.db.codec import to_db_time
from infovore.db.connection import migrate, open_database
from infovore.rows import ClaimKind, ExtractionRunRow, Novelty, RunMode, RunOutcome

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def insert_message(
    conn: sqlite3.Connection,
    message_id: int,
    author_id: int = 1,
    deleted_at: datetime | None = None,
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, deleted_at, ingested_at, raw_json)"
        " VALUES (?, 1, 1, ?, 'a', ?, 'x', ?, ?, '{}')",
        (message_id, author_id, to_db_time(NOW), to_db_time(deleted_at), to_db_time(NOW)),
    )


def insert_exchange(conn: sqlite3.Connection, exchange_id: int = 1, message_id: int = 1) -> None:
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, 1, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
        (exchange_id, message_id, message_id, to_db_time(NOW), to_db_time(NOW), f"h{exchange_id}"),
    )


def a_run(
    exchange_id: int = 1,
    mode: RunMode = RunMode.LIVE,
    outcome: RunOutcome = RunOutcome.OK,
    prompt_version: str = "v1",
    error: str | None = None,
) -> ExtractionRunRow:
    return ExtractionRunRow(
        id=None,
        exchange_id=exchange_id,
        model="m",
        prompt_version=prompt_version,
        started_at=NOW,
        finished_at=NOW,
        input_tokens=1,
        output_tokens=1,
        mode=mode,
        outcome=outcome,
        error=error,
    )


def a_claim(
    exchange_id: int = 1,
    statement: str = "Octane2 needs PROM 6.5",
    subject: str = "IP30",
    supersedes_claim_id: int | None = None,
    source_message_ids: tuple[int, ...] = (1,),
) -> NewClaim:
    return NewClaim(
        exchange_id=exchange_id,
        statement=statement,
        subject=subject,
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question="what prom does the octane2 need?",
        permalink="https://discord.com/channels/1/1/1",
        supersedes_claim_id=supersedes_claim_id,
        source_message_ids=source_message_ids,
    )


def setup_basic(conn: sqlite3.Connection) -> None:
    insert_message(conn, 1)
    insert_exchange(conn, 1, 1)
    register_prompt_version(conn, "v1", "sha", NOW)


def test_register_prompt_version_is_idempotent_for_same_sha(tmp_path: Path) -> None:
    conn = db(tmp_path)
    register_prompt_version(conn, "v1", "sha", NOW)
    register_prompt_version(conn, "v1", "sha", NOW)
    row = conn.execute("SELECT text_sha256 FROM prompt_versions WHERE version = 'v1'").fetchone()
    assert row[0] == "sha"


def test_register_prompt_version_conflict_raises(tmp_path: Path) -> None:
    conn = db(tmp_path)
    register_prompt_version(conn, "v1", "sha", NOW)
    with pytest.raises(PromptVersionConflictError):
        register_prompt_version(conn, "v1", "other-sha", NOW)


def test_promote_unknown_prompt_version_raises(tmp_path: Path) -> None:
    conn = db(tmp_path)
    with pytest.raises(UnknownPromptVersionError):
        promote_prompt_version(conn, "nope", NOW)


def test_live_prompt_version_is_most_recently_promoted(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert live_prompt_version(conn) is None
    register_prompt_version(conn, "v1", "sha1", NOW)
    register_prompt_version(conn, "v2", "sha2", NOW)
    promote_prompt_version(conn, "v1", NOW)
    assert live_prompt_version(conn) == "v1"
    promote_prompt_version(conn, "v2", NOW + timedelta(seconds=1))
    assert live_prompt_version(conn) == "v2"


def test_record_run_inserts_run_and_claims_with_sources(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    result = record_run(conn, a_run(), [a_claim(source_message_ids=(1, 2))])
    assert isinstance(result, RecordedRun)
    assert result.run_id > 0
    assert len(result.claim_ids) == 1
    claim = get_claim(conn, result.claim_ids[0])
    assert claim is not None
    assert claim.exchange_id == 1
    assert claim.extraction_run_id == result.run_id
    assert claim.novelty == Novelty.UNPROBED
    assert claim_source_ids(conn, result.claim_ids[0]) == [1, 2]
    assert claims_for_run(conn, result.run_id) == [claim]


def test_record_run_failed_outcome_with_claims_raises(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    with pytest.raises(ValueError, match="failed"):
        record_run(conn, a_run(outcome=RunOutcome.FAILED), [a_claim()])
    assert conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0] == 0


def test_record_run_failed_outcome_without_claims_succeeds(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    result = record_run(conn, a_run(outcome=RunOutcome.FAILED, error="boom"), [])
    assert result.claim_ids == ()
    assert claims_for_run(conn, result.run_id) == []


def test_record_run_rejects_claim_with_no_sources(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    with pytest.raises(ValueError, match="source"):
        record_run(conn, a_run(), [a_claim(source_message_ids=())])
    assert conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0] == 0


def test_record_run_supersedes_link_resolves(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    first = record_run(conn, a_run(), [a_claim()])
    original_id = first.claim_ids[0]
    second = record_run(conn, a_run(), [a_claim(supersedes_claim_id=original_id)])
    superseding = get_claim(conn, second.claim_ids[0])
    assert superseding is not None
    assert superseding.supersedes_claim_id == original_id


def test_record_run_rolls_back_everything_on_failure(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    with pytest.raises(sqlite3.IntegrityError):
        record_run(
            conn,
            a_run(),
            [a_claim(), a_claim(source_message_ids=(999,))],
        )
    assert conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0


def test_get_claim_returns_none_for_unknown_id(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert get_claim(conn, 999) is None


def test_claim_source_ids_and_claims_for_run_empty(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    result = record_run(conn, a_run(), [])
    assert claims_for_run(conn, result.run_id) == []
    assert claim_source_ids(conn, 999) == []


def test_unprobed_claims_excludes_retracted_and_respects_mode_and_limit(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_message(conn, 3)
    insert_exchange(conn, 2, 2)
    live_result = record_run(conn, a_run(), [a_claim(statement="a", subject="s")])
    trial_result = record_run(
        conn,
        a_run(exchange_id=2, mode=RunMode.TRIAL),
        [a_claim(exchange_id=2, statement="b", subject="s")],
    )
    retracted_result = record_run(conn, a_run(), [a_claim(statement="c", subject="s")])
    retract_claim(conn, retracted_result.claim_ids[0], "sources_deleted", NOW)

    all_unprobed = unprobed_claims(conn, limit=10)
    ids = {c.id for c in all_unprobed}
    assert live_result.claim_ids[0] in ids
    assert trial_result.claim_ids[0] in ids
    assert retracted_result.claim_ids[0] not in ids

    live_only = unprobed_claims(conn, limit=10, mode=RunMode.LIVE)
    live_only_ids = {c.id for c in live_only}
    assert live_result.claim_ids[0] in live_only_ids
    assert trial_result.claim_ids[0] not in live_only_ids

    limited = unprobed_claims(conn, limit=1, mode=RunMode.LIVE)
    assert len(limited) == 1


def test_claims_needing_probe(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_message(conn, 3)
    insert_exchange(conn, 2, 2)
    insert_exchange(conn, 3, 3)
    unprobed_result = record_run(conn, a_run(), [a_claim(statement="a", subject="s")])
    stale_result = record_run(
        conn, a_run(exchange_id=2), [a_claim(exchange_id=2, statement="b", subject="s")]
    )
    set_novelty(conn, stale_result.claim_ids[0], Novelty.KNOWN, "old-model", "answer", NOW)
    fresh_result = record_run(
        conn, a_run(exchange_id=3), [a_claim(exchange_id=3, statement="c", subject="s")]
    )
    set_novelty(conn, fresh_result.claim_ids[0], Novelty.KNOWN, "new-model", "answer", NOW)
    retracted_result = record_run(
        conn, a_run(exchange_id=3), [a_claim(exchange_id=3, statement="d", subject="s")]
    )
    retract_claim(conn, retracted_result.claim_ids[0], "sources_deleted", NOW)

    needing = {c.id for c in claims_needing_probe(conn, "new-model", limit=10)}
    assert unprobed_result.claim_ids[0] in needing
    assert stale_result.claim_ids[0] in needing
    assert fresh_result.claim_ids[0] not in needing
    assert retracted_result.claim_ids[0] not in needing

    limited = claims_needing_probe(conn, "new-model", limit=1)
    assert len(limited) == 1


def test_set_probe_error_leaves_novelty_and_set_novelty_clears_error(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    result = record_run(conn, a_run(), [a_claim()])
    claim_id = result.claim_ids[0]
    set_probe_error(conn, claim_id, "timeout")
    claim = get_claim(conn, claim_id)
    assert claim is not None
    assert claim.probe_error == "timeout"
    assert claim.novelty == Novelty.UNPROBED

    set_novelty(conn, claim_id, Novelty.CONTRADICTS, "m", "answer", NOW)
    claim = get_claim(conn, claim_id)
    assert claim is not None
    assert claim.novelty == Novelty.CONTRADICTS
    assert claim.probe_model == "m"
    assert claim.probe_answer == "answer"
    assert claim.probed_at == NOW
    assert claim.probe_error is None


def test_retract_claim_unknown_returns_false(tmp_path: Path) -> None:
    conn = db(tmp_path)
    assert retract_claim(conn, 999, "sources_deleted", NOW) is False


def test_retract_claim_already_retracted_keeps_first_retraction(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    result = record_run(conn, a_run(), [a_claim()])
    claim_id = result.claim_ids[0]
    assert retract_claim(conn, claim_id, "sources_deleted", NOW) is True
    later = NOW + timedelta(days=1)
    assert retract_claim(conn, claim_id, "sources_opted_out", later) is True
    claim = get_claim(conn, claim_id)
    assert claim is not None
    assert claim.retraction_reason == "sources_deleted"
    assert claim.retracted_at == NOW


def test_retract_claims_with_all_sources_deleted(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_message(conn, 3, deleted_at=NOW)
    result_all_deleted = record_run(conn, a_run(), [a_claim(source_message_ids=(1,))])
    result_partial = record_run(conn, a_run(), [a_claim(source_message_ids=(2, 3))])

    assert retract_claims_with_all_sources_deleted(conn, NOW) == []
    conn.execute("UPDATE messages SET deleted_at = ? WHERE id = 1", (to_db_time(NOW),))

    retracted = retract_claims_with_all_sources_deleted(conn, NOW)
    assert retracted == [result_all_deleted.claim_ids[0]]
    claim = get_claim(conn, result_all_deleted.claim_ids[0])
    assert claim is not None
    assert claim.retraction_reason == "sources_deleted"
    partial_claim = get_claim(conn, result_partial.claim_ids[0])
    assert partial_claim is not None
    assert partial_claim.retracted_at is None

    assert retract_claims_with_all_sources_deleted(conn, NOW) == []


def test_retract_claims_with_all_sources_opted_out(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2, author_id=42)
    insert_message(conn, 3, author_id=7)
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (42, ?)", (to_db_time(NOW),))
    all_opted_out = record_run(conn, a_run(), [a_claim(source_message_ids=(2,))])
    mixed = record_run(conn, a_run(), [a_claim(source_message_ids=(2, 3))])

    retracted = retract_claims_with_all_sources_opted_out(conn, NOW)
    assert retracted == [all_opted_out.claim_ids[0]]
    claim = get_claim(conn, all_opted_out.claim_ids[0])
    assert claim is not None
    assert claim.retraction_reason == "sources_opted_out"
    mixed_claim = get_claim(conn, mixed.claim_ids[0])
    assert mixed_claim is not None
    assert mixed_claim.retracted_at is None

    assert retract_claims_with_all_sources_opted_out(conn, NOW) == []


def test_related_claims_matches_and_weights_subject_over_statement(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_exchange(conn, 2, 2)
    subject_hit = record_run(
        conn, a_run(), [a_claim(statement="unrelated text here", subject="Octane2")]
    )
    statement_hit = record_run(
        conn,
        a_run(exchange_id=2),
        [a_claim(exchange_id=2, statement="mentions Octane2 in passing", subject="other")],
    )
    results = related_claims(conn, "Octane2", limit=10)
    ids = [c.id for c in results]
    assert subject_hit.claim_ids[0] in ids
    assert statement_hit.claim_ids[0] in ids
    assert ids.index(subject_hit.claim_ids[0]) < ids.index(statement_hit.claim_ids[0])


def test_related_claims_excludes_retracted_and_trial(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_message(conn, 3)
    insert_exchange(conn, 2, 2)
    insert_exchange(conn, 3, 3)
    retracted_result = record_run(
        conn, a_run(), [a_claim(statement="Zeta widget info", subject="s")]
    )
    retract_claim(conn, retracted_result.claim_ids[0], "sources_deleted", NOW)
    record_run(
        conn,
        a_run(exchange_id=2, mode=RunMode.TRIAL),
        [a_claim(exchange_id=2, statement="Zeta widget trial", subject="s")],
    )
    live_result = record_run(
        conn,
        a_run(exchange_id=3),
        [a_claim(exchange_id=3, statement="Zeta widget live", subject="s")],
    )
    ids = [c.id for c in related_claims(conn, "Zeta widget", limit=10)]
    assert ids == [live_result.claim_ids[0]]


def test_related_claims_respects_limit(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    for i in range(2, 5):
        insert_message(conn, i)
        insert_exchange(conn, i, i)
    for i in range(2, 5):
        record_run(
            conn,
            a_run(exchange_id=i),
            [a_claim(exchange_id=i, statement=f"Widget model {i}", subject="s")],
        )
    results = related_claims(conn, "Widget", limit=2)
    assert len(results) == 2


def test_related_claims_empty_and_punctuation_only_text_returns_empty_list(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    record_run(conn, a_run(), [a_claim()])
    assert related_claims(conn, "", limit=10) == []
    assert related_claims(conn, "!!! ??? ---", limit=10) == []
    assert related_claims(conn, '"quoted" text', limit=10) is not None


@given(st.text())
def test_related_claims_never_raises_for_arbitrary_text(text: str) -> None:
    conn = open_database(":memory:")
    migrate(conn)
    result = related_claims(conn, text, limit=10)
    assert isinstance(result, list)


def test_related_claims_matches_hyphenated_part_numbers_as_a_single_token(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    record_run(
        conn, a_run(), [a_claim(statement="Replace assembly 030-1234-001 on failure", subject="s")]
    )
    results = related_claims(
        conn, "Is part 030-1234-001 right? Confirm 030-1234-001 please.", limit=10
    )
    assert len(results) == 1


def test_related_claims_matches_dotted_version_with_sentence_final_period(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    record_run(conn, a_run(), [a_claim(statement="xyzzy142 uses 6.5.22 internally", subject="s")])
    results = related_claims(conn, "Only 6.5.22.", limit=10)
    assert len(results) == 1


def test_related_claims_matches_paths_with_trailing_period(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    record_run(
        conn, a_run(), [a_claim(statement="xyzzy142 uses /usr/sbin/inst internally", subject="s")]
    )
    results = related_claims(conn, "Only /usr/sbin/inst.", limit=10)
    assert len(results) == 1


def test_claims_for_runs_needing_probe_scopes_to_runs_and_excludes_retracted(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_message(conn, 3)
    insert_exchange(conn, 2, 2)
    insert_exchange(conn, 3, 3)
    in_scope = record_run(conn, a_run(), [a_claim(statement="a", subject="s")])
    also_in_scope = record_run(
        conn, a_run(exchange_id=2), [a_claim(exchange_id=2, statement="b", subject="s")]
    )
    out_of_scope = record_run(
        conn, a_run(exchange_id=3), [a_claim(exchange_id=3, statement="c", subject="s")]
    )
    retracted_in_scope = record_run(conn, a_run(), [a_claim(statement="d", subject="s")])
    retract_claim(conn, retracted_in_scope.claim_ids[0], "sources_deleted", NOW)

    run_ids = (in_scope.run_id, also_in_scope.run_id, retracted_in_scope.run_id)
    result = claims_for_runs_needing_probe(conn, run_ids, None, limit=10)
    ids = {c.id for c in result}
    assert in_scope.claim_ids[0] in ids
    assert also_in_scope.claim_ids[0] in ids
    assert out_of_scope.claim_ids[0] not in ids
    assert retracted_in_scope.claim_ids[0] not in ids


def test_claims_for_runs_needing_probe_filters_by_probe_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_exchange(conn, 2, 2)
    unprobed_result = record_run(conn, a_run(), [a_claim(statement="a", subject="s")])
    stale_result = record_run(
        conn, a_run(exchange_id=2), [a_claim(exchange_id=2, statement="b", subject="s")]
    )
    set_novelty(conn, stale_result.claim_ids[0], Novelty.KNOWN, "old-model", "answer", NOW)

    run_ids = (unprobed_result.run_id, stale_result.run_id)
    needing = {c.id for c in claims_for_runs_needing_probe(conn, run_ids, "new-model", limit=10)}
    assert unprobed_result.claim_ids[0] in needing
    assert stale_result.claim_ids[0] in needing

    set_novelty(conn, stale_result.claim_ids[0], Novelty.KNOWN, "new-model", "answer", NOW)
    still_needing = {
        c.id for c in claims_for_runs_needing_probe(conn, run_ids, "new-model", limit=10)
    }
    assert stale_result.claim_ids[0] not in still_needing
    assert unprobed_result.claim_ids[0] in still_needing


def test_claims_for_runs_needing_probe_without_probe_model_is_unprobed_only(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_exchange(conn, 2, 2)
    unprobed_result = record_run(conn, a_run(), [a_claim(statement="a", subject="s")])
    probed_result = record_run(
        conn, a_run(exchange_id=2), [a_claim(exchange_id=2, statement="b", subject="s")]
    )
    set_novelty(conn, probed_result.claim_ids[0], Novelty.KNOWN, "any-model", "answer", NOW)

    run_ids = (unprobed_result.run_id, probed_result.run_id)
    needing = {c.id for c in claims_for_runs_needing_probe(conn, run_ids, None, limit=10)}
    assert unprobed_result.claim_ids[0] in needing
    assert probed_result.claim_ids[0] not in needing


def test_claims_for_runs_needing_probe_respects_limit_and_empty_run_ids(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_exchange(conn, 2, 2)
    first = record_run(conn, a_run(), [a_claim(statement="a", subject="s")])
    second = record_run(
        conn, a_run(exchange_id=2), [a_claim(exchange_id=2, statement="b", subject="s")]
    )
    run_ids = (first.run_id, second.run_id)
    limited = claims_for_runs_needing_probe(conn, run_ids, None, limit=1)
    assert len(limited) == 1
    assert claims_for_runs_needing_probe(conn, (), None, limit=10) == []


def test_unprobed_claims_excludes_failed_unless_include_failed(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_exchange(conn, 2, 2)
    ok_result = record_run(conn, a_run(), [a_claim(statement="a", subject="s")])
    failed_result = record_run(
        conn, a_run(exchange_id=2), [a_claim(exchange_id=2, statement="b", subject="s")]
    )
    set_probe_error(conn, failed_result.claim_ids[0], "boom")

    default_ids = {c.id for c in unprobed_claims(conn, limit=10)}
    assert ok_result.claim_ids[0] in default_ids
    assert failed_result.claim_ids[0] not in default_ids

    with_failed_ids = {c.id for c in unprobed_claims(conn, limit=10, include_failed=True)}
    assert ok_result.claim_ids[0] in with_failed_ids
    assert failed_result.claim_ids[0] in with_failed_ids


def test_claims_needing_probe_excludes_failed_unless_include_failed(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_exchange(conn, 2, 2)
    ok_result = record_run(conn, a_run(), [a_claim(statement="a", subject="s")])
    failed_result = record_run(
        conn, a_run(exchange_id=2), [a_claim(exchange_id=2, statement="b", subject="s")]
    )
    set_probe_error(conn, failed_result.claim_ids[0], "boom")

    default_ids = {c.id for c in claims_needing_probe(conn, "m", limit=10)}
    assert ok_result.claim_ids[0] in default_ids
    assert failed_result.claim_ids[0] not in default_ids

    with_failed_ids = {c.id for c in claims_needing_probe(conn, "m", limit=10, include_failed=True)}
    assert ok_result.claim_ids[0] in with_failed_ids
    assert failed_result.claim_ids[0] in with_failed_ids


def test_claims_for_runs_needing_probe_excludes_failed_unless_include_failed(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    insert_message(conn, 2)
    insert_exchange(conn, 2, 2)
    ok_result = record_run(conn, a_run(), [a_claim(statement="a", subject="s")])
    failed_result = record_run(
        conn, a_run(exchange_id=2), [a_claim(exchange_id=2, statement="b", subject="s")]
    )
    set_probe_error(conn, failed_result.claim_ids[0], "boom")
    run_ids = (ok_result.run_id, failed_result.run_id)

    default_ids = {c.id for c in claims_for_runs_needing_probe(conn, run_ids, None, limit=10)}
    assert ok_result.claim_ids[0] in default_ids
    assert failed_result.claim_ids[0] not in default_ids

    with_failed_ids = {
        c.id
        for c in claims_for_runs_needing_probe(conn, run_ids, None, limit=10, include_failed=True)
    }
    assert ok_result.claim_ids[0] in with_failed_ids
    assert failed_result.claim_ids[0] in with_failed_ids


def test_related_claims_matches_non_ascii_words(tmp_path: Path) -> None:
    conn = db(tmp_path)
    setup_basic(conn)
    record_run(
        conn, a_run(), [a_claim(statement="check chassis Größe before shipping", subject="s")]
    )
    results = related_claims(conn, "Was ist die Größe?", limit=10)
    assert len(results) == 1
