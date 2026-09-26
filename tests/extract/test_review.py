import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.claims import NewClaim, record_run, register_prompt_version, set_novelty
from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import insert_exchange
from infovore.db.labels import set_label
from infovore.db.raw import upsert_channel
from infovore.extract.review import (
    Review,
    UnknownRunIdsError,
    VersionKey,
    build_review,
    render_review_html,
)
from infovore.rows import (
    ChannelKind,
    ChannelRow,
    ClaimKind,
    ExchangeRow,
    ExtractionRunRow,
    ExtractionStatus,
    GroupingRule,
    Label,
    LabelSource,
    MessageRow,
    Novelty,
    RunMode,
    RunOutcome,
)

GUILD_ID = 500
NOW = datetime(2026, 1, 1, tzinfo=UTC)
V1 = VersionKey("v1", "m")
V2 = VersionKey("v2", "m")


def db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def a_message(
    message_id: int,
    channel_id: int = 1,
    author_id: int = 1,
    author: str = "alice",
    content: str = "chatter",
) -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=channel_id,
        guild_id=GUILD_ID,
        author_id=author_id,
        author_name_at_time=author,
        author_is_bot=False,
        created_at=NOW,
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        deleted_at=None,
        ingested_at=NOW,
        raw_json="{}",
    )


def seed_exchange(
    conn: sqlite3.Connection, channel_id: int, exchange_id_hint: str, messages: list[MessageRow]
) -> int:
    for message in messages:
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " author_is_bot, created_at, content, ingested_at, raw_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                message.id,
                message.channel_id,
                message.guild_id,
                message.author_id,
                message.author_name_at_time,
                int(message.author_is_bot),
                message.created_at.isoformat(),
                message.content,
                message.ingested_at.isoformat(),
                message.raw_json,
            ),
        )
    row = ExchangeRow(
        id=None,
        channel_id=channel_id,
        thread_id=None,
        first_message_id=messages[0].id,
        last_message_id=messages[-1].id,
        started_at=NOW,
        ended_at=NOW,
        message_count=len(messages),
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=exchange_id_hint,
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.DONE,
        retry_count=0,
        last_error=None,
    )
    return insert_exchange(conn, row, [message.id for message in messages])


def seed_run(
    conn: sqlite3.Connection,
    exchange_id: int,
    prompt_version: str,
    claims: list[NewClaim],
    model: str = "m",
    input_tokens: int | None = 10,
    output_tokens: int | None = 5,
    outcome: RunOutcome = RunOutcome.OK,
    error: str | None = None,
) -> tuple[int, list[int]]:
    register_prompt_version(conn, prompt_version, f"sha-{prompt_version}", NOW)
    recorded = record_run(
        conn,
        ExtractionRunRow(
            id=None,
            exchange_id=exchange_id,
            model=model,
            prompt_version=prompt_version,
            started_at=NOW,
            finished_at=NOW,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            mode=RunMode.TRIAL,
            outcome=outcome,
            error=error,
        ),
        claims if outcome is RunOutcome.OK else [],
    )
    return recorded.run_id, list(recorded.claim_ids)


def a_claim(
    exchange_id: int,
    statement: str,
    subject: str = "Octane2",
    kind: ClaimKind = ClaimKind.FACT,
    confidence: float = 0.9,
    source_message_ids: tuple[int, ...] = (1001,),
) -> NewClaim:
    return NewClaim(
        exchange_id=exchange_id,
        statement=statement,
        subject=subject,
        kind=kind,
        confidence=confidence,
        probe_question=f"what about {subject}?",
        permalink=f"https://discord.com/channels/{GUILD_ID}/1/{source_message_ids[0]}",
        supersedes_claim_id=None,
        source_message_ids=source_message_ids,
    )


def test_build_review_raises_on_unknown_run_id(tmp_path: Path) -> None:
    conn = db(tmp_path)
    with pytest.raises(UnknownRunIdsError) as excinfo:
        build_review(conn, [999])
    assert excinfo.value.run_ids == (999,)
    assert "999" in str(excinfo.value)


def test_build_review_single_version_has_no_diff(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, "h1", [a_message(1001)])
    claim = a_claim(exchange_id, "needs a jumper on pin 3")
    run_id, claim_ids = seed_run(conn, exchange_id, "v1", [claim])
    set_novelty(conn, claim_ids[0], Novelty.UNKNOWN, "probe-model", "I don't know", NOW)

    review = build_review(conn, [run_id])

    assert review.versions == (V1,)
    assert len(review.exchanges) == 1
    exchange_review = review.exchanges[0]
    assert exchange_review.exchange_id == exchange_id
    assert exchange_review.diff is None
    assert set(exchange_review.runs) == {V1}
    run = exchange_review.runs[V1]
    assert run.run_id == run_id
    assert run.outcome is RunOutcome.OK
    assert len(run.claims) == 1
    review_claim = run.claims[0]
    assert review_claim.subject == "Octane2"
    assert review_claim.statement == "needs a jumper on pin 3"
    assert review_claim.novelty is Novelty.UNKNOWN
    assert review_claim.probe_answer == "I don't know"
    assert review_claim.source_message_ids == (1001,)
    assert exchange_review.triage_score is None
    assert exchange_review.p_lore is None
    assert exchange_review.triage_reasons == ()
    assert exchange_review.effective_label is None
    summary = review.summaries[V1]
    assert summary.exchanges == 1
    assert summary.claims == 1
    assert summary.claims_per_exchange == 1.0
    assert summary.known_share == 0.0
    assert summary.failed_runs == 0


def test_build_review_two_versions_diff_added_dropped_changed_verdict_shift(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    upsert_channel(
        conn,
        ChannelRow(
            id=1,
            guild_id=GUILD_ID,
            parent_id=None,
            name="general",
            kind=ChannelKind.TEXT,
            archived=False,
            last_backfilled_message_id=None,
        ),
    )
    exchange_id = seed_exchange(
        conn, 1, "h1", [a_message(1001, author="alice"), a_message(1002, author="bob")]
    )

    claim_a = a_claim(
        exchange_id, "needs a jumper on pin 3", subject="Octane2", source_message_ids=(1001,)
    )
    claim_b = a_claim(
        exchange_id,
        "fan spins at 3000rpm",
        subject="Octane2 fan",
        source_message_ids=(1001, 1002),
    )
    claim_d = a_claim(
        exchange_id,
        "uses a PS/2 keyboard",
        subject="Octane2 keyboard",
        source_message_ids=(1001,),
    )
    run1_id, run1_claim_ids = seed_run(
        conn, exchange_id, "v1", [claim_a, claim_b, claim_d], input_tokens=100, output_tokens=50
    )
    set_novelty(conn, run1_claim_ids[0], Novelty.UNKNOWN, "probe-model", "I don't know", NOW)
    set_novelty(conn, run1_claim_ids[1], Novelty.KNOWN, "probe-model", "yes it does", NOW)
    set_novelty(conn, run1_claim_ids[2], Novelty.UNKNOWN, "probe-model", "I don't know", NOW)

    claim_a2 = a_claim(
        exchange_id,
        "needs a jumper on pin 3 to enable fast SCSI",
        subject="Octane2",
        source_message_ids=(1001, 1002),
    )
    claim_c = a_claim(
        exchange_id, "ip30prom 6.5 required", subject="Octane2 PROM", source_message_ids=(1002,)
    )
    claim_d2 = a_claim(
        exchange_id,
        "uses a PS/2 keyboard",
        subject="Octane2 keyboard",
        source_message_ids=(1001,),
    )
    run2_id, run2_claim_ids = seed_run(
        conn, exchange_id, "v2", [claim_a2, claim_c, claim_d2], input_tokens=120, output_tokens=60
    )
    set_novelty(conn, run2_claim_ids[0], Novelty.CONTRADICTS, "probe-model", "no it's fine", NOW)
    set_novelty(conn, run2_claim_ids[2], Novelty.KNOWN, "probe-model", "yes, confirmed", NOW)

    review = build_review(conn, [run1_id, run2_id])

    assert review.versions == (V1, V2)
    assert len(review.exchanges) == 1
    diff = review.exchanges[0].diff
    assert diff is not None
    assert [claim.subject for claim in diff.added] == ["Octane2 PROM"]
    assert [claim.subject for claim in diff.dropped] == ["Octane2 fan"]
    assert len(diff.changed) == 1
    before, after = diff.changed[0]
    assert before.statement == "needs a jumper on pin 3"
    assert after.statement == "needs a jumper on pin 3 to enable fast SCSI"
    assert len(diff.verdict_shifts) == 2
    keyboard_shift = next(
        shift for shift in diff.verdict_shifts if shift.subject == "Octane2 keyboard"
    )
    assert keyboard_shift.before is Novelty.UNKNOWN
    assert keyboard_shift.after is Novelty.KNOWN
    octane_shift = next(shift for shift in diff.verdict_shifts if shift.subject == "Octane2")
    assert octane_shift.before is Novelty.UNKNOWN
    assert octane_shift.after is Novelty.CONTRADICTS

    v1_summary = review.summaries[V1]
    assert v1_summary.claims == 3
    assert v1_summary.known_share == pytest.approx(1 / 3)
    assert v1_summary.verdict_distribution == {
        Novelty.UNKNOWN: 2,
        Novelty.KNOWN: 1,
    }
    assert v1_summary.input_tokens_per_100_exchanges == pytest.approx(10000.0)
    assert v1_summary.output_tokens_per_100_exchanges == pytest.approx(5000.0)

    v2_summary = review.summaries[V2]
    assert v2_summary.claims == 3
    assert v2_summary.known_share == pytest.approx(1 / 3)
    assert v2_summary.verdict_distribution == {
        Novelty.CONTRADICTS: 1,
        Novelty.UNPROBED: 1,
        Novelty.KNOWN: 1,
    }

    output = render_review_html(review)
    assert "Changed" in output
    assert "Verdict shifts" in output
    assert "Octane2 keyboard" in output


def test_build_review_spans_multiple_exchanges_and_partial_version_coverage(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    exchange1 = seed_exchange(conn, 1, "h1", [a_message(1001)])
    exchange2 = seed_exchange(conn, 1, "h2", [a_message(2001)])

    claim1 = a_claim(exchange1, "statement one", source_message_ids=(1001,))
    run1_id, _ = seed_run(conn, exchange1, "v1", [claim1])

    claim2 = a_claim(exchange2, "statement two", source_message_ids=(2001,))
    run2_id, _ = seed_run(conn, exchange2, "v1", [claim2])

    claim3 = a_claim(exchange1, "statement one changed", source_message_ids=(1001,))
    run3_id, _ = seed_run(conn, exchange1, "v2", [claim3])

    review = build_review(conn, [run1_id, run2_id, run3_id])

    assert review.versions == (V1, V2)
    assert {exchange.exchange_id for exchange in review.exchanges} == {exchange1, exchange2}
    exchange1_review = next(e for e in review.exchanges if e.exchange_id == exchange1)
    exchange2_review = next(e for e in review.exchanges if e.exchange_id == exchange2)
    assert set(exchange1_review.runs) == {V1, V2}
    assert exchange1_review.diff is not None
    assert set(exchange2_review.runs) == {V1}
    assert exchange2_review.diff is None

    v1_summary = review.summaries[V1]
    assert v1_summary.exchanges == 2
    assert v1_summary.claims == 2
    assert v1_summary.claims_per_exchange == 1.0

    v2_summary = review.summaries[V2]
    assert v2_summary.exchanges == 1
    assert v2_summary.claims == 1


def test_build_review_records_a_failed_run(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, "h1", [a_message(1001)])
    run_id, _ = seed_run(
        conn, exchange_id, "v1", [], outcome=RunOutcome.FAILED, error="invalid_output: bad json"
    )

    review = build_review(conn, [run_id])

    run = review.exchanges[0].runs[V1]
    assert run.outcome is RunOutcome.FAILED
    assert run.error == "invalid_output: bad json"
    assert run.claims == ()
    summary = review.summaries[V1]
    assert summary.failed_runs == 1
    assert summary.claims == 0
    assert summary.claims_per_exchange == 0.0
    assert summary.known_share == 0.0


def test_render_review_html_escapes_user_content_and_has_no_external_urls(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(
        conn, 1, "h1", [a_message(1001, content="<script>alert(1)</script>")]
    )
    claim = a_claim(exchange_id, "<img src=x onerror=alert(1)>", subject="XSS<subject>")
    run_id, claim_ids = seed_run(conn, exchange_id, "v1", [claim])
    set_novelty(conn, claim_ids[0], Novelty.UNKNOWN, "probe-model", "<b>answer</b>", NOW)

    review = build_review(conn, [run_id])
    output = render_review_html(review)

    assert "<script>alert(1)</script>" not in output
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in output
    assert "<img src=x onerror=alert(1)>" not in output
    assert "XSS<subject>" not in output
    assert "<b>answer</b>" not in output

    import re

    urls = re.findall(r'href="([^"]+)"', output)
    for url in urls:
        assert not url.startswith("http://")
        if url.startswith("https://"):
            assert url.startswith("https://discord.com/")


def test_render_review_html_is_a_single_document_with_no_external_scripts_or_styles(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, "h1", [a_message(1001)])
    claim = a_claim(exchange_id, "needs a jumper on pin 3")
    run_id, _ = seed_run(conn, exchange_id, "v1", [claim])

    review = build_review(conn, [run_id])
    output = render_review_html(review)

    assert "<!DOCTYPE html>" in output
    assert "<style>" in output
    assert "prefers-color-scheme" in output
    assert "cdn." not in output
    assert "<link " not in output
    assert "<script src=" not in output


def test_render_review_html_renders_two_version_diff_sections(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, "h1", [a_message(1001), a_message(1002, author="bob")])
    claim_a = a_claim(exchange_id, "needs a jumper on pin 3", source_message_ids=(1001,))
    claim_b = a_claim(
        exchange_id, "fan spins at 3000rpm", subject="Octane2 fan", source_message_ids=(1002,)
    )
    run1_id, _ = seed_run(conn, exchange_id, "v1", [claim_a, claim_b])

    claim_c = a_claim(
        exchange_id, "ip30prom 6.5 required", subject="Octane2 PROM", source_message_ids=(1002,)
    )
    run2_id, _ = seed_run(conn, exchange_id, "v2", [claim_c])

    review = build_review(conn, [run1_id, run2_id])
    output = render_review_html(review)

    assert "Added" in output
    assert "Dropped" in output
    assert "Octane2 PROM" in output
    assert "Octane2 fan" in output


def test_review_dataclass_is_frozen() -> None:
    review = Review(versions=(), exchanges=(), summaries={})
    with pytest.raises(AttributeError):
        review.versions = ("x",)  # type: ignore[misc,assignment]


def test_build_review_rejects_empty_run_ids(tmp_path: Path) -> None:
    conn = db(tmp_path)
    with pytest.raises(ValueError, match="run_ids must not be empty"):
        build_review(conn, [])


def test_render_review_html_shows_no_differences_when_versions_agree(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, "h1", [a_message(1001)])
    claim = a_claim(exchange_id, "needs a jumper on pin 3")
    run1_id, _ = seed_run(conn, exchange_id, "v1", [claim])
    claim_again = a_claim(exchange_id, "needs a jumper on pin 3")
    run2_id, _ = seed_run(conn, exchange_id, "v2", [claim_again])

    review = build_review(conn, [run1_id, run2_id])
    diff = review.exchanges[0].diff
    assert diff is not None
    assert diff.added == ()
    assert diff.dropped == ()
    assert diff.changed == ()
    assert diff.verdict_shifts == ()

    output = render_review_html(review)
    assert "no differences" in output


def test_build_review_groups_runs_by_prompt_version_and_model(tmp_path: Path) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, "h1", [a_message(1001)])
    claim_local = a_claim(exchange_id, "local model claim")
    claim_claude = a_claim(exchange_id, "claude claim")
    run_local_id, _ = seed_run(conn, exchange_id, "v1", [claim_local], model="local-llama")
    run_claude_id, _ = seed_run(conn, exchange_id, "v1", [claim_claude], model="claude-sonnet-5")

    review = build_review(conn, [run_local_id, run_claude_id])

    local_key = VersionKey("v1", "local-llama")
    claude_key = VersionKey("v1", "claude-sonnet-5")
    assert set(review.versions) == {local_key, claude_key}
    exchange_review = review.exchanges[0]
    assert exchange_review.runs[local_key].claims[0].statement == "local model claim"
    assert exchange_review.runs[claude_key].claims[0].statement == "claude claim"
    assert exchange_review.diff is not None


def test_build_review_exposes_rule_score_p_lore_reasons_and_effective_label(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    exchange_id = seed_exchange(conn, 1, "h1", [a_message(1001)])
    conn.execute(
        "UPDATE exchanges SET triage_score = 0.6, triage_reasons = ?, triage_version = 't1',"
        " p_lore = 0.8 WHERE id = ?",
        ('[["domain_terms", 0.15]]', exchange_id),
    )
    set_label(conn, exchange_id, Label.LORE, LabelSource.HUMAN, None, NOW)
    claim = a_claim(exchange_id, "needs a jumper on pin 3")
    run_id, _ = seed_run(conn, exchange_id, "v1", [claim])

    review = build_review(conn, [run_id])

    exchange_review = review.exchanges[0]
    assert exchange_review.triage_score == 0.6
    assert exchange_review.p_lore == 0.8
    assert exchange_review.triage_reasons == (("domain_terms", 0.15),)
    assert exchange_review.effective_label is Label.LORE

    output = render_review_html(review)
    assert "0.600" in output
    assert "0.800" in output
    assert "domain_terms" in output
    assert "lore" in output
