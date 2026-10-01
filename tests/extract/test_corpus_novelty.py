import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.extract.schema import (
    ClaimOutV6,
    ExtractionOutV6,
    InvalidExtractionError,
    output_model_for,
    parse_extraction,
)

AT = datetime(2026, 1, 1, tzinfo=UTC)


def test_v5_still_requires_a_probe_question() -> None:
    """v5 is promoted and its 12,656 recorded runs must stay reproducible, so
    it keeps asking for the field even though nothing consumes it any more."""
    required = output_model_for("v5").model_json_schema()["$defs"]["ClaimOut"]["required"]

    assert set(required) == {
        "statement",
        "subject",
        "kind",
        "confidence",
        "probe_question",
        "sources",
        "supersedes",
    }


def test_v6_does_not_ask_the_model_for_a_probe_question() -> None:
    """probe_question exists only to feed the novelty probe. Once novelty
    means corpus novelty the field is 34.8% of the claim payload bought for
    nothing (issue #165)."""
    schema = output_model_for("v6").model_json_schema()

    assert "probe_question" not in schema["$defs"]["ClaimOutV6"]["properties"]
    assert output_model_for("v6") is ExtractionOutV6


def test_a_v6_claim_parses_without_a_probe_question() -> None:
    payload = json.dumps(
        {
            "claims": [
                {
                    "statement": "The SGI O2 power supply accepts a Meanwell modular unit.",
                    "subject": "SGI O2 power supply",
                    "kind": "fact",
                    "confidence": 0.9,
                    "sources": ["m1"],
                    "supersedes": None,
                }
            ]
        }
    )

    claims = parse_extraction(payload, {"m1": 1}, set(), version="v6")

    assert len(claims) == 1
    assert claims[0].statement.startswith("The SGI O2")


def test_a_v6_claim_records_an_empty_probe_question() -> None:
    """`claims.probe_question` is NOT NULL and rebuilding a 77k-row table with
    FTS triggers to relax it is not worth the risk, so the v6 path writes an
    empty string: no probe question was asked for."""
    payload = json.dumps(
        {
            "claims": [
                {
                    "statement": "IRIX 6.5.22m needs patchSG0007188 first.",
                    "subject": "IRIX 6.5.22m",
                    "kind": "fact",
                    "confidence": 0.9,
                    "sources": ["m1"],
                    "supersedes": None,
                }
            ]
        }
    )

    claims = parse_extraction(payload, {"m1": 1}, set(), version="v6")

    assert claims[0].probe_question == ""


def test_v6_rejects_a_probe_question_it_was_not_asked_for() -> None:
    payload = json.dumps(
        {
            "claims": [
                {
                    "statement": "The Indy maxes out at 256MB of RAM.",
                    "subject": "SGI Indy",
                    "kind": "fact",
                    "confidence": 0.9,
                    "probe_question": "How much RAM does an Indy take?",
                    "sources": ["m1"],
                    "supersedes": None,
                }
            ]
        }
    )

    with pytest.raises(InvalidExtractionError):
        parse_extraction(payload, {"m1": 1}, set(), version="v6")


def test_v6_keeps_every_other_claim_validator() -> None:
    blank_subject = json.dumps(
        {
            "claims": [
                {
                    "statement": "The Indy maxes out at 256MB.",
                    "subject": "   ",
                    "kind": "fact",
                    "confidence": 0.9,
                    "sources": ["m1"],
                    "supersedes": None,
                }
            ]
        }
    )

    with pytest.raises(InvalidExtractionError):
        parse_extraction(blank_subject, {"m1": 1}, set(), version="v6")


def test_claim_out_v6_has_no_probe_question_field() -> None:
    assert "probe_question" not in ClaimOutV6.model_fields


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    return connection


def _claim(conn: sqlite3.Connection, claim_id: int, novelty: str) -> None:
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind) VALUES (?, 1, NULL, ?, 'text')",
        (claim_id, f"c{claim_id}"),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, ?, 1, 1, ?, ?, 1, 'quiet_gap', ?)",
        (claim_id, claim_id, AT.isoformat(), AT.isoformat(), f"h{claim_id}"),
    )
    conn.execute(
        "INSERT INTO prompt_versions (version, text_sha256, created_at)"
        " VALUES (?, 'sha', ?) ON CONFLICT DO NOTHING",
        ("v6", AT.isoformat()),
    )
    conn.execute(
        "INSERT INTO extraction_runs (id, exchange_id, model, prompt_version, started_at,"
        " mode, outcome) VALUES (?, ?, 'm', 'v6', ?, 'live', 'ok')",
        (claim_id, claim_id, AT.isoformat()),
    )
    conn.execute(
        "INSERT INTO claims (id, exchange_id, extraction_run_id, statement, subject, kind,"
        " confidence, probe_question, permalink, novelty)"
        " VALUES (?, ?, ?, 's', 'sub', 'fact', 0.9, '', 'p', ?)",
        (claim_id, claim_id, claim_id, novelty),
    )


def test_the_lore_view_publishes_unprobed_claims(conn: sqlite3.Connection) -> None:
    """Retiring the probe means most claims stay `unprobed` forever. The old
    view required a verdict, so it would have published nothing (issue #165)."""
    _claim(conn, 1, "unprobed")

    rows = conn.execute("SELECT claim_id FROM lore").fetchall()

    assert [row["claim_id"] for row in rows] == [1]


def test_the_lore_view_still_suppresses_known_claims(conn: sqlite3.Connection) -> None:
    """The 3,048 `known` verdicts already paid for keep working: the new rule
    is `novelty != 'known'`, not "publish everything"."""
    _claim(conn, 2, "known")

    assert conn.execute("SELECT COUNT(*) AS n FROM lore").fetchone()["n"] == 0


@pytest.mark.parametrize("novelty", ["unknown", "partial", "contradicts"])
def test_the_lore_view_keeps_publishing_existing_verdicts(
    conn: sqlite3.Connection, novelty: str
) -> None:
    _claim(conn, 3, novelty)

    assert conn.execute("SELECT COUNT(*) AS n FROM lore").fetchone()["n"] == 1


def test_an_unknown_prompt_version_has_no_output_schema() -> None:
    with pytest.raises(InvalidExtractionError):
        output_model_for("v99")
