import sqlite3
from pathlib import Path

from infovore.db.connection import load_migrations, migrate, open_database

NOW = "2026-01-01T00:00:00+00:00"


def seeded(path: Path) -> sqlite3.Connection:
    conn = open_database(path)
    migrate(conn)
    conn.executescript(
        f"""
        INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,
          created_at, content, ingested_at, raw_json)
          VALUES (1, 10, 9, 1, 'a', '{NOW}', 'x', '{NOW}', '{{}}'),
                 (2, 10, 9, 1, 'a', '{NOW}', 'y', '{NOW}', '{{}}');
        INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,
          ended_at, message_count, grouping_rule, content_hash)
          VALUES (10, 1, 2, '{NOW}', '{NOW}', 2, 'quiet_gap', 'h');
        INSERT INTO prompt_versions VALUES ('v1', 'sha', '{NOW}', '{NOW}');
        INSERT INTO extraction_runs (id, exchange_id, model, prompt_version, started_at, mode,
          outcome) VALUES (1, 1, 'm', 'v1', '{NOW}', 'live', 'ok'),
                          (2, 1, 'm', 'v1', '{NOW}', 'trial', 'ok');
        INSERT INTO claims (id, exchange_id, extraction_run_id, statement, subject, kind,
          confidence, probe_question, permalink, novelty, retracted_at, supersedes_claim_id)
        VALUES
          (1, 1, 1, 'Octane2 PSU is 030-1234-001', 'Octane2', 'fact', 0.9, 'q?', 'p1',
           'unknown', NULL, NULL),
          (2, 1, 1, 'partial one', 'Fuel', 'fact', 0.5, 'q?', 'p2', 'partial', NULL, NULL),
          (3, 1, 1, 'contradicting one', 'IP35', 'fact', 0.7, 'q?', 'p3', 'contradicts',
           NULL, NULL),
          (4, 1, 1, 'already known', 'MIPS', 'fact', 0.9, 'q?', 'p4', 'known', NULL, NULL),
          (5, 1, 1, 'not yet probed', 'Tezro', 'fact', 0.9, 'q?', 'p5', 'unprobed', NULL, NULL),
          (6, 1, 1, 'retracted', 'O2', 'fact', 0.9, 'q?', 'p6', 'unknown', '{NOW}', NULL),
          (7, 1, 2, 'trial only', 'Onyx', 'fact', 0.9, 'q?', 'p7', 'unknown', NULL, NULL),
          (8, 1, 1, 'old version', 'Indy', 'fact', 0.9, 'q?', 'p8', 'unknown', NULL, NULL),
          (9, 1, 1, 'corrected version', 'Indy', 'correction', 0.9, 'q?', 'p9', 'unknown',
           NULL, 8),
          (10, 1, 1, 'kept: superseder retracted', 'Indigo2', 'fact', 0.9, 'q?', 'p10',
           'unknown', NULL, NULL),
          (11, 1, 1, 'retracted superseder', 'Indigo2', 'correction', 0.9, 'q?', 'p11',
           'unknown', '{NOW}', 10),
          (12, 1, 1, 'kept: superseder is trial', 'Crimson', 'fact', 0.9, 'q?', 'p12',
           'unknown', NULL, NULL),
          (13, 1, 2, 'trial superseder', 'Crimson', 'correction', 0.9, 'q?', 'p13',
           'unknown', NULL, 12);
        INSERT INTO claim_sources VALUES (1, 2), (1, 1), (2, 1), (3, 1), (9, 2), (10, 1), (12, 1);
        """
    )
    return conn


def test_lore_contains_exactly_current_live_net_new_claims(tmp_path: Path) -> None:
    conn = seeded(tmp_path / "x.db")
    ids = [row["claim_id"] for row in conn.execute("SELECT claim_id FROM lore ORDER BY claim_id")]
    assert ids == [1, 2, 3, 9, 10, 12]


def test_lore_columns(tmp_path: Path) -> None:
    conn = seeded(tmp_path / "x.db")
    row = conn.execute("SELECT * FROM lore WHERE claim_id = 1").fetchone()
    assert dict(row) == {
        "claim_id": 1,
        "subject": "Octane2",
        "statement": "Octane2 PSU is 030-1234-001",
        "kind": "fact",
        "confidence": 0.9,
        "novelty": "unknown",
        "permalink": "p1",
        "source_message_ids": "1,2",
        "channel_id": 10,
        "extracted_at": NOW,
        "supersedes_claim_id": None,
    }


def test_documented_fts_query_works_against_lore(tmp_path: Path) -> None:
    conn = seeded(tmp_path / "x.db")
    rows = conn.execute(
        "SELECT lore.claim_id FROM claims_fts"
        " JOIN lore ON lore.claim_id = claims_fts.rowid"
        " WHERE claims_fts MATCH ? ORDER BY bm25(claims_fts, 2.0, 1.0)",
        ('"030-1234-001"',),
    ).fetchall()
    assert [row[0] for row in rows] == [1]


def test_user_version_is_the_latest_migration(tmp_path: Path) -> None:
    conn = seeded(tmp_path / "x.db")
    latest = max(m.version for m in load_migrations())
    assert conn.execute("PRAGMA user_version").fetchone()[0] == latest
    assert latest >= 4


def test_read_only_reader_works_while_a_writer_holds_the_database(tmp_path: Path) -> None:
    path = tmp_path / "x.db"
    writer = seeded(path)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("UPDATE claims SET confidence = 0.1 WHERE id = 1")
    reader = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    assert reader.execute("SELECT confidence FROM lore WHERE claim_id = 1").fetchone()[0] == 0.9
    writer.execute("COMMIT")
    assert reader.execute("SELECT confidence FROM lore WHERE claim_id = 1").fetchone()[0] == 0.1
