import io
import sqlite3
from pathlib import Path

from infovore.cli import main
from infovore.db.connection import open_database
from infovore.eval.judge import JUDGE_SCORER, UNCERTAIN, queue_stats
from infovore.eval.slices import GOLD, HOLDOUT
from infovore.triage.human import fit_human, trainable_counts

AT = "2026-10-02T00:00:00+00:00"


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_JUDGE_BACKEND": "fake",
        "INFOVORE_EXCLUDE_CHANNELS": "food",
    }


def _run(argv: list[str], tmp_path: Path) -> str:
    out = io.StringIO()
    main(argv, environ=_env(tmp_path), dotenv_path=None, stdout=out, stderr=io.StringIO())
    return out.getvalue()


def _seed(tmp_path: Path) -> sqlite3.Connection:
    _run(["status"], tmp_path)
    conn = open_database(tmp_path / "infovore.db")
    for channel, name in ((1, "hardware"), (2, "food")):
        conn.execute(
            "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
            " VALUES (?, 1, NULL, ?, 'text')",
            (channel, name),
        )
    layout = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 1, 8: 1}
    for eid, channel in layout.items():
        conn.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
            " started_at, ended_at, message_count, grouping_rule, content_hash)"
            " VALUES (?, ?, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
            (eid, channel, eid, eid, AT, AT, f"h{eid}"),
        )
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, ?, 1, 5, 'hal', ?, 'hi', ?, '{}')",
            (eid, channel, AT, AT),
        )
        conn.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
            (eid, eid),
        )
    for name, eid in ((GOLD, 4), (HOLDOUT, 5)):
        conn.execute(
            "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
            " VALUES (?, ?, 1, 'test', 0, ?)",
            (name, eid, AT),
        )
    labels = [
        (1, "relevant"),
        (2, "relevant"),
        (3, "irrelevant"),
        (4, "relevant"),
        (5, "irrelevant"),
        (6, "relevant"),
        (7, "bad_grouping"),
        (8, "relevant"),
        (8, "irrelevant"),
    ]
    for eid, label in labels:
        conn.execute(
            "INSERT INTO annotations (subject_kind, subject_id, scorer, scorer_version,"
            " reproducibility, label, source_ref, created_at)"
            " VALUES ('exchange', ?, ?, 2, 'recorded', ?, ?, ?)",
            (eid, JUDGE_SCORER, label, f"judge:{UNCERTAIN}:{eid}", AT),
        )
    conn.commit()
    return conn


def test_report_header_and_trainer_agree_on_trainable_counts(tmp_path: Path) -> None:
    conn = _seed(tmp_path)
    excluded = frozenset({"food"})

    fit = fit_human(conn, minimum=1, exclude_channels=excluded).report
    report = _run(["judge", "report"], tmp_path)
    header = queue_stats(conn, UNCERTAIN, [], excluded)

    assert trainable_counts(conn, excluded) == (2, 2)
    assert (fit.relevant, fit.irrelevant) == (2, 2)
    assert "trainable relevant: 2\n" in report
    assert "trainable irrelevant: 2\n" in report
    assert "needed: 198 more relevant to reach 200" in report
    assert header["irrelevant_needed"] == 198
    assert header["relevant_needed"] == 198
