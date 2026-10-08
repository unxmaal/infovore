import io
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from infovore.claims.extract import SCHEMA, SYSTEM, render_window, windows
from infovore.claims.redact import pseudonym, redact_conversation
from infovore.cli import ExitCode, main
from infovore.db.annotations import Annotation, record_annotation
from infovore.db.claims_v2 import (
    ClaimIn,
    ExchangeOutcome,
    create_run,
    record_exchange,
    record_review,
)
from infovore.triage.human import HUMAN_SCORER
from tests.claims.seed import SALT, conversation, db, environment, messages_of

NOW = datetime(2026, 1, 1, tzinfo=UTC)
LINES = [
    (11, "Alice Smith", "my Indy runs IRIX 6.5"),
    (22, "bobby", "<@11> try the PROM"),
    (11, "Alice Smith", "thanks, the PROM fixed it"),
]
A, B = pseudonym(11, SALT), pseudonym(22, SALT)
Claim = tuple[str, str, tuple[int, ...]]


def run_cli(tmp_path: Path, argv: list[str], salt: str | None = SALT) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=environment(tmp_path, salt), dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def new_run(conn: sqlite3.Connection) -> int:
    return create_run(
        conn,
        endpoint="e",
        model_alias="m",
        model_id="m",
        model_id_source="alias",
        prompt_hash="h",
        selection="s",
        recipe={},
        now=NOW,
    )


def store(conn: sqlite3.Connection, run: int, exchange: int, claims: list[Claim]) -> list[int]:
    outcome = ExchangeOutcome("ok", None, 1, 0, 0, 0.0)
    record_exchange(conn, run, exchange, outcome, [ClaimIn(*c) for c in claims], [])
    sql = "SELECT id FROM claims_v2 WHERE run_id = ? AND exchange_id = ? ORDER BY id"
    return [r[0] for r in conn.execute(sql, (run, exchange))]


def export(tmp_path: Path, runs: str, *more: str) -> tuple[int, str, list[dict[str, Any]]]:
    target = tmp_path / "outside" / "cases.jsonl"
    argv = ["claims", "export-cases", "--runs", runs, "--out", str(target), *more]
    code, out, _ = run_cli(tmp_path, argv)
    rows = [json.loads(x) for x in target.read_text().splitlines()] if target.exists() else []
    return code, out, rows


def test_exports_one_case_per_reviewed_window(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ex, ids = conversation(conn, LINES, 1)
    other, oids = conversation(conn, [(33, "carol", "hello there")], 2)
    run = new_run(conn)
    c1, _ = store(conn, run, ex, [(A, "Indy runs IRIX", (ids[0], ids[1])), (B, "PROM", (ids[1],))])
    (c3,) = store(conn, run, other, [(pseudonym(33, SALT), "hi", (oids[0],))])
    record_review(conn, c1, "good", NOW)
    record_review(conn, c3, "wrong", NOW)
    conn.commit()
    redacted = redact_conversation(messages_of(conn, ex), SALT)
    transcript = render_window(windows(redacted.lines, 6000)[0])
    conn.close()

    code, out, rows = export(tmp_path, str(run))

    assert code == ExitCode.OK and "2 cases" in out
    assert rows[0] == {
        "id": str(ex),
        "system": SYSTEM,
        "transcript": transcript,
        "schema": SCHEMA,
        "reviews": [{"user": A, "claim": "Indy runs IRIX", "refs": [1, 2], "verdict": "good"}],
    }
    assert rows[1]["id"] == str(other) and rows[1]["reviews"][0]["verdict"] == "wrong"
    assert "Alice" not in json.dumps(rows) and "bobby" not in json.dumps(rows)


def test_latest_review_wins_and_unreviewed_claims_are_omitted(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ex, ids = conversation(conn, LINES, 1)
    run = new_run(conn)
    c1, _ = store(conn, run, ex, [(A, "one", (ids[0],)), (B, "two", (ids[1],))])
    record_review(conn, c1, "wrong", NOW)
    record_review(conn, c1, "good", NOW)
    conn.commit()
    conn.close()

    _, _, rows = export(tmp_path, str(run))

    assert [(r["claim"], r["verdict"]) for r in rows[0]["reviews"]] == [("one", "good")]


def test_an_exchange_without_reviews_is_not_exported(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ex, ids = conversation(conn, LINES, 1)
    store(conn, new_run(conn), ex, [(A, "one", (ids[0],))])
    conn.commit()
    conn.close()

    code, out, rows = export(tmp_path, "1")

    assert code == ExitCode.OK and rows == [] and "0 cases" in out


def test_same_claim_in_two_runs_keeps_the_latest_review(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ex, ids = conversation(conn, LINES, 1)
    r1, r2 = new_run(conn), new_run(conn)
    (c1,) = store(conn, r1, ex, [(A, "same", (ids[0],))])
    c2, c3 = store(conn, r2, ex, [(A, "same", (ids[0],)), (A, "other", (ids[2],))])
    record_review(conn, c2, "good", NOW)
    record_review(conn, c1, "made_up", NOW)
    record_review(conn, c3, "not_useful", NOW)
    conn.commit()
    conn.close()

    _, _, rows = export(tmp_path, f"{r1},{r2}")

    assert len(rows) == 1
    verdicts = {r["claim"]: r["verdict"] for r in rows[0]["reviews"]}
    assert verdicts == {"same": "made_up", "other": "not_useful"}


def test_only_the_given_runs_are_exported(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ex, ids = conversation(conn, LINES, 1)
    r1, r2 = new_run(conn), new_run(conn)
    (c1,) = store(conn, r1, ex, [(A, "one", (ids[0],))])
    (c2,) = store(conn, r2, ex, [(A, "two", (ids[0],))])
    record_review(conn, c1, "good", NOW)
    record_review(conn, c2, "good", NOW)
    conn.commit()
    conn.close()

    _, _, rows = export(tmp_path, str(r2))

    assert [r["claim"] for r in rows[0]["reviews"]] == ["two"]


def test_claims_go_to_the_window_their_refs_fall_in(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ex, ids = conversation(conn, LINES, 1)
    run = new_run(conn)
    c1, c2 = store(conn, run, ex, [(A, "first", (ids[0],)), (A, "last", (ids[2],))])
    record_review(conn, c1, "good", NOW)
    record_review(conn, c2, "wrong", NOW)
    conn.commit()
    conn.close()

    _, _, rows = export(tmp_path, str(run), "--window-chars", "40")

    assert [r["id"] for r in rows] == [f"{ex}/0", f"{ex}/2"]
    assert rows[0]["reviews"][0]["refs"] == [1] and rows[1]["reviews"][0]["refs"] == [3]
    assert "my Indy" in rows[0]["transcript"] and "fixed it" not in rows[0]["transcript"]


def test_a_claim_whose_sources_left_the_transcript_is_skipped(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ex, ids = conversation(conn, [*LINES, (11, "Alice Smith", "x")], 1)
    run = new_run(conn)
    (c1,) = store(conn, run, ex, [(A, "gone", (ids[3],))])
    record_review(conn, c1, "good", NOW)
    conn.execute("UPDATE messages SET content = '' WHERE id = ?", (ids[3],))
    conn.commit()
    conn.close()

    code, out, rows = export(tmp_path, str(run))

    assert code == ExitCode.OK and "0 cases" in out and rows == []


@pytest.mark.parametrize("marker", ["dir", "file"])
def test_refuses_an_output_path_inside_a_git_work_tree(tmp_path: Path, marker: str) -> None:
    db(tmp_path).close()
    repo = tmp_path / "repo"
    repo.mkdir()
    if marker == "dir":
        (repo / ".git").mkdir()
    else:
        (repo / ".git").write_text("gitdir: elsewhere")
    target = repo / "deep" / "cases.jsonl"

    code, _, err = run_cli(
        tmp_path, ["claims", "export-cases", "--runs", "1", "--out", str(target)]
    )

    assert code == ExitCode.CONFIG and "git work tree" in err
    assert not target.exists() and not target.parent.exists()


@pytest.mark.parametrize(
    ("runs", "salt", "message"),
    [("1", None, "INFOVORE_PSEUDONYM_SALT"), ("x", SALT, "--runs"), ("99", SALT, "unknown run 99")],
)
def test_refuses_bad_input(tmp_path: Path, runs: str, salt: str | None, message: str) -> None:
    conn = db(tmp_path)
    new_run(conn)
    conn.commit()
    conn.close()
    target = tmp_path / "out" / "c.jsonl"

    argv = ["claims", "export-cases", "--runs", runs, "--out", str(target)]
    code, _, err = run_cli(tmp_path, argv, salt)

    assert code == ExitCode.CONFIG and message in err
    assert not target.exists()


def test_the_database_is_not_modified(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ex, ids = conversation(conn, LINES, 1)
    run = new_run(conn)
    (c1,) = store(conn, run, ex, [(A, "one", (ids[0],))])
    record_review(conn, c1, "good", NOW)
    conn.commit()
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    before = [list(conn.execute(f"SELECT * FROM {n}")) for n in names]
    conn.close()

    _, _, rows = export(tmp_path, str(run))

    assert len(rows) == 1
    conn = db(tmp_path)
    assert [list(conn.execute(f"SELECT * FROM {n}")) for n in names] == before


def annotate(
    conn: sqlite3.Connection, eid: int, scorer: str, label: str | None, day: int = 0
) -> None:
    note = Annotation("exchange", eid, scorer, 1 + day, "recorded", label=label)
    record_annotation(conn, note, datetime(2026, 2, 1 + day, tzinfo=UTC))


def negatives(tmp_path: Path, *more: str) -> tuple[int, str, list[dict[str, Any]]]:
    target = tmp_path / "outside" / "neg.jsonl"
    argv = ["claims", "export-cases", "--negatives", "--out", str(target), *more]
    code, out, _ = run_cli(tmp_path, argv)
    rows = [json.loads(x) for x in target.read_text().splitlines()] if target.exists() else []
    return code, out, rows


def labelled(
    conn: sqlite3.Connection, base: int, label: str | None, cascade: str | None = None
) -> int:
    lines = [(11, "Alice Smith", f"chatter {base}"), (22, "bobby", "lol")]
    eid, _ = conversation(conn, lines, base, None)
    if label:
        annotate(conn, eid, HUMAN_SCORER, label)
    if cascade:
        verdict = "residue" if cascade == "residue" else "irrelevant"
        annotate(conn, eid, f"relevance_{cascade}", verdict)
    return eid


def test_negatives_export_human_irrelevant_windows_with_their_origin(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ex = labelled(conn, 1, "irrelevant", "residue")
    redacted = redact_conversation(messages_of(conn, ex), SALT)
    transcript = render_window(windows(redacted.lines, 6000)[0])
    conn.commit()
    conn.close()

    code, out, rows = negatives(tmp_path)

    assert code == ExitCode.OK
    assert rows == [
        {
            "id": str(ex),
            "system": SYSTEM,
            "transcript": transcript,
            "schema": SCHEMA,
            "reviews": [],
            "expect_empty": True,
            "basis": "human_irrelevant",
            "origin": "undecided",
        }
    ]
    assert "wrote 1 cases to" in out and "undecided: 1" in out
    assert "Alice" not in json.dumps(rows)


def test_negatives_skip_unlabelled_relevant_and_relabelled(tmp_path: Path) -> None:
    conn = db(tmp_path)
    labelled(conn, 1, None, "residue")
    labelled(conn, 2, "relevant", "residue")
    flipped = labelled(conn, 3, "irrelevant")
    annotate(conn, flipped, HUMAN_SCORER, "relevant", 1)
    conn.commit()
    conn.close()

    code, out, rows = negatives(tmp_path)

    assert code == ExitCode.OK and rows == [] and "wrote 0 cases" in out


def test_negatives_origin_is_the_stage_that_decided_in_the_latest_run(tmp_path: Path) -> None:
    conn = db(tmp_path)
    stages = ["denylist", "no_text", "lexicon", "short_no_tech", "embed"]
    ids = {stage: labelled(conn, n, "irrelevant", stage) for n, stage in enumerate(stages, 1)}
    never = labelled(conn, 6, "irrelevant")
    moved = labelled(conn, 7, "irrelevant", "residue")
    annotate(conn, moved, "relevance_lexicon", None, 3)
    annotate(conn, moved, "relevance_embed", "irrelevant", 3)
    conn.commit()
    conn.close()

    _, out, rows = negatives(tmp_path)

    origin = {r["id"]: r["origin"] for r in rows}
    assert all(origin[str(eid)] == stage for stage, eid in ids.items())
    assert origin[str(never)] == "unsorted" and origin[str(moved)] == "embed"
    assert "embed: 2" in out and "unsorted: 1" in out


def test_negatives_are_windowed_and_drop_excluded_channels(tmp_path: Path) -> None:
    conn = db(tmp_path)
    ex = labelled(conn, 1, "irrelevant")
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (2, 9, NULL, 'food', 'text')"
    )
    hidden = labelled(conn, 2, "irrelevant")
    conn.execute("UPDATE exchanges SET channel_id = 2 WHERE id = ?", (hidden,))
    conn.commit()
    conn.close()
    target = tmp_path / "outside" / "neg.jsonl"
    argv = ["claims", "export-cases", "--negatives", "--out", str(target), "--window-chars", "10"]
    env = environment(tmp_path) | {"INFOVORE_EXCLUDE_CHANNELS": "food"}

    main(argv, environ=env, dotenv_path=None, stdout=io.StringIO(), stderr=io.StringIO())

    ids = [json.loads(x)["id"] for x in target.read_text().splitlines()]
    assert ids == [f"{ex}/0", f"{ex}/1"]


def test_negatives_ignore_labels_on_superseded_exchanges(tmp_path: Path) -> None:
    conn = db(tmp_path)
    old = labelled(conn, 1, "irrelevant")
    current = labelled(conn, 2, "irrelevant")
    conn.execute("UPDATE exchanges SET superseded_by_recipe = 'v2' WHERE id = ?", (old,))
    conn.commit()
    conn.close()

    _, _, rows = negatives(tmp_path)

    assert [r["id"] for r in rows] == [str(current)]


def test_negatives_are_guarded_and_exclusive_with_runs(tmp_path: Path) -> None:
    db(tmp_path).close()
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    target = repo / "n.jsonl"

    argv = ["claims", "export-cases", "--negatives", "--out", str(target)]
    code, _, err = run_cli(tmp_path, argv)
    assert code == ExitCode.CONFIG and "work tree" in err and not target.exists()

    both = ["claims", "export-cases", "--negatives", "--runs", "1", "--out", str(tmp_path / "x")]
    assert run_cli(tmp_path, both)[0] == ExitCode.CONFIG
    neither = ["claims", "export-cases", "--out", str(tmp_path / "x")]
    assert run_cli(tmp_path, neither)[0] == ExitCode.CONFIG
