import io
import json
import os
import re
import signal
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from infovore.claims.extract import Transport
from infovore.cli import ExitCode, main
from infovore.db.wiki_articles import sections_for
from infovore.db.wiki_tags import create_tag_run, record_tags
from infovore.wiki import write_command
from tests.claims.seed import NOW, db, environment
from tests.wiki.seed import add_claim, add_exchange, wiki_db

BASE = ["wiki", "write", "--tag-run", "1", "--endpoint", "http://fake", "--model", "m"]
INVENTED = "Gadget numbers quantum flux capacitor"
STATE = {"poison": True}


def fake_post(endpoint: str) -> Transport:
    def send(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], float]:
        text = payload["messages"][1]["content"]
        if STATE["poison"] and "poison" in text:
            reply: Mapping[str, Any] = {"choices": [{"finish_reason": "length", "message": {}}]}
            return reply, 0.0
        first = re.findall(r"^1\. (.*)$", text, re.M)[0]
        body = {"s": [[first, [1]], [INVENTED, [1]]]}
        content = {"finish_reason": "stop", "message": {"content": json.dumps(body)}}
        return {"choices": [content]}, 0.0

    return send


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    STATE["poison"] = True
    monkeypatch.setattr(write_command, "post_for", fake_post)
    monkeypatch.setattr(write_command, "get", lambda url: {"data": []})


def seed(tmp_path: Path, extra: list[tuple[str, list[str]]] | None = None) -> dict[str, str]:
    conn = wiki_db(tmp_path, runs=2)
    exchange = add_exchange(conn, 1, "2026-02-01")
    rows = [
        ("Indigo2 runs IRIX fast", ["Indigo2", "IRIX"]),
        ("Indigo2 runs IRIX smoothly", ["Indigo2", "IRIX"]),
        ("Indigo2 has an R4400 cpu", ["Indigo2"]),
        *(extra or []),
    ]
    run = create_tag_run(
        conn, endpoint="e", model_alias="m", model_id="m", prompt_hash="p", now=NOW
    )
    tags = {}
    for i, (statement, names) in enumerate(rows):
        tags[add_claim(conn, exchange, f"user-{i:04d}", statement)] = names
    tags[add_claim(conn, exchange, "user-zzzz", "elsewhere", run_id=2)] = ["Indigo2"]
    record_tags(conn, run, tags)
    conn.commit()
    conn.close()
    return environment(tmp_path)


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_writes_a_general_and_a_with_section_and_drops_unsupported_sentences(
    tmp_path: Path,
) -> None:
    env = seed(tmp_path)
    code, out, _ = run(
        [*BASE, "--min-claims", "3", "--min-section", "1", "--runs", "1", "--write"], env
    )
    assert code == ExitCode.OK
    lines = out.splitlines()
    assert lines[-2].startswith("sections=2 failed=0 sentences=2 dropped=2 seconds=")
    assert lines[-1] == "article run 1 written"
    assert sections_for(db(tmp_path), [1]) == {
        "Indigo2": {
            "With IRIX": [("Indigo2 runs IRIX fast", [1])],
            "General": [("Indigo2 has an R4400 cpu", [3])],
        }
    }


def test_failed_sections_are_retried_and_written_ones_skipped_on_resume(tmp_path: Path) -> None:
    env = seed(tmp_path, [("poison pill", ["Indigo2"])])
    code, out, _ = run(
        [*BASE, "--min-claims", "3", "--min-section", "1", "--write", "--concurrency", "1"], env
    )
    assert code == ExitCode.OK
    assert "sections=2 failed=1 sentences=1 dropped=1 " in out
    STATE["poison"] = False
    code, out, _ = run(
        [*BASE, "--min-claims", "3", "--min-section", "1", "--write", "--resume"], env
    )
    assert "sections=1 failed=0 sentences=1 dropped=1 " in out
    assert out.splitlines()[-1] == "article run 2 written"
    code, out, _ = run(
        [*BASE, "--min-claims", "3", "--min-section", "1", "--write", "--resume"], env
    )
    assert out == "nothing to do\n"


def test_small_co_topic_sections_fold_into_general_by_default(tmp_path: Path) -> None:
    env = seed(tmp_path)
    code, _, _ = run([*BASE, "--min-claims", "3", "--runs", "1", "--write"], env)
    assert code == ExitCode.OK
    assert list(sections_for(db(tmp_path), [1])["Indigo2"]) == ["General"]


def test_limit_counts_topics_in_descending_claim_count(tmp_path: Path) -> None:
    extra = [(f"Octane two {w}", ["Octane"]) for w in ("alpha", "beta", "gamma", "delta")]
    env = seed(tmp_path, extra)
    code, _, _ = run([*BASE, "--min-claims", "3", "--write", "--limit", "1", "--runs", "1"], env)
    assert code == ExitCode.OK
    assert set(sections_for(db(tmp_path), [1])) == {"Octane"}


def test_sigint_stops_after_the_sections_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupting_post(endpoint: str) -> Transport:
        inner = fake_post(endpoint)

        def send(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], float]:
            os.kill(os.getpid(), signal.SIGINT)
            return inner(payload)

        return send

    monkeypatch.setattr(write_command, "post_for", interrupting_post)
    env = seed(tmp_path)
    code, out, _ = run(
        [*BASE, "--min-claims", "3", "--min-section", "1", "--write", "--concurrency", "1"], env
    )
    assert code == 130
    assert "interrupted after 1 sections; rerun with --resume" in out
    assert len(sections_for(db(tmp_path), [1])["Indigo2"]) == 1


def test_progress_goes_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = seed(tmp_path)
    run(
        [
            *BASE,
            "--min-claims",
            "3",
            "--min-section",
            "1",
            "--runs",
            "1",
            "--write",
            "--progress-every",
            "1",
        ],
        env,
    )
    assert "progress: 2/2 sections" in capsys.readouterr().err


def test_dry_run_prints_the_first_request_and_writes_nothing(tmp_path: Path) -> None:
    env = seed(tmp_path)
    code, out, _ = run([*BASE, "--min-claims", "3", "--min-section", "1"], env)
    assert code == ExitCode.OK
    assert out.startswith("Topic: Indigo2\nSection: With IRIX\n\n1. Indigo2 runs IRIX fast\n")
    assert db(tmp_path).execute("SELECT COUNT(*) FROM article_runs").fetchone()[0] == 0


def test_dry_run_with_no_eligible_topic(tmp_path: Path) -> None:
    env = seed(tmp_path)
    code, out, _ = run([*BASE, "--min-claims", "99"], env)
    assert code == ExitCode.OK and out == "nothing to do\n"


def test_bad_arguments_are_config_errors(tmp_path: Path) -> None:
    env = seed(tmp_path)
    code, _, err = run([*BASE[:3], "9", *BASE[4:]], env)
    assert code == ExitCode.CONFIG and "unknown tag run 9" in err
    code, _, err = run([*BASE, "--runs", "9"], env)
    assert code == ExitCode.CONFIG and "unknown run 9" in err
    code, _, err = run([*BASE, "--concurrency", "0"], env)
    assert code == ExitCode.CONFIG and "--concurrency must be between 1 and 32" in err
