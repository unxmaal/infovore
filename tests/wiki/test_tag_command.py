import io
import json
import os
import re
import signal
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from infovore.claims.extract import ClaimsReplyError
from infovore.cli import ExitCode, main
from infovore.db.wiki_tags import tags_for
from infovore.wiki import tag_command, tagger
from tests.claims.seed import db, environment
from tests.wiki.seed import add_claim, add_exchange, wiki_db

BASE = ["wiki", "tag", "--runs", "1", "--endpoint", "http://fake", "--model", "m"]


def fake_post(endpoint: str) -> Any:
    def send(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], float]:
        text = payload["messages"][1]["content"]
        if "poison" in text:
            raise ClaimsReplyError("boom")
        pairs = re.findall(r"^(\d+)\. (.*)$", text, re.M)
        body = {"t": [[int(n), [f"tag-{s}"]] for n, s in pairs]}
        reply = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(body)}}]}
        return reply, 0.0

    return send


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tag_command, "post_for", fake_post)
    monkeypatch.setattr(tag_command, "get", lambda url: {"data": []})
    monkeypatch.setattr(tagger, "BATCH", 2)


def seed(tmp_path: Path, statements: list[str]) -> dict[str, str]:
    conn = wiki_db(tmp_path, runs=2)
    exchange = add_exchange(conn, 1, "2026-02-01")
    for statement in statements:
        add_claim(conn, exchange, "user-aaaa", statement)
    add_claim(conn, exchange, "user-bbbb", "other run", run_id=2)
    conn.commit()
    conn.close()
    return environment(tmp_path)


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_writes_tags_for_the_chosen_run_only(tmp_path: Path) -> None:
    env = seed(tmp_path, ["one", "two", "three"])
    code, out, _ = run([*BASE, "--write"], env)
    assert code == ExitCode.OK
    lines = out.splitlines()
    assert lines[-2].startswith("claims=3 batches=2 failed=0 tags=3 seconds=")
    assert lines[-1] == "tag run 1 written"
    conn = db(tmp_path)
    assert tags_for(conn, [1]) == {1: ["tag-one"], 2: ["tag-two"], 3: ["tag-three"]}


def test_resume_skips_claims_already_tagged(tmp_path: Path) -> None:
    env = seed(tmp_path, ["one", "two", "three"])
    run([*BASE, "--write", "--limit", "2"], env)
    code, out, _ = run([*BASE, "--write", "--resume"], env)
    assert code == ExitCode.OK
    assert "claims=1 batches=1 failed=0 tags=1 " in out
    assert tags_for(db(tmp_path), [2]) == {3: ["tag-three"]}


def test_a_failed_batch_is_counted_and_the_others_are_written(tmp_path: Path) -> None:
    env = seed(tmp_path, ["one", "two", "poison", "three", "four"])
    code, out, _ = run([*BASE, "--write", "--concurrency", "1", "--progress-every", "1"], env)
    assert code == ExitCode.OK
    assert "claims=5 batches=3 failed=1 tags=3 " in out
    assert tags_for(db(tmp_path), [1]) == {
        1: ["tag-one"],
        2: ["tag-two"],
        5: ["tag-four"],
    }


def test_sigint_stops_after_the_batches_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupting_post(endpoint: str) -> Any:
        inner = fake_post(endpoint)

        def send(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], float]:
            os.kill(os.getpid(), signal.SIGINT)
            return inner(payload)

        return send

    monkeypatch.setattr(tag_command, "post_for", interrupting_post)
    env = seed(tmp_path, ["one", "two", "three", "four"])
    code, out, _ = run([*BASE, "--write", "--concurrency", "1"], env)
    assert code == 130
    assert "interrupted after 1 batches; rerun with --resume" in out
    assert tags_for(db(tmp_path), [1]) == {1: ["tag-one"], 2: ["tag-two"]}


def test_dry_run_prints_the_first_request_and_writes_nothing(tmp_path: Path) -> None:
    env = seed(tmp_path, ["one", "two", "three"])
    code, out, _ = run(BASE, env)
    assert code == ExitCode.OK
    assert out == "1. one\n2. two\n"
    assert db(tmp_path).execute("SELECT COUNT(*) FROM tag_runs").fetchone()[0] == 0


def test_nothing_to_do(tmp_path: Path) -> None:
    env = seed(tmp_path, [])
    for extra in ([], ["--write"]):
        code, out, _ = run([*BASE, *extra], env)
        assert code == ExitCode.OK and out == "nothing to do\n"


def test_unknown_run_and_bad_arguments_are_config_errors(tmp_path: Path) -> None:
    env = seed(tmp_path, ["one"])
    code, _, err = run(["wiki", "tag", "--runs", "9", "--endpoint", "e", "--model", "m"], env)
    assert code == ExitCode.CONFIG and "unknown run 9" in err
    code, _, err = run(["wiki", "tag", "--runs", "x", "--endpoint", "e", "--model", "m"], env)
    assert code == ExitCode.CONFIG and "comma-separated" in err
    code, _, err = run([*BASE, "--concurrency", "0"], env)
    assert code == ExitCode.CONFIG and "--concurrency must be between 1 and 32" in err
    code, _, err = run([*BASE, "--concurrency", "33"], env)
    assert code == ExitCode.CONFIG
