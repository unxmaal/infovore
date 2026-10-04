import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from infovore.cli import ExitCode
from infovore.db.annotations import annotation_history
from infovore.db.connection import open_database
from infovore.rows import MessageRow
from infovore.triage.llm_score import (
    SCORER_PREFIX,
    LlmCallError,
    build_request,
    http_transport,
    parse_reply,
    prompt_hash,
    render,
    score_windows,
    windows,
)
from tests.triage.test_cascade import build
from tests.triage.test_command import run


class Fake(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, Any]]] = []
    mode = "ok"

    def log_message(self, *args: object) -> None:
        return

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append(body)
        if Fake.mode == "500":
            self.send_response(500)
            self.end_headers()
            return
        text = body["messages"][1]["content"]
        if Fake.mode == "garbage":
            content = "not json"
        elif text.count("scsi") or text.count("kernel"):
            content = json.dumps({"decision": "relevant", "probability": 0.9})
        else:
            content = json.dumps({"decision": "irrelevant", "probability": 0.8})
        raw = json.dumps(
            {
                "model": "fake-4b-q4",
                "system_fingerprint": "b1-fake",
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 5},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server() -> Iterator[str]:
    Fake.seen = []
    Fake.mode = "ok"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()
    httpd.server_close()


def message(content: str, author: str = "a") -> MessageRow:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return MessageRow(1, 1, 9, 1, author, False, now, None, content, None, None, None, now, "{}")


def test_render_skips_blank_messages_and_names_authors() -> None:
    assert render([message("hi", "bob"), message("  "), message("yo")]) == ["bob: hi", "a: yo"]


def test_short_exchanges_make_one_window_and_long_ones_split() -> None:
    assert windows(["a: x", "b: y"], 100, 8) == ["a: x\nb: y"]
    assert windows(["a: x" * 10, "b: y" * 10], 60, 8) == ["a: x" * 10, "b: y" * 10]
    assert windows(["z" * 25], 10, 8) == ["z" * 10, "z" * 10, "z" * 5]
    assert windows(["z" * 25], 10, 2) == ["z" * 10, "z" * 10]
    assert windows([], 10, 8) == []


def test_the_request_enforces_a_json_schema_at_temperature_zero() -> None:
    request = build_request("eval-4b", "a: hi")

    assert request["model"] == "eval-4b" and request["temperature"] == 0
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "retrocomputing" in request["messages"][0]["content"]
    assert request["messages"][1]["content"].endswith("a: hi")
    assert len(prompt_hash()) == 12 and prompt_hash() == prompt_hash()


def reply(decision: str, p: float) -> dict[str, Any]:
    return {
        "model": "m",
        "choices": [{"message": {"content": json.dumps({"decision": decision, "probability": p})}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2},
    }


def test_the_probability_is_of_relevance_whatever_the_stated_decision() -> None:
    assert parse_reply(reply("relevant", 0.9), 0.5).p_relevant == 0.9
    assert parse_reply(reply("irrelevant", 0.9), 0.5).p_relevant == pytest.approx(0.1)
    assert parse_reply(reply("relevant", 7), 0.5).p_relevant == 1.0
    call = parse_reply(reply("relevant", 0.5), 1.5)
    assert (call.prompt_tokens, call.completion_tokens, call.seconds) == (7, 2, 1.5)
    bare = parse_reply(
        {"choices": [{"message": {"content": '{"decision": "relevant", "probability": 1}'}}]}, 0
    )
    assert (bare.prompt_tokens, bare.server_model, bare.fingerprint) == (0, "", "")


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"choices": [{"message": {"content": "nope"}}]},
        {"choices": [{"message": {"content": '{"decision": "maybe", "probability": 1}'}}]},
        {"choices": [{"message": {"content": '{"decision": "relevant"}'}}]},
    ],
)
def test_unusable_replies_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(LlmCallError):
        parse_reply(bad, 0.0)


def test_exchange_score_is_the_max_over_windows() -> None:
    answers = iter([reply("irrelevant", 0.9), reply("relevant", 0.7), reply("irrelevant", 0.6)])

    def fake(payload: Any) -> tuple[Any, float]:
        return next(answers), 0.25

    calls = score_windows(fake, "m", ["w1", "w2", "w3"])

    assert max(c.p_relevant for c in calls) == 0.7
    assert sum(c.seconds for c in calls) == 0.75


def test_the_real_transport_posts_and_reports_latency_and_failures(server: str) -> None:
    body, seconds = http_transport(server)(build_request("eval-4b", "a: scsi"))

    assert body["model"] == "fake-4b-q4" and seconds >= 0
    assert Fake.seen[0]["model"] == "eval-4b"
    Fake.mode = "500"
    with pytest.raises(LlmCallError):
        http_transport(server)(build_request("eval-4b", "x"))
    with pytest.raises(LlmCallError):
        http_transport("http://127.0.0.1:1/v1", 1.0)(build_request("eval-4b", "x"))


def llm(args: list[str], env: dict[str, str], endpoint: str) -> tuple[int, str, str]:
    return run(["relevance", "llm-score", "--endpoint", endpoint, "--model", "eval-4b", *args], env)


def test_it_refuses_without_a_limit_or_all_residue(tmp_path: Path, server: str) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, _, err = llm(["--residue", "--slices", "s1"], env, server)
    assert code == ExitCode.CONFIG and "--limit" in err
    assert llm(["--limit", "5"], env, server)[0] == ExitCode.CONFIG
    assert llm(["--all-residue", "--slices", "s1"], env, server)[0] == ExitCode.CONFIG
    assert llm(["--slices", "nope", "--limit", "5"], env, server)[0] == ExitCode.CONFIG
    assert Fake.seen == []


def test_dry_run_prints_the_requests_and_sends_nothing(tmp_path: Path, server: str) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = llm(["--residue", "--slices", "s1", "--limit", "2", "--dry-run"], env, server)

    assert code == ExitCode.OK
    assert out.count('"response_format"') == 2
    assert "dry-run: 2 requests for 2 exchanges, none sent" in out
    assert Fake.seen == []
    conn = open_database(env["INFOVORE_DB_PATH"])
    assert annotation_history(conn, "exchange", 1, SCORER_PREFIX + "eval-4b") == []


def test_scoring_the_residue_without_write_records_nothing(tmp_path: Path, server: str) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = llm(["--residue", "--slices", "s1", "--limit", "10"], env, server)

    assert code == ExitCode.OK
    assert "scored=3 failed=0 tokens=" in out and "server_model=fake-4b-q4" in out
    assert "tokens=30+5" in out and "wrote" not in out
    assert "labelled: relevant=" in out and "(no training)" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    assert (
        conn.execute("SELECT COUNT(*) FROM annotations WHERE scorer LIKE 'p_relevant%'").fetchone()[
            0
        ]
        == 0
    )


def test_write_records_derived_annotations_the_human_report_can_read(
    tmp_path: Path, server: str
) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = llm(
        ["--residue", "--slices", "s1", "--labelled-only", "--limit", "10", "--write"], env, server
    )

    assert code == ExitCode.OK
    assert "wrote 3: p_relevant_llm_eval-4b v1" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    row = annotation_history(conn, "exchange", 7, SCORER_PREFIX + "eval-4b")[0]
    assert row["reproducibility"] == "derived"
    assert row["score"] == 0.9 and row["label"] == "relevant"
    recipe = json.loads(row["recipe_json"])
    assert recipe["endpoint"] == server and recipe["model"] == "eval-4b"
    assert recipe["server_model"] == "fake-4b-q4" and recipe["server_fingerprint"] == "b1-fake"
    assert recipe["prompt_sha"] == prompt_hash() and recipe["aggregate"] == "max"
    assert recipe["windows"] == 1 and recipe["prompt_tokens"] == 30 and "seconds" in recipe
    code, out, _ = run(
        ["triage", "--human-report", "--scorer", SCORER_PREFIX + "eval-4b", "--include-training"],
        env,
    )
    assert code == ExitCode.OK
    assert f"scorer {SCORER_PREFIX}eval-4b v1: evaluated=" in out
    llm(["--slices", "s1", "--limit", "1", "--write"], env, server)
    assert len(annotation_history(conn, "exchange", 1, SCORER_PREFIX + "eval-4b")) == 1
    assert (
        annotation_history(conn, "exchange", 1, SCORER_PREFIX + "eval-4b")[0]["scorer_version"] == 2
    )


def test_long_exchanges_are_windowed_and_scored_by_their_best_window(
    tmp_path: Path, server: str
) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = llm(
        ["--slices", "s1", "--limit", "10", "--window-chars", "12", "--max-windows", "3"],
        env,
        server,
    )

    assert code == ExitCode.OK
    assert max(len(s["messages"][1]["content"]) for s in Fake.seen) <= 12 + len("Exchange:\n")
    assert "windows=2" in out or "windows=3" in out


def test_residue_over_every_current_exchange_and_all_residue(tmp_path: Path, server: str) -> None:
    env, conn = build(tmp_path)
    conn.close()

    code, out, _ = llm(["--residue", "--all-residue"], env, server)

    assert code == ExitCode.OK and "scored=" in out


def test_failures_are_reported_per_exchange_and_fail_the_run_when_total(
    tmp_path: Path, server: str
) -> None:
    env, conn = build(tmp_path)
    conn.close()
    Fake.mode = "garbage"

    code, out, _ = llm(["--slices", "s1", "--limit", "2"], env, server)

    assert code == ExitCode.BACKEND
    assert "ERROR" in out and "scored=0 failed=2" in out and "auc=n/a" in out
    Fake.mode = "500"
    assert llm(["--slices", "s1", "--limit", "1"], env, server)[0] == ExitCode.BACKEND


def test_exchanges_without_text_are_skipped(tmp_path: Path, server: str) -> None:
    env, conn = build(tmp_path)
    conn.execute("UPDATE messages SET content = '' WHERE id = 1")
    conn.commit()
    conn.close()

    code, out, _ = llm(["--slices", "s1", "--limit", "10"], env, server)

    assert code == ExitCode.OK and "scored=5" in out
