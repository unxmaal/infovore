import io
import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from infovore.claims.redact import pseudonym
from infovore.cli import ExitCode, main
from infovore.db.connection import open_database
from tests.claims.seed import SALT, conversation, db, environment

LINES_A = [(11, "Alice Smith", "my Indy runs IRIX 6.5"), (22, "bobby", "<@11> try the PROM")]
LINES_B = [(33, "carol", "lol")]


class Fake(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, Any]]] = []
    mode = "ok"
    info = True

    def log_message(self, *args: object) -> None:
        return

    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        assert self.headers["Authorization"] == "Bearer sk-local"
        if self.path != "/model/info" or not Fake.info:
            self._send(404, b"{}")
            return
        data = {"data": [{"model_name": "eval-4b", "litellm_params": {"model": "openai/q4b"}}]}
        self._send(200, json.dumps(data).encode())

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append(body)
        if Fake.mode == "500":
            self._send(500, b"{}")
            return
        user = body["messages"][1]["content"]
        claims = []
        if "Indy" in user:
            speaker = re.search(r"\[1\] (user-[0-9a-f]+):", user)
            assert speaker
            who = speaker.group(1)
            claims = [{"speaker": who, "statement": f"{who} said the Indy runs IRIX", "refs": [1]}]
        content = json.dumps({"claims": claims})
        reply = {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 8},
        }
        self._send(200, json.dumps(reply).encode())


@pytest.fixture
def server() -> Iterator[str]:
    Fake.seen = []
    Fake.mode = "ok"
    Fake.info = True
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()
    httpd.server_close()


def run(tmp_path: Path, argv: list[str], salt: str | None = SALT) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=environment(tmp_path, salt), dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def seeded(tmp_path: Path) -> tuple[int, int]:
    conn = db(tmp_path)
    a, _ = conversation(conn, LINES_A, 1)
    b, _ = conversation(conn, LINES_B, 2)
    conn.close()
    return a, b


def extract(endpoint: str, *more: str) -> list[str]:
    return ["claims", "extract", "--endpoint", endpoint, "--model", "eval-4b", *more]


def test_extract_refuses_without_a_limit(tmp_path: Path, server: str) -> None:
    seeded(tmp_path)

    code, _, err = run(tmp_path, extract(server, "--slices", "gold"))

    assert code == ExitCode.CONFIG and "--limit" in err
    assert Fake.seen == []


@pytest.mark.parametrize(
    ("extra", "salt", "message"),
    [
        (["--limit", "0", "--slices", "gold"], SALT, "--limit must be positive"),
        (["--limit", "2"], SALT, "--slices, --ids or --channels"),
        (["--limit", "2", "--slices", "nope"], SALT, "unknown slice"),
        (["--limit", "2", "--slices", "gold"], None, "INFOVORE_PSEUDONYM_SALT"),
        (["--limit", "2", "--ids", "999"], SALT, "unknown exchange"),
        (["--limit", "2", "--ids", "x"], SALT, "--ids"),
    ],
)
def test_extract_refuses_bad_input(
    tmp_path: Path, server: str, extra: list[str], salt: str | None, message: str
) -> None:
    seeded(tmp_path)

    code, _, err = run(tmp_path, extract(server, *extra), salt)

    assert code == ExitCode.CONFIG and message in err
    assert Fake.seen == []


def test_dry_run_prints_redacted_requests_and_sends_nothing(tmp_path: Path) -> None:
    seeded(tmp_path)

    code, out, _ = run(
        tmp_path, extract("http://127.0.0.1:9/v1", "--slices", "gold", "--limit", "5", "--dry-run")
    )

    assert code == ExitCode.OK
    assert "dry-run: 2 requests for 2 conversations, none sent" in out
    for real in ("Alice", "Smith", "bobby", "carol", "<@"):
        assert real not in out
    assert pseudonym(11, SALT) in out
    conn = open_database(tmp_path / "infovore.db")
    assert conn.execute("SELECT COUNT(*) FROM claim_runs").fetchone()[0] == 0


def test_without_write_the_model_runs_but_nothing_is_stored(tmp_path: Path, server: str) -> None:
    seeded(tmp_path)

    code, out, _ = run(tmp_path, extract(server, "--slices", "gold", "--limit", "5"))

    assert code == ExitCode.OK
    assert "claims=1" in out and "not written" in out
    assert len(Fake.seen) == 2
    conn = open_database(tmp_path / "infovore.db")
    assert conn.execute("SELECT COUNT(*) FROM claim_runs").fetchone()[0] == 0


def test_write_stores_the_run_claims_and_real_model_id(tmp_path: Path, server: str) -> None:
    a, b = seeded(tmp_path)

    code, out, _ = run(tmp_path, extract(server, "--ids", f"{a},{b}", "--limit", "5", "--write"))

    assert code == ExitCode.OK
    conn = open_database(tmp_path / "infovore.db")
    run_row = conn.execute("SELECT * FROM claim_runs").fetchone()
    assert (run_row["model_alias"], run_row["model_id"]) == ("eval-4b", "openai/q4b")
    assert run_row["model_id_source"] == "model_info" and len(run_row["prompt_hash"]) == 12
    recipe = json.loads(run_row["recipe_json"])
    assert recipe["recipe"] == "claims-v2" and "salt_fingerprint" in recipe
    assert SALT not in run_row["recipe_json"]
    claim = conn.execute("SELECT * FROM claims_v2").fetchone()
    assert claim["exchange_id"] == a and claim["speaker"] == pseudonym(11, SALT)
    assert "Alice" not in claim["statement"]
    cited = [r[0] for r in conn.execute("SELECT message_id FROM claims_v2_sources")]
    assert cited == [101]
    per = conn.execute(
        "SELECT SUM(input_tokens), SUM(output_tokens), COUNT(*) FROM claim_run_exchanges"
    ).fetchone()
    assert tuple(per) == (80, 16, 2)
    assert f"run {run_row['id']}" in out and "model_id=openai/q4b" in out


def test_no_real_name_ever_reaches_the_model(tmp_path: Path, server: str) -> None:
    seeded(tmp_path)

    run(tmp_path, extract(server, "--slices", "gold", "--limit", "5"))

    wire = json.dumps(Fake.seen)
    for real in ("Alice", "Smith", "bobby", "carol", "<@", "11>"):
        assert real not in wire


def test_limit_caps_conversations_and_missing_model_info_falls_back(
    tmp_path: Path, server: str
) -> None:
    seeded(tmp_path)
    Fake.info = False

    code, out, _ = run(tmp_path, extract(server, "--slices", "gold", "--limit", "1", "--write"))

    assert code == ExitCode.OK and len(Fake.seen) == 1
    assert "model_id=eval-4b (alias; /model/info unavailable)" in out
    conn = open_database(tmp_path / "infovore.db")
    assert conn.execute("SELECT model_id_source FROM claim_runs").fetchone()[0] == "alias"


def test_a_failing_backend_is_recorded_per_conversation(tmp_path: Path, server: str) -> None:
    seeded(tmp_path)
    Fake.mode = "500"

    code, out, _ = run(tmp_path, extract(server, "--slices", "gold", "--limit", "5", "--write"))

    assert code == ExitCode.BACKEND and "failed=2" in out
    conn = open_database(tmp_path / "infovore.db")
    rows = conn.execute("SELECT outcome, error FROM claim_run_exchanges").fetchall()
    assert [r["outcome"] for r in rows] == ["failed", "failed"]
    assert all(r["error"] for r in rows)


def test_a_failure_without_write_stores_nothing(tmp_path: Path, server: str) -> None:
    seeded(tmp_path)
    Fake.mode = "500"

    code, out, _ = run(tmp_path, extract(server, "--slices", "gold", "--limit", "5"))

    assert code == ExitCode.BACKEND and "not written" in out
    conn = open_database(tmp_path / "infovore.db")
    assert conn.execute("SELECT COUNT(*) FROM claim_run_exchanges").fetchone()[0] == 0
