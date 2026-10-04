import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from infovore.claims.httpd import listening_url, shutdown_all, start_all
from infovore.cli import ExitCode, main
from infovore.db.claims_v2 import ClaimIn, ExchangeOutcome, record_exchange, record_review
from infovore.db.connection import open_database
from infovore.timing import SystemClock
from tests.claims.seed import SALT, conversation, db, environment
from tests.claims.test_store import AT, make_run


def fetch(url: str, body: Any = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data)) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[str, int, Any]]:
    conn = db(tmp_path)
    eid, ids = conversation(conn, [(1, "Ann Real", "hello there"), (2, "bob", "ok")], 1)
    run = make_run(conn)
    ok = ExchangeOutcome("ok", None, 1, 1, 1, 1.0)
    claims = [ClaimIn("user-aaaa", f"user-aaaa said {i}", (ids[0],)) for i in range(3)]
    record_exchange(conn, run, eid, ok, claims, [])
    servers = start_all(["127.0.0.1"], 0, conn, SystemClock(), run)
    yield listening_url(servers[0]), run, conn
    shutdown_all(servers)


def test_the_page_lists_claims_with_real_sources_and_the_keys(
    served: tuple[str, int, Any],
) -> None:
    url, _, _ = served

    code, page = fetch(url)
    assert code == 200
    for key in ("'g'", "'w'", "'m'", "'n'"):
        assert key in page.decode()

    code, body = fetch(url + "api/claims")
    data = json.loads(body)
    assert code == 200 and len(data["claims"]) == 3
    first = data["claims"][0]
    assert first["sources"] == [{"id": 101, "author": "Ann Real", "text": "hello there"}]
    assert first["verdict"] is None and data["first_unreviewed"] == 0


def test_a_verdict_is_appended_and_review_resumes_at_the_first_unreviewed(
    served: tuple[str, int, Any],
) -> None:
    url, _, conn = served
    claims = json.loads(fetch(url + "api/claims")[1])["claims"]

    code, _ = fetch(url + "api/review", {"claim_id": claims[0]["id"], "verdict": "good"})
    assert code == 200
    fetch(url + "api/review", {"claim_id": claims[0]["id"], "verdict": "wrong"})

    data = json.loads(fetch(url + "api/claims")[1])
    assert data["claims"][0]["verdict"] == "wrong" and data["first_unreviewed"] == 1
    assert conn.execute("SELECT COUNT(*) FROM claim_reviews").fetchone()[0] == 2


@pytest.mark.parametrize(
    "body",
    [
        {"claim_id": 1, "verdict": "bogus"},
        {"claim_id": 999, "verdict": "good"},
        {"verdict": "good"},
        [],
    ],
)
def test_bad_reviews_are_refused_and_not_stored(served: tuple[str, int, Any], body: Any) -> None:
    url, _, conn = served

    code, _ = fetch(url + "api/review", body)

    assert code == 400
    assert conn.execute("SELECT COUNT(*) FROM claim_reviews").fetchone()[0] == 0


def test_unknown_paths_404_and_the_page_is_fast(served: tuple[str, int, Any]) -> None:
    url, _, _ = served

    assert fetch(url + "nope")[0] == 404
    assert fetch(url + "nope", {})[0] == 404
    start = time.perf_counter()
    fetch(url + "api/claims")
    assert time.perf_counter() - start < 0.1


def test_serve_command_listens_and_reviews_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = db(tmp_path)
    eid, ids = conversation(conn, [(1, "ann", "hi")], 1)
    run = make_run(conn)
    record_exchange(
        conn,
        run,
        eid,
        ExchangeOutcome("ok", None, 1, 1, 1, 1.0),
        [ClaimIn("u", "u said", (ids[0],))],
        [],
    )
    record_review(conn, 1, "good", AT)
    conn.close()
    seen: list[str] = []

    def stop(event: Any) -> None:
        seen.append("blocked")

    monkeypatch.setattr("infovore.sift.httpd.block_until_interrupted", stop)
    out, err = io.StringIO(), io.StringIO()

    code = main(
        ["claims", "serve", "--run", str(run), "--port", "0"],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=err,
    )

    assert code == ExitCode.OK and seen == ["blocked"]
    assert "listening on http://127.0.0.1:" in out.getvalue()
    assert "1 claims, 1 reviewed" in out.getvalue()
    check = open_database(tmp_path / "infovore.db")
    assert check.execute("SELECT COUNT(*) FROM claim_reviews").fetchone()[0] == 1


def test_serve_refuses_an_unknown_run(tmp_path: Path) -> None:
    db(tmp_path).close()
    out, err = io.StringIO(), io.StringIO()

    code = main(
        ["claims", "serve", "--run", "7"],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=err,
    )

    assert code == ExitCode.CONFIG and "unknown run 7" in err.getvalue()
    assert SALT not in err.getvalue()
