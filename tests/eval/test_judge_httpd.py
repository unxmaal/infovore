import json
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.eval.judge import frozen_queue
from infovore.eval.judge_httpd import JudgeServer, listening_url, shutdown_all, start_all
from infovore.eval.slices import GOLD, GOLD_REPEATS
from infovore.timing import FixedClock

AT = datetime(2026, 10, 2, tzinfo=UTC)


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 9, NULL, 'hardware', 'text')"
    )
    for exchange_id, message_ids in ((1, [11, 12]), (2, [21])):
        conn.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
            " ended_at, message_count, grouping_rule, content_hash)"
            " VALUES (?, 1, ?, ?, ?, ?, ?, 'quiet_gap', ?)",
            (
                exchange_id,
                message_ids[0],
                message_ids[-1],
                AT.isoformat(),
                AT.isoformat(),
                len(message_ids),
                f"h{exchange_id}",
            ),
        )
        for position, message_id in enumerate(message_ids, start=1):
            conn.execute(
                "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
                " created_at, content, ingested_at, raw_json)"
                " VALUES (?, 1, 9, 5, 'hal', ?, 'hi', ?, '{}')",
                (message_id, AT.isoformat(), AT.isoformat()),
            )
            conn.execute(
                "INSERT INTO exchange_messages (exchange_id, message_id, position)"
                " VALUES (?, ?, ?)",
                (exchange_id, message_id, position),
            )
    for name, ids in ((GOLD, [1, 2]), (GOLD_REPEATS, [1])):
        for position, exchange_id in enumerate(ids, start=1):
            conn.execute(
                "INSERT INTO eval_slices (name, exchange_id, position, population, seed, frozen_at)"
                " VALUES (?, ?, ?, 't', 0, ?)",
                (name, exchange_id, position, AT.isoformat()),
            )
    return conn


@pytest.fixture
def server(tmp_path: Path) -> Iterator[JudgeServer]:
    servers = start_all(["127.0.0.1"], 0, _db(tmp_path), FixedClock(AT), frozen_queue)
    yield servers[0]
    shutdown_all(servers)


def _get(server: JudgeServer, path: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(listening_url(server).rstrip("/") + path) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
    return status, (json.loads(body) if path.startswith("/api") else body)


def _post(server: JudgeServer, path: str, payload: bytes) -> tuple[int, object]:
    request = urllib.request.Request(
        listening_url(server).rstrip("/") + path,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_page_is_served(server: JudgeServer) -> None:
    status, body = _get(server, "/")

    assert status == 200
    assert "bad grouping" in str(body)


def test_next_points_at_the_first_unjudged_item(server: JudgeServer) -> None:
    assert _get(server, "/api/next") == (200, {"index": 0, "progress": {"done": 0, "total": 3}})


def test_an_exchange_comes_back_with_its_messages_and_interface_version(
    server: JudgeServer,
) -> None:
    status, body = _get(server, "/api/exchange?index=0")

    assert status == 200
    assert isinstance(body, dict)
    assert [m["id"] for m in body["messages"]] == [11, 12]
    assert body["label"] is None
    assert body["interface_version"] == 3


def _submit(server: JudgeServer, index: int, exchange_id: int, label: str) -> tuple[int, object]:
    payload = {"index": index, "exchange_id": exchange_id, "label": label}
    return _post(server, "/api/submit", json.dumps(payload).encode())


def test_a_submission_is_saved_and_moves_the_resume_point(server: JudgeServer) -> None:
    status, body = _submit(server, 0, 1, "relevant")

    assert (status, body) == (200, {"progress": {"done": 1, "total": 3}})
    assert _get(server, "/api/next")[1] == {"index": 1, "progress": {"done": 1, "total": 3}}
    assert _get(server, "/api/exchange?index=0")[1]["label"] == "relevant"  # type: ignore[index]


def test_every_item_judged_leaves_no_resume_point(server: JudgeServer) -> None:
    for index, exchange_id in enumerate([1, 2, 1]):
        _submit(server, index, exchange_id, "irrelevant")

    assert _get(server, "/api/next")[1] == {"index": None, "progress": {"done": 3, "total": 3}}


@pytest.mark.parametrize(
    ("path", "status"),
    [("/api/exchange?index=abc", 400), ("/api/exchange?index=9", 404), ("/nope", 404)],
)
def test_bad_reads_are_refused(server: JudgeServer, path: str, status: int) -> None:
    assert _get(server, path)[0] == status


@pytest.mark.parametrize(
    ("path", "payload", "status"),
    [
        ("/api/submit", b"not json", 400),
        ("/api/submit", json.dumps({"index": 0, "label": "relevant"}).encode(), 400),
        (
            "/api/submit",
            json.dumps({"index": 9, "exchange_id": 1, "label": "relevant"}).encode(),
            404,
        ),
        (
            "/api/submit",
            json.dumps({"index": 0, "exchange_id": 2, "label": "relevant"}).encode(),
            400,
        ),
        ("/api/submit", json.dumps({"index": 0, "exchange_id": 1, "label": "fact"}).encode(), 400),
        ("/api/other", b"{}", 404),
    ],
)
def test_bad_submissions_are_refused(
    server: JudgeServer, path: str, payload: bytes, status: int
) -> None:
    assert _post(server, path, payload)[0] == status


def test_the_listening_url_names_host_and_port(server: JudgeServer) -> None:
    assert listening_url(server).startswith("http://127.0.0.1:")
