import json
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from infovore.db.connection import migrate, open_database
from infovore.sift.httpd import listening_url, load_page, shutdown_all, start_all
from infovore.sift.serve import ServeApp, load_batch_messages
from infovore.timing import FixedClock

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _channel(conn: sqlite3.Connection, channel_id: int, name: str) -> None:
    conn.execute(
        "INSERT INTO channels (id, guild_id, name, kind) VALUES (?, 1, ?, 'text')",
        (channel_id, name),
    )


def _message_with_exchange(
    conn: sqlite3.Connection, message_id: int, channel_id: int, content: str
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 1, 1, 'alice', ?, ?, ?, '{}')",
        (message_id, channel_id, NOW.isoformat(), content, NOW.isoformat()),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, ?, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
        (
            message_id,
            channel_id,
            message_id,
            message_id,
            NOW.isoformat(),
            NOW.isoformat(),
            f"h{message_id}",
        ),
    )
    conn.execute(
        "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
        (message_id, message_id),
    )


@pytest.fixture
def app(tmp_path: Path) -> ServeApp:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, 1, 1, "general chatter one")
    _message_with_exchange(conn, 2, 1, "general chatter two")
    messages = load_batch_messages(conn, [1, 2])
    return ServeApp(conn, messages, "batch1", tmp_path / "scratch", FixedClock(NOW))


@pytest.fixture
def base_url(app: ServeApp) -> Iterator[str]:
    servers = start_all(["127.0.0.1"], 0, app)
    url = listening_url(servers[0]).rstrip("/")
    try:
        yield url
    finally:
        shutdown_all(servers)


def _get(url: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("Content-Type", "")


def _post(url: str, payload: dict[str, object]) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_get_root_serves_a_self_contained_html_page_with_no_external_resources(
    base_url: str,
) -> None:
    status, body, content_type = _get(base_url + "/")
    assert status == 200
    assert "text/html" in content_type
    text = body.decode("utf-8")
    assert "<html" in text.lower()
    assert "http://" not in text
    assert "https://" not in text


def test_load_page_is_non_empty() -> None:
    assert "<html" in load_page().lower()


def test_get_unknown_path_is_404(base_url: str) -> None:
    status, _, _ = _get(base_url + "/nope")
    assert status == 404


def test_get_state_returns_the_batch(base_url: str) -> None:
    status, body, content_type = _get(base_url + "/api/state")
    assert status == 200
    assert "application/json" in content_type
    data = json.loads(body)
    assert data["batch"] == "batch1"
    assert [m["id"] for m in data["messages"]] == [1, 2]
    assert all(m["label"] is None for m in data["messages"])


def test_post_label_then_undo_round_trip(base_url: str) -> None:
    status, data = _post(base_url + "/api/label", {"message_id": 1, "label": "trash"})
    assert status == 200
    labels = {m["id"]: m["label"] for m in data["state"]["messages"]}
    assert labels[1] == "trash"

    status, data = _post(base_url + "/api/undo", {})
    assert status == 200
    assert data["undone"] is True
    labels = {m["id"]: m["label"] for m in data["state"]["messages"]}
    assert labels[1] is None


def test_post_trash_channel(base_url: str) -> None:
    status, data = _post(base_url + "/api/trash_channel", {"channel": "general"})
    assert status == 200
    assert data["trashed"] == 2


def test_post_rules_preview_and_apply(base_url: str) -> None:
    status, data = _post(base_url + "/api/rules/preview", {"type": "contains", "value": "chatter"})
    assert status == 200
    assert data["matched"] == 2
    assert data["conflicts"] == 0

    status, data = _post(
        base_url + "/api/rules/apply",
        {"type": "contains", "value": "chatter", "name": "general-chatter"},
    )
    assert status == 200
    assert data["applied"] == 2
    assert Path(data["saved_to"]).exists()


def test_post_label_with_unknown_message_is_400(base_url: str) -> None:
    status, data = _post(base_url + "/api/label", {"message_id": 999, "label": "trash"})
    assert status == 400
    assert "error" in data


def test_post_rules_preview_with_invalid_rule_is_400(base_url: str) -> None:
    status, data = _post(base_url + "/api/rules/preview", {"type": "regex", "value": "("})
    assert status == 400
    assert "error" in data


def test_post_rules_apply_without_a_name_is_400(base_url: str) -> None:
    status, data = _post(base_url + "/api/rules/apply", {"type": "contains", "value": "chatter"})
    assert status == 400
    assert "error" in data


def test_post_unknown_path_is_404(base_url: str) -> None:
    status, _ = _post(base_url + "/nope", {})
    assert status == 404


def test_get_bad_body_on_post_is_400(base_url: str) -> None:
    request = urllib.request.Request(
        base_url + "/api/label",
        data=b"not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request)
        raise AssertionError("expected an HTTPError")
    except urllib.error.HTTPError as error:
        assert error.code == 400


def test_post_body_that_is_not_a_json_object_is_400(base_url: str) -> None:
    request = urllib.request.Request(
        base_url + "/api/label",
        data=b"[1, 2, 3]",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request)
        raise AssertionError("expected an HTTPError")
    except urllib.error.HTTPError as error:
        assert error.code == 400


def test_block_until_interrupted_returns_once_the_event_is_set() -> None:
    import threading
    import time

    from infovore.sift.httpd import block_until_interrupted

    event = threading.Event()

    def _set_soon() -> None:
        time.sleep(0.02)
        event.set()

    threading.Thread(target=_set_soon, daemon=True).start()
    block_until_interrupted(event)
    assert event.is_set()


def test_block_until_interrupted_swallows_keyboard_interrupt() -> None:
    from infovore.sift.httpd import block_until_interrupted

    class _RaisingEvent:
        def wait(self) -> None:
            raise KeyboardInterrupt

    block_until_interrupted(_RaisingEvent())  # type: ignore[arg-type]
