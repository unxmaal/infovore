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
    assert [m["id"] for m in data["messages"]] == ["1", "2"]
    assert all(m["label"] is None for m in data["messages"])


def test_post_label_then_undo_round_trip(base_url: str) -> None:
    status, data = _post(base_url + "/api/label", {"message_id": "1", "label": "trash"})
    assert status == 200
    labels = {m["id"]: m["label"] for m in data["state"]["messages"]}
    assert labels["1"] == "trash"

    status, data = _post(base_url + "/api/undo", {})
    assert status == 200
    assert data["undone"] is True
    labels = {m["id"]: m["label"] for m in data["state"]["messages"]}
    assert labels["1"] is None


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
    status, data = _post(base_url + "/api/label", {"message_id": "999", "label": "trash"})
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


SNOWFLAKE = 527504908467830833


@pytest.fixture
def snowflake_url(tmp_path: Path) -> Iterator[str]:
    conn = open_database(tmp_path / "s.db")
    migrate(conn)
    _channel(conn, 1, "general")
    _message_with_exchange(conn, SNOWFLAKE, 1, "a real discord-sized id")
    messages = load_batch_messages(conn, [SNOWFLAKE])
    app = ServeApp(conn, messages, "batch1", tmp_path / "scratch", FixedClock(NOW))
    servers = start_all(["127.0.0.1"], 0, app)
    try:
        yield listening_url(servers[0]).rstrip("/")
    finally:
        shutdown_all(servers)


def test_state_sends_message_ids_as_exact_strings(snowflake_url: str) -> None:
    _, body, _ = _get(snowflake_url + "/api/state")
    assert json.loads(body)["messages"][0]["id"] == str(SNOWFLAKE)


def test_label_accepts_a_snowflake_id_sent_as_a_string(snowflake_url: str) -> None:
    status, data = _post(
        snowflake_url + "/api/label", {"message_id": str(SNOWFLAKE), "label": "trash"}
    )
    assert status == 200
    assert data["state"]["messages"][0]["label"] == "trash"


def test_label_rejects_a_numeric_message_id(snowflake_url: str) -> None:
    status, data = _post(snowflake_url + "/api/label", {"message_id": SNOWFLAKE, "label": "trash"})
    assert status == 400
    assert "string" in data["error"]


def test_page_reports_api_errors_instead_of_ignoring_them() -> None:
    page = load_page()
    assert 'id="error"' in page
    assert "showError" in page


# --- /api/context (issue #137) ------------------------------------------------

_CONVERSATION = [
    (1, 1, "alice", "one", "2026-01-01T00:00:00+00:00"),
    (2, 2, "bob", "two", "2026-01-01T00:01:00+00:00"),
    (3, 1, "alice", "three", "2026-01-01T00:02:00+00:00"),
    (4, 2, "bob", "four", "2026-01-01T00:03:00+00:00"),
    (5, 1, "alice", "five", "2026-01-01T00:04:00+00:00"),
]


def _exchange_with_messages(
    conn: sqlite3.Connection,
    exchange_id: int,
    channel_id: int,
    message_specs: list[tuple[int, int, str, str, str]],
) -> None:
    first_id, last_id = message_specs[0][0], message_specs[-1][0]
    first_created, last_created = message_specs[0][4], message_specs[-1][4]
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, 'quiet_gap', ?)",
        (
            exchange_id,
            channel_id,
            first_id,
            last_id,
            first_created,
            last_created,
            len(message_specs),
            f"hx{exchange_id}",
        ),
    )
    for position, (message_id, author_id, author_name, content, created_at) in enumerate(
        message_specs, start=1
    ):
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, ?, 1, ?, ?, ?, ?, ?, '{}')",
            (message_id, channel_id, author_id, author_name, created_at, content, NOW.isoformat()),
        )
        conn.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, ?)",
            (exchange_id, message_id, position),
        )


@pytest.fixture
def context_url(tmp_path: Path) -> Iterator[str]:
    conn = open_database(tmp_path / "ctx.db")
    migrate(conn)
    _channel(conn, 1, "general")
    _exchange_with_messages(conn, 100, 1, _CONVERSATION)
    messages = load_batch_messages(conn, [3])
    app = ServeApp(conn, messages, "batch1", tmp_path / "scratch", FixedClock(NOW))
    servers = start_all(["127.0.0.1"], 0, app)
    try:
        yield listening_url(servers[0]).rstrip("/")
    finally:
        shutdown_all(servers)


def test_get_context_returns_the_window_in_position_order(context_url: str) -> None:
    status, body, content_type = _get(context_url + "/api/context?message_id=3&before=1&after=1")
    assert status == 200
    assert "application/json" in content_type
    data = json.loads(body)
    assert data["message_id"] == "3"
    assert [m["id"] for m in data["messages"]] == ["2", "3", "4"]
    assert [m["focused"] for m in data["messages"]] == [False, True, False]


def test_get_context_uses_defaults_when_before_after_omitted(context_url: str) -> None:
    status, body, _ = _get(context_url + "/api/context?message_id=3")
    assert status == 200
    data = json.loads(body)
    assert [m["id"] for m in data["messages"]] == ["1", "2", "3", "4", "5"]


def test_get_context_caps_an_oversized_window(context_url: str) -> None:
    status, body, _ = _get(context_url + "/api/context?message_id=3&before=999&after=999")
    assert status == 200
    data = json.loads(body)
    assert data["before"] == 20
    assert data["after"] == 20


def test_get_context_redacts_opted_out_authors(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "ctx-redact.db")
    migrate(conn)
    _channel(conn, 1, "general")
    _exchange_with_messages(conn, 100, 1, _CONVERSATION)
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (2, ?)", (NOW.isoformat(),))
    messages = load_batch_messages(conn, [3])
    app = ServeApp(conn, messages, "batch1", tmp_path / "scratch", FixedClock(NOW))
    servers = start_all(["127.0.0.1"], 0, app)
    try:
        url = listening_url(servers[0]).rstrip("/")
        status, body, _ = _get(url + "/api/context?message_id=3&before=4&after=4")
    finally:
        shutdown_all(servers)
    assert status == 200
    by_id = {m["id"]: m for m in json.loads(body)["messages"]}
    assert by_id["2"]["author"] == "[redacted]"
    assert by_id["2"]["content"] == "[redacted]"


def test_get_context_missing_message_id_is_400(context_url: str) -> None:
    status, body, _ = _get(context_url + "/api/context")
    assert status == 400
    assert "error" in json.loads(body)


def test_get_context_non_numeric_message_id_is_400(context_url: str) -> None:
    status, body, _ = _get(context_url + "/api/context?message_id=not-a-number")
    assert status == 400
    assert "error" in json.loads(body)


def test_get_context_unknown_message_id_is_400(context_url: str) -> None:
    status, body, _ = _get(context_url + "/api/context?message_id=999999")
    assert status == 400
    assert "error" in json.loads(body)


def test_get_context_invalid_before_is_400(context_url: str) -> None:
    status, body, _ = _get(context_url + "/api/context?message_id=3&before=nope")
    assert status == 400
    assert "error" in json.loads(body)


def test_get_context_invalid_after_is_400(context_url: str) -> None:
    status, body, _ = _get(context_url + "/api/context?message_id=3&after=nope")
    assert status == 400
    assert "error" in json.loads(body)


def test_get_context_sends_a_real_size_snowflake_id_as_an_exact_string(
    tmp_path: Path,
) -> None:
    conn = open_database(tmp_path / "ctx-snowflake.db")
    migrate(conn)
    _channel(conn, 1, "general")
    snowflake_conversation = [
        (SNOWFLAKE, 1, "alice", "one", "2026-01-01T00:00:00+00:00"),
        (SNOWFLAKE + 1, 2, "bob", "two", "2026-01-01T00:01:00+00:00"),
        (SNOWFLAKE + 2, 1, "alice", "three", "2026-01-01T00:02:00+00:00"),
    ]
    _exchange_with_messages(conn, 100, 1, snowflake_conversation)
    messages = load_batch_messages(conn, [SNOWFLAKE + 1])
    app = ServeApp(conn, messages, "batch1", tmp_path / "scratch", FixedClock(NOW))
    servers = start_all(["127.0.0.1"], 0, app)
    try:
        url = listening_url(servers[0]).rstrip("/")
        status, body, _ = _get(url + f"/api/context?message_id={SNOWFLAKE + 1}")
    finally:
        shutdown_all(servers)
    assert status == 200
    data = json.loads(body)
    assert data["message_id"] == str(SNOWFLAKE + 1)
    assert {m["id"] for m in data["messages"]} == {
        str(SNOWFLAKE),
        str(SNOWFLAKE + 1),
        str(SNOWFLAKE + 2),
    }


def test_page_has_a_context_container_and_c_key_toggle() -> None:
    page = load_page()
    assert "context-row" in page
    assert 'case "c":' in page
    assert 'case "+":' in page
    assert 'case "-":' in page
    assert "contextEnabled" in page
    assert "/api/context" in page


def test_page_help_mentions_the_context_keys() -> None:
    page = load_page()
    assert "context" in page.lower()
