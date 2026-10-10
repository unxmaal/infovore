import json
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from infovore.db.author_ratings import ratings, record_rating
from infovore.reputation.rate import (
    SAMPLES,
    RateApp,
    UnknownAuthorError,
    author_cards,
    listening_url,
    load_page,
    shutdown_all,
    start_all,
)
from infovore.timing import FixedClock
from tests.reputation.world import world

NOW = datetime(2026, 1, 1, tzinfo=UTC)
LONG = "a message long enough to be shown as a sample of what this author writes about"


def _message(
    conn: sqlite3.Connection,
    message_id: int,
    author: int,
    name: str,
    content: str,
    *,
    bot: bool = False,
    deleted: bool = False,
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, content, deleted_at, ingested_at, raw_json)"
        " VALUES (?, 1, 9, ?, ?, ?, ?, ?, ?, ?, '{}')",
        (
            message_id,
            author,
            name,
            int(bot),
            datetime(2026, 1, 1, 0, message_id % 60, tzinfo=UTC).isoformat(),
            content,
            NOW.isoformat() if deleted else None,
            NOW.isoformat(),
        ),
    )


def populated(tmp_path: Path) -> sqlite3.Connection:
    conn = world(tmp_path)
    for i in range(1, 7):
        _message(conn, i, 1, "alice-old" if i < 6 else "alice", f"{LONG} {i}")
    _message(conn, 10, 2, "bob", "short")
    _message(conn, 11, 2, "bob", LONG)
    _message(conn, 12, 3, "carol", LONG)
    _message(conn, 13, 4, "bot", LONG, bot=True)
    _message(conn, 14, 5, "gone", LONG, deleted=True)
    conn.commit()
    return conn


def test_cards_rank_authors_by_volume_with_latest_name_share_and_samples(tmp_path: Path) -> None:
    cards = author_cards(populated(tmp_path), top=2)

    assert [(c.author_id, c.name, c.messages) for c in cards] == [(1, "alice", 6), (2, "bob", 2)]
    assert cards[0].share == pytest.approx(6 / 9)
    assert len(cards[0].samples) == SAMPLES
    assert all(sample.startswith(LONG) for sample in cards[0].samples)
    assert cards[1].samples == (LONG,)


def test_the_seed_changes_which_samples_are_shown(tmp_path: Path) -> None:
    conn = populated(tmp_path)

    shown = {author_cards(conn, top=1, seed=seed)[0].samples for seed in range(20)}

    assert len(shown) > 1


def test_an_empty_archive_has_no_cards(tmp_path: Path) -> None:
    assert author_cards(world(tmp_path), top=5) == []


def test_ratings_are_append_only_and_the_latest_wins(tmp_path: Path) -> None:
    conn = world(tmp_path)
    record_rating(conn, 1, 1, NOW)
    record_rating(conn, 1, 3, NOW)
    record_rating(conn, 2, 0, NOW)

    assert ratings(conn) == {1: 3, 2: 0}
    with pytest.raises(ValueError, match="rating must be one of"):
        record_rating(conn, 1, 4, NOW)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM author_ratings")


@pytest.fixture
def app(tmp_path: Path) -> RateApp:
    conn = populated(tmp_path)
    return RateApp(conn, author_cards(conn, top=2), FixedClock(NOW))


def test_app_state_reflects_ratings_and_refuses_unknown_authors(app: RateApp) -> None:
    assert app.state()["rated"] == 0
    app.rate(1, 2)
    state = app.state()

    authors = state["authors"]
    assert isinstance(authors, list)
    assert state["rated"] == 1 and state["total"] == 2
    assert [a["rating"] for a in authors] == [2, None]
    assert authors[0]["author_id"] == "1"
    with pytest.raises(UnknownAuthorError):
        app.rate(3, 1)


@pytest.fixture
def base_url(app: RateApp) -> Iterator[str]:
    servers = start_all(["127.0.0.1"], 0, app)
    url = listening_url(servers[0]).rstrip("/")
    try:
        yield url
    finally:
        shutdown_all(servers)


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _post(url: str, payload: object) -> tuple[int, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_page_is_self_contained_and_the_api_round_trips(base_url: str) -> None:
    status, body = _get(base_url + "/")
    assert status == 200
    assert body.decode() == load_page()
    assert 'src="http' not in body.decode() and 'href="http' not in body.decode()

    status, state = _get(base_url + "/api/state")
    assert status == 200
    assert json.loads(state)["total"] == 2

    status, reply = _post(base_url + "/api/rate", {"author_id": "1", "rating": 3})
    assert status == 200
    assert reply["state"]["authors"][0]["rating"] == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"author_id": 1, "rating": 3},
        {"author_id": "x", "rating": 3},
        {"author_id": "1", "rating": 4},
        {"author_id": "1", "rating": True},
        {"author_id": "3", "rating": 1},
        {"rating": 1},
        [1, 2],
    ],
)
def test_bad_rate_requests_are_400(base_url: str, payload: object) -> None:
    status, reply = _post(base_url + "/api/rate", payload)

    assert status == 400
    assert "error" in reply


def test_unknown_paths_are_404(base_url: str) -> None:
    assert _get(base_url + "/nope")[0] == 404
    assert _post(base_url + "/api/nope", {})[0] == 404
