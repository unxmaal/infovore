import json
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.reviewed_words import approved_words, decided_words
from infovore.words.httpd import listening_url, load_page, shutdown_all, start_all
from infovore.words.review import ReviewQueue
from tests.triage.test_human import db


class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)


Served = tuple[str, sqlite3.Connection]


@pytest.fixture
def served(tmp_path: Path) -> Iterator[Served]:
    conn = db(tmp_path)
    queue = ReviewQueue([("ubr", 9), ("g5", 5), ("lunch", 2)])
    servers = start_all(["127.0.0.1"], 0, conn, FixedClock(), queue)
    yield listening_url(servers[0]), conn
    shutdown_all(servers)


def call(url: str, body: object | None = None) -> tuple[int, dict[str, object]]:
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data)) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_page_is_self_contained_html(served: Served) -> None:
    url, _ = served
    with urllib.request.urlopen(url) as response:
        text = response.read().decode()

    assert text == load_page()
    assert 'id="all"' in text
    assert "Enter" in text


def test_a_page_of_rows_comes_highest_frequency_first(served: Served) -> None:
    url, _ = served
    status, payload = call(f"{url}api/page?size=2")

    assert status == 200
    assert payload["rows"] == [["ubr", 9], ["g5", 5]]
    assert payload["remaining"] == 3


def test_a_bad_size_is_refused(served: Served) -> None:
    url, _ = served

    assert call(f"{url}api/page?size=x")[0] == 400
    assert call(f"{url}api/page?size=0")[0] == 400
    assert call(f"{url}api/page")[0] == 200


def test_submit_records_checked_and_unchecked_and_returns_the_next_page(served: Served) -> None:
    url, conn = served
    status, payload = call(f"{url}api/submit", {"words": ["ubr", "g5"], "tech": ["ubr"], "size": 5})

    assert status == 200
    assert payload["rows"] == [["lunch", 2]]
    assert payload["remaining"] == 1
    assert decided_words(conn) == {"ubr", "g5"}
    assert approved_words(conn) == {"ubr"}


def test_a_malformed_submit_is_refused(served: Served) -> None:
    url, conn = served

    assert call(f"{url}api/submit", {"words": "ubr"})[0] == 400
    assert call(f"{url}api/submit", {"tech": []})[0] == 400
    assert call(f"{url}api/submit", {"words": ["nope"], "tech": []})[0] == 400
    assert call(f"{url}api/submit", {"words": ["ubr"], "tech": ["g5"]})[0] == 400
    assert call(f"{url}api/submit", {"words": ["ubr"], "tech": [], "size": "x"})[0] == 400
    assert decided_words(conn) == frozenset()


def test_unknown_paths_are_not_found(served: Served) -> None:
    url, _ = served

    assert call(f"{url}nope")[0] == 404
    assert call(f"{url}nope", {})[0] == 404


def test_every_bound_address_shares_one_lock(tmp_path: Path) -> None:
    conn = db(tmp_path)
    servers = start_all(["127.0.0.1", "127.0.0.1"], 0, conn, FixedClock(), ReviewQueue([]))
    try:
        assert servers[0].lock is servers[1].lock
    finally:
        shutdown_all(servers)
