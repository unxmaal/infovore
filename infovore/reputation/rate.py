import json
import sqlite3
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Final

from infovore.db.author_ratings import RATINGS, ratings, record_rating
from infovore.timing import Clock

DEFAULT_TOP: Final = 200
DEFAULT_PORT: Final = 8769
SAMPLES: Final = 3
SAMPLE_MIN_CHARS: Final = 60

_AUTHORS: Final = (
    "SELECT author_id, COUNT(*) AS n,"
    " (SELECT author_name_at_time FROM messages x WHERE x.author_id = m.author_id"
    "  ORDER BY x.created_at DESC, x.id DESC LIMIT 1) AS name"
    " FROM messages m WHERE deleted_at IS NULL AND author_is_bot = 0"
    " GROUP BY author_id ORDER BY n DESC, author_id LIMIT ?"
)
_SAMPLES: Final = (
    "SELECT content FROM messages WHERE author_id = ? AND deleted_at IS NULL"
    " AND length(trim(content)) >= ? ORDER BY ((id + ?) * 2654435761) % 4294967296 LIMIT ?"
)


class UnknownAuthorError(Exception):
    pass


@dataclass(frozen=True)
class AuthorCard:
    author_id: int
    name: str
    messages: int
    share: float
    samples: tuple[str, ...]


def author_cards(conn: sqlite3.Connection, top: int, seed: int = 0) -> list[AuthorCard]:
    total = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE deleted_at IS NULL AND author_is_bot = 0"
    ).fetchone()[0]
    cards = []
    for row in conn.execute(_AUTHORS, (top,)).fetchall():
        samples = conn.execute(
            _SAMPLES, (row["author_id"], SAMPLE_MIN_CHARS, seed, SAMPLES)
        ).fetchall()
        cards.append(
            AuthorCard(
                row["author_id"],
                row["name"],
                row["n"],
                row["n"] / total if total else 0.0,
                tuple(sample[0] for sample in samples),
            )
        )
    return cards


class RateApp:
    def __init__(self, conn: sqlite3.Connection, cards: list[AuthorCard], clock: Clock) -> None:
        self._conn = conn
        self._cards = cards
        self._ids = frozenset(card.author_id for card in cards)
        self._clock = clock
        self._lock = threading.Lock()

    def cards(self) -> list[AuthorCard]:
        return list(self._cards)

    def state(self) -> dict[str, object]:
        with self._lock:
            rated = ratings(self._conn)
        return {
            "authors": [
                {
                    "author_id": str(card.author_id),
                    "name": card.name,
                    "messages": card.messages,
                    "share": card.share,
                    "samples": list(card.samples),
                    "rating": rated.get(card.author_id),
                }
                for card in self._cards
            ],
            "rated": sum(1 for card in self._cards if card.author_id in rated),
            "total": len(self._cards),
        }

    def rate(self, author_id: int, rating: int) -> None:
        if author_id not in self._ids:
            raise UnknownAuthorError(author_id)
        with self._lock:
            record_rating(self._conn, author_id, rating, self._clock.now())
            self._conn.commit()


def load_page() -> str:
    return (resources.files(__package__) / "templates" / "rate.html").read_text(encoding="utf-8")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: RateApp, page: str) -> None:
        self.app = app
        self.page = page
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(HTTPStatus.OK, self.server.page.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send_json(HTTPStatus.OK, self.server.app.state())
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"not found: {self.path}"})

    def do_POST(self) -> None:
        if self.path != "/api/rate":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"not found: {self.path}"})
            return
        try:
            body = self._body()
            author_id, rating = body["author_id"], body["rating"]
            if not isinstance(author_id, str) or not author_id.isdigit():
                raise ValueError("author_id must be a numeric string")
            if isinstance(rating, bool) or rating not in RATINGS:
                raise ValueError(f"rating must be one of {list(RATINGS)}")
            self.server.app.rate(int(author_id), int(rating))
        except (KeyError, ValueError, UnknownAuthorError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._send_json(HTTPStatus.OK, {"state": self.server.app.state()})

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        parsed = json.loads(self.rfile.read(length) if length else b"{}")
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed


def start_all(hosts: list[str], port: int, app: RateApp) -> list[_Server]:
    page = load_page()
    servers = []
    for host in hosts:
        server = _Server((host, port), app, page)
        threading.Thread(target=server.serve_forever, daemon=True, name=f"rate-{host}").start()
        servers.append(server)
    return servers


def listening_url(server: _Server) -> str:
    host, port = server.server_address[0], server.server_address[1]
    return f"http://{host!s}:{port}/"


def shutdown_all(servers: list[_Server]) -> None:
    for server in servers:
        server.shutdown()
        server.server_close()
