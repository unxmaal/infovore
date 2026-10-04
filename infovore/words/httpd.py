import json
import sqlite3
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlsplit

from infovore.db.reviewed_words import record_decisions
from infovore.timing import Clock
from infovore.words.review import ReviewQueue, UnknownWordError

DEFAULT_WORDS_PORT = 8767
DEFAULT_PAGE_SIZE = 40
MAX_PAGE_SIZE = 1000


def load_page() -> str:
    return (resources.files(__package__) / "templates" / "words.html").read_text(encoding="utf-8")


class WordServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        conn: sqlite3.Connection,
        lock: threading.Lock,
        clock: Clock,
        page: str,
        queue: ReviewQueue,
    ) -> None:
        self.conn = conn
        self.lock = lock
        self.clock = clock
        self.page = page
        self.queue = queue
        super().__init__(address, _Handler)


def _size(raw: object) -> int:
    size = int(str(raw))
    if not 1 <= size <= MAX_PAGE_SIZE:
        raise ValueError("size out of range")
    return size


def _payload(queue: ReviewQueue, size: int) -> dict[str, Any]:
    return {"rows": queue.page(size), "remaining": queue.remaining}


class _Handler(BaseHTTPRequestHandler):
    server: WordServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path == "/":
            self._send(HTTPStatus.OK, self.server.page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if url.path == "/api/page":
            try:
                size = _size(parse_qs(url.query).get("size", [DEFAULT_PAGE_SIZE])[0])
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "size must be 1-1000"})
                return
            with self.server.lock:
                payload = _payload(self.server.queue, size)
            self._send_json(HTTPStatus.OK, payload)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/submit":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            words, tech = body["words"], body["tech"]
            if not (isinstance(words, list) and isinstance(tech, list)):
                raise TypeError("lists expected")
            size = _size(body.get("size", DEFAULT_PAGE_SIZE))
            if not set(tech) <= set(words):
                raise ValueError("tech must be a subset of words")
            checked = set(tech)
            decisions = {str(word): word in checked for word in words}
        except (ValueError, KeyError, TypeError):
            self._send_json(
                HTTPStatus.BAD_REQUEST, {"error": "expected {words: [...], tech: [...], size}"}
            )
            return
        try:
            with self.server.lock:
                self.server.queue.check(list(decisions))
                record_decisions(self.server.conn, decisions, self.server.clock.now())
                self.server.queue.mark_decided(list(decisions))
                payload = _payload(self.server.queue, size)
        except UnknownWordError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._send_json(HTTPStatus.OK, payload)


def start_all(
    hosts: list[str],
    port: int,
    conn: sqlite3.Connection,
    clock: Clock,
    queue: ReviewQueue,
) -> list[WordServer]:
    page = load_page()
    lock = threading.Lock()
    servers = []
    for host in hosts:
        server = WordServer((host, port), conn, lock, clock, page, queue)
        threading.Thread(target=server.serve_forever, daemon=True, name=f"words-{host}").start()
        servers.append(server)
    return servers


def listening_url(server: WordServer) -> str:
    host, port = str(server.server_address[0]), server.server_address[1]
    return f"http://{host}:{port}/"


def shutdown_all(servers: list[WordServer]) -> None:
    for server in servers:
        server.shutdown()
        server.server_close()
