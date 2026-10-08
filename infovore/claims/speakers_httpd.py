import json
import sqlite3
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

from infovore.claims.speakers import SpeakerGroup
from infovore.db.speaker_drops import DECISIONS, decisions, record_decision
from infovore.timing import Clock


def load_page() -> str:
    return (resources.files(__package__) / "templates" / "speakers.html").read_text(
        encoding="utf-8"
    )


class SpeakerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        conn: sqlite3.Connection,
        lock: threading.Lock,
        clock: Clock,
        page: str,
        run_id: int,
        groups: list[SpeakerGroup],
    ) -> None:
        self.conn = conn
        self.lock = lock
        self.clock = clock
        self.page = page
        self.run_id = run_id
        self.groups = groups
        super().__init__(address, _Handler)


def payload(conn: sqlite3.Connection, run_id: int, groups: list[SpeakerGroup]) -> dict[str, Any]:
    decided = decisions(conn)
    return {
        "run": run_id,
        "speakers": [
            {
                "rank": rank,
                "label": g.label,
                "total": g.total,
                "decision": decided.get(g.author_id),
                "sample": [{"statement": c.statement, "verdict": c.verdict} for c in g.sample],
            }
            for rank, g in enumerate(groups)
        ],
    }


class _Handler(BaseHTTPRequestHandler):
    server: SpeakerServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, data: object) -> None:
        self._send(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self._send(HTTPStatus.OK, self.server.page.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/speakers":
            with self.server.lock:
                data = payload(self.server.conn, self.server.run_id, self.server.groups)
            self._send_json(HTTPStatus.OK, data)
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/speaker":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            rank, decision = int(body["rank"]), str(body["decision"])
            if decision not in DECISIONS or not 0 <= rank < len(self.server.groups):
                raise ValueError(decision)
        except (ValueError, KeyError, TypeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "expected {rank, decision}"})
            return
        author = self.server.groups[rank].author_id
        with self.server.lock:
            record_decision(self.server.conn, author, decision, self.server.clock.now())
        self._send_json(HTTPStatus.OK, {"rank": rank, "decision": decision})


def start_all(
    hosts: list[str],
    port: int,
    conn: sqlite3.Connection,
    clock: Clock,
    run_id: int,
    groups: list[SpeakerGroup],
) -> list[SpeakerServer]:
    page = load_page()
    lock = threading.Lock()
    servers = []
    for host in hosts:
        server = SpeakerServer((host, port), conn, lock, clock, page, run_id, groups)
        threading.Thread(target=server.serve_forever, daemon=True, name=f"speakers-{host}").start()
        servers.append(server)
    return servers
