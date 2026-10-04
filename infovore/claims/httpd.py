import json
import sqlite3
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

from infovore.db.claims_v2 import VERDICTS, claim_exists, record_review, review_rows
from infovore.timing import Clock

DEFAULT_CLAIMS_PORT = 8768


def load_page() -> str:
    return (resources.files(__package__) / "templates" / "claims.html").read_text(encoding="utf-8")


class ClaimServer(ThreadingHTTPServer):
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
    ) -> None:
        self.conn = conn
        self.lock = lock
        self.clock = clock
        self.page = page
        self.run_id = run_id
        super().__init__(address, _Handler)


def payload(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    rows = review_rows(conn, run_id)
    claims = [
        {
            "id": r.claim_id,
            "exchange": r.exchange_id,
            "speaker": r.speaker,
            "statement": r.statement,
            "verdict": r.verdict,
            "sources": [{"id": i, "author": a, "text": t} for i, a, t in r.sources],
        }
        for r in rows
    ]
    first = next((i for i, r in enumerate(rows) if r.verdict is None), len(rows))
    return {"claims": claims, "first_unreviewed": first, "run": run_id}


class _Handler(BaseHTTPRequestHandler):
    server: ClaimServer

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
        elif path == "/api/claims":
            with self.server.lock:
                data = payload(self.server.conn, self.server.run_id)
            self._send_json(HTTPStatus.OK, data)
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/review":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            claim_id, verdict = int(body["claim_id"]), str(body["verdict"])
            if verdict not in VERDICTS:
                raise ValueError(verdict)
        except (ValueError, KeyError, TypeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "expected {claim_id, verdict}"})
            return
        with self.server.lock:
            known = claim_exists(self.server.conn, claim_id)
            if known:
                record_review(self.server.conn, claim_id, verdict, self.server.clock.now())
        if not known:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "unknown claim"})
            return
        self._send_json(HTTPStatus.OK, {"claim_id": claim_id, "verdict": verdict})


def start_all(
    hosts: list[str], port: int, conn: sqlite3.Connection, clock: Clock, run_id: int
) -> list[ClaimServer]:
    page = load_page()
    lock = threading.Lock()
    servers = []
    for host in hosts:
        server = ClaimServer((host, port), conn, lock, clock, page, run_id)
        threading.Thread(target=server.serve_forever, daemon=True, name=f"claims-{host}").start()
        servers.append(server)
    return servers


def listening_url(server: ClaimServer) -> str:
    host, port = str(server.server_address[0]), server.server_address[1]
    return f"http://{host}:{port}/"


def shutdown_all(servers: list[ClaimServer]) -> None:
    for server in servers:
        server.shutdown()
        server.server_close()
