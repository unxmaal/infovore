"""The stdlib HTTP layer for `infovore judge serve` (#190). One server per
bound address, all sharing one database connection behind one lock, the
same shape as `sift serve`. No new runtime dependency."""

import json
import sqlite3
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlsplit

from infovore.eval.judge import (
    JUDGE_INTERFACE_VERSION,
    NotInExchangeError,
    UnknownQueueItemError,
    exchange_view,
    first_unjudged,
    progress,
    submit,
)
from infovore.timing import Clock

DEFAULT_JUDGE_PORT = 8766


def load_page() -> str:
    return (resources.files(__package__) / "templates" / "judge.html").read_text(encoding="utf-8")


class JudgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        conn: sqlite3.Connection,
        lock: threading.Lock,
        clock: Clock,
        page: str,
    ) -> None:
        self.conn = conn
        self.lock = lock
        self.clock = clock
        self.page = page
        super().__init__(address, _Handler)


def _view_payload(server: JudgeServer, index: int) -> dict[str, Any]:
    view = exchange_view(server.conn, index)
    done, total = progress(server.conn)
    payload = asdict(view)
    payload["marked"] = sorted(view.marked)
    payload["progress"] = {"done": done, "total": total}
    payload["interface_version"] = JUDGE_INTERFACE_VERSION
    return payload


class _Handler(BaseHTTPRequestHandler):
    server: JudgeServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path == "/":
            data = self.server.page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if url.path == "/api/next":
            with self.server.lock:
                index = first_unjudged(self.server.conn)
                done, total = progress(self.server.conn)
            self._send_json(
                HTTPStatus.OK, {"index": index, "progress": {"done": done, "total": total}}
            )
            return
        if url.path == "/api/exchange":
            raw = parse_qs(url.query).get("index", ["0"])[0]
            if not raw.lstrip("-").isdigit():
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "index must be an integer"})
                return
            try:
                with self.server.lock:
                    payload = _view_payload(self.server, int(raw))
            except UnknownQueueItemError as error:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
                return
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
            index = int(body["index"])
            facts = [int(i) for i in body.get("facts", [])]
        except (ValueError, KeyError, TypeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "expected {index, facts: [ids]}"})
            return
        try:
            with self.server.lock:
                written = submit(self.server.conn, index, facts, self.server.clock.now())
                done, total = progress(self.server.conn)
        except UnknownQueueItemError as error:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
            return
        except NotInExchangeError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._send_json(
            HTTPStatus.OK, {"written": written, "progress": {"done": done, "total": total}}
        )


def start_all(
    hosts: list[str], port: int, conn: sqlite3.Connection, clock: Clock
) -> list[JudgeServer]:
    page = load_page()
    lock = threading.Lock()
    servers = []
    for host in hosts:
        server = JudgeServer((host, port), conn, lock, clock, page)
        threading.Thread(target=server.serve_forever, daemon=True, name=f"judge-{host}").start()
        servers.append(server)
    return servers


def listening_url(server: JudgeServer) -> str:
    host, port = str(server.server_address[0]), server.server_address[1]
    return f"http://{host}:{port}/"


def shutdown_all(servers: list[JudgeServer]) -> None:
    for server in servers:
        server.shutdown()
        server.server_close()
