"""The stdlib HTTP layer for `sift serve` (issue #131): one
`socketserver.ThreadingHTTPServer` per bound address (each in its own
daemon thread), all sharing one `infovore.sift.serve.ServeApp` instance so
a label written through one address is visible immediately through
another. No new runtime dependency — `http.server`/`socketserver` only.

The page itself (`templates/serve.html`) is a single self-contained HTML
document with inline CSS/JS and no external requests; this module reads it
once per process (`load_page`) and hands the same string to every bound
address's server.
"""

import contextlib
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from infovore.rows import MessageLabel
from infovore.sift.rules import InvalidRuleError, parse_bulk_rule
from infovore.sift.serve import ServeApp, UnknownMessageError


def load_page() -> str:
    return (resources.files(__package__) / "templates" / "serve.html").read_text(encoding="utf-8")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: ServeApp, page: str) -> None:
        self.app = app
        self.page = page
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: object) -> None:
        # `sift serve` is a local, trusted-LAN tool (issue #131); the
        # stdlib's default per-request access log to stderr is noise here.
        pass

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_html(self.server.page)
            return
        if self.path == "/api/state":
            self._send_json(HTTPStatus.OK, self.server.app.state())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"not found: {self.path}"})

    def do_POST(self) -> None:
        app = self.server.app
        try:
            body = self._read_json_body()
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        try:
            if self.path == "/api/label":
                self._handle_label(app, body)
            elif self.path == "/api/undo":
                self._send_json(HTTPStatus.OK, {"undone": app.undo(), "state": app.state()})
            elif self.path == "/api/trash_channel":
                self._handle_trash_channel(app, body)
            elif self.path == "/api/rules/preview":
                self._handle_preview(app, body)
            elif self.path == "/api/rules/apply":
                self._handle_apply(app, body)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"not found: {self.path}"})
        except (InvalidRuleError, UnknownMessageError, KeyError, ValueError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _handle_label(self, app: ServeApp, body: dict[str, Any]) -> None:
        raw_id = body["message_id"]
        if not isinstance(raw_id, str):
            raise ValueError(
                "message_id must be a string: Discord ids exceed JavaScript's safe integer range"
            )
        message_id = int(raw_id)
        label = MessageLabel(body["label"])
        app.label(message_id, label)
        self._send_json(HTTPStatus.OK, {"state": app.state()})

    def _handle_trash_channel(self, app: ServeApp, body: dict[str, Any]) -> None:
        channel = str(body["channel"])
        trashed = app.trash_rest_of_channel(channel)
        self._send_json(HTTPStatus.OK, {"trashed": trashed, "state": app.state()})

    def _handle_preview(self, app: ServeApp, body: dict[str, Any]) -> None:
        rule = parse_bulk_rule(body)
        preview = app.preview_rule(rule)
        self._send_json(
            HTTPStatus.OK,
            {"matched": preview.matched_count, "conflicts": preview.conflict_count},
        )

    def _handle_apply(self, app: ServeApp, body: dict[str, Any]) -> None:
        rule = parse_bulk_rule(body)
        name = str(body.get("name") or "").strip()
        if not name:
            raise InvalidRuleError("a rule name is required to save it")
        preview, saved_path = app.apply_rule(rule, name)
        self._send_json(
            HTTPStatus.OK,
            {
                "applied": preview.matched_count,
                "conflicts": preview.conflict_count,
                "saved_to": str(saved_path),
                "state": app.state(),
            },
        )


def start_server(host: str, port: int, app: ServeApp, page: str) -> _Server:
    """Bind and start serving one address, in its own daemon thread, so
    binding several addresses (issue #131: "listens on several addresses
    at once") just means calling this once per host."""
    server = _Server((host, port), app, page)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name=f"sift-serve-{host}")
    thread.start()
    return server


def start_all(hosts: list[str], port: int, app: ServeApp) -> list[_Server]:
    page = load_page()
    return [start_server(host, port, app, page) for host in hosts]


def listening_url(server: _Server) -> str:
    host, port = str(server.server_address[0]), server.server_address[1]
    return f"http://{host}:{port}/"


def shutdown_all(servers: list[_Server]) -> None:
    for server in servers:
        server.shutdown()
        server.server_close()


def block_until_interrupted(stop_event: threading.Event) -> None:
    """Keep `infovore sift serve` running in the foreground until Ctrl-C
    (issue #131 is a long-running local server, not a one-shot command).
    `stop_event` is real production `threading.Event` that is never set
    except by the `KeyboardInterrupt` below; tests substitute a stub for
    this whole function so the CLI test suite doesn't block."""
    with contextlib.suppress(KeyboardInterrupt):
        stop_event.wait()
