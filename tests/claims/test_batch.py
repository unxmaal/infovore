import argparse
import io
import json
import re
import signal
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from infovore.claims import command
from infovore.claims.command import StopFlag, progress_line
from infovore.claims.extract import TIMEOUT, ClaimsReplyError, extract_concurrently
from infovore.claims.redact import Redacted, RenderedLine
from infovore.cli import ExitCode, main
from infovore.db.claims_v2 import (
    ExchangeOutcome,
    archive_exchange_ids,
    archived_exchange_ids,
    create_run,
    processed_ok,
    record_exchange,
)
from infovore.db.connection import open_database
from tests.cascade_marks import mark
from tests.claims.seed import NOW, conversation, db, environment


class Fake(BaseHTTPRequestHandler):
    lock: ClassVar[threading.Lock] = threading.Lock()
    in_flight = 0
    peak = 0
    seen: ClassVar[list[str]] = []
    delays: ClassVar[dict[str, float]] = {}
    fail_on: ClassVar[set[str]] = set()
    on_request: ClassVar[list[Callable[[str], object]]] = []

    def log_message(self, *args: object) -> None:
        return

    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(404, b"{}")

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][1]["content"]
        tag = re.search(r"Indy (\w+)", user)
        assert tag
        name = tag.group(1)
        with Fake.lock:
            Fake.in_flight += 1
            Fake.peak = max(Fake.peak, Fake.in_flight)
            Fake.seen.append(name)
        try:
            for hook in Fake.on_request:
                hook(name)
            time.sleep(Fake.delays.get(name, 0.0))
            if name in Fake.fail_on:
                self._send(500, b"{}")
                return
            who = re.search(r"\[1\] (user-[0-9a-f]+):", user)
            assert who
            claim = [who.group(1), f"claim {name}", [1]]
            reply = {
                "choices": [{"message": {"content": json.dumps({"c": [claim]})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
            self._send(200, json.dumps(reply).encode())
        finally:
            with Fake.lock:
                Fake.in_flight -= 1


@pytest.fixture
def server() -> Iterator[str]:
    Fake.in_flight = Fake.peak = 0
    Fake.seen, Fake.delays, Fake.fail_on, Fake.on_request = [], {}, set(), []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()
    httpd.server_close()


def go(tmp_path: Path, argv: list[str], exclude: str | None = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    env = environment(tmp_path)
    if exclude:
        env["INFOVORE_EXCLUDE_CHANNELS"] = exclude
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def extract(endpoint: str, *more: str) -> list[str]:
    return ["claims", "extract", "--endpoint", endpoint, "--model", "eval-4b", *more]


def seed(
    tmp_path: Path, names: list[str], channel: str = "c", kind: str | None = "lexicon"
) -> list[int]:
    conn = db(tmp_path)
    cid = sum(map(ord, channel)) + 100
    conn.execute(
        "INSERT OR IGNORE INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (?, 9, NULL, ?, 'text')",
        (cid, channel),
    )
    ids = []
    for name in names:
        base = int(conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM exchanges").fetchone()[0])
        eid, _ = conversation(conn, [(11, "alice", f"my Indy {name}")], base, None)
        conn.execute("UPDATE exchanges SET channel_id = ? WHERE id = ?", (cid, eid))
        if kind:
            mark(conn, eid, kind)
        ids.append(eid)
    conn.commit()
    conn.close()
    return ids


def stored(tmp_path: Path) -> list[tuple[int, str]]:
    conn = open_database(tmp_path / "infovore.db")
    rows = conn.execute("SELECT exchange_id, statement FROM claims_v2 ORDER BY 1")
    return [(r[0], r[1]) for r in rows]


def sent(out: str) -> list[int]:
    return [int(m) for m in re.findall(r"^exchange (\d+)\t", out, re.M)]


def test_channels_select_archived_conversations_in_id_order(tmp_path: Path, server: str) -> None:
    a = seed(tmp_path, ["a1", "a2"], "alpha")
    (b,) = seed(tmp_path, ["b1"], "beta", "residue")
    seed(tmp_path, ["c1"], "gamma", "denylist")
    seed(tmp_path, ["d1"], "delta")
    seed(tmp_path, ["e1"], "alpha", None)

    code, out, _ = go(
        tmp_path,
        extract(server, "--channels", "beta,alpha,gamma", "--limit", "99", "--concurrency", "1"),
    )

    assert code == ExitCode.OK
    assert sent(out) == [*a, b]
    assert sorted(Fake.seen) == ["a1", "a2", "b1"]


def test_channels_combine_with_limit_and_ignore_excluded_channels(
    tmp_path: Path, server: str
) -> None:
    a = seed(tmp_path, ["a1", "a2"], "alpha")
    seed(tmp_path, ["b1"], "beta")

    code, out, _ = go(tmp_path, extract(server, "--channels", "alpha,beta", "--limit", "1"))
    assert code == ExitCode.OK and sent(out) == a[:1]

    code, out, _ = go(
        tmp_path, extract(server, "--channels", "alpha,beta", "--limit", "9"), exclude="beta"
    )
    assert sorted(sent(out)) == a


def test_unknown_channel_is_refused(tmp_path: Path, server: str) -> None:
    seed(tmp_path, ["a1"], "alpha")

    code, _, err = go(tmp_path, extract(server, "--channels", "nope", "--limit", "9"))

    assert code == ExitCode.CONFIG and "unknown channel" in err and "alpha" in err
    assert Fake.seen == []


def test_a_selection_is_required_and_channels_exclude_the_others(
    tmp_path: Path, server: str, capsys: pytest.CaptureFixture[str]
) -> None:
    seed(tmp_path, ["a1"], "alpha")

    code, _, err = go(tmp_path, extract(server, "--limit", "2"))
    assert code == ExitCode.CONFIG and "--slices, --ids, --channels or --archive" in err

    argv = extract(server, "--limit", "2", "--channels", "alpha", "--ids", "1")
    with pytest.raises(SystemExit):
        main(
            argv,
            environ=environment(tmp_path),
            dotenv_path=None,
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
    assert "not allowed with" in capsys.readouterr().err


def test_archived_exchange_ids_helper(tmp_path: Path) -> None:
    a = seed(tmp_path, ["a1", "a2"], "alpha")
    seed(tmp_path, ["b1"], "beta", "embed_irrelevant")
    conn = open_database(tmp_path / "infovore.db")

    assert archived_exchange_ids(conn, frozenset({"alpha", "beta"}), frozenset()) == a
    assert archived_exchange_ids(conn, frozenset({"alpha"}), frozenset({"alpha"})) == []


def test_archive_selects_by_sort_across_channels(tmp_path: Path) -> None:
    lex = seed(tmp_path, ["a1"], "alpha")
    emb = seed(tmp_path, ["b1"], "beta", "embed_relevant")
    und = seed(tmp_path, ["c1", "c2"], "gamma", "residue")
    seed(tmp_path, ["d1"], "delta", "embed_irrelevant")
    seed(tmp_path, ["e1"], "eps", "short_no_tech")
    seed(tmp_path, ["f1"], "zeta", None)
    off = seed(tmp_path, ["g1"], "off")
    seed(tmp_path, ["h1"], "off", "residue")
    conn = open_database(tmp_path / "infovore.db")

    assert archive_exchange_ids(conn, "relevant", frozenset()) == sorted(lex + emb + off)
    assert archive_exchange_ids(conn, "relevant", frozenset({"off"})) == sorted(lex + emb)
    assert archive_exchange_ids(conn, "undecided", frozenset({"off"})) == und


def test_archive_rejects_an_unknown_sort(tmp_path: Path) -> None:
    seed(tmp_path, ["a1"], "alpha")
    conn = open_database(tmp_path / "infovore.db")

    with pytest.raises(ValueError, match="unknown archive sort"):
        archive_exchange_ids(conn, "irrelevant", frozenset())


def test_archive_flag_runs_only_the_chosen_sort(tmp_path: Path, server: str) -> None:
    seed(tmp_path, ["r1"], "alpha")
    seed(tmp_path, ["u1"], "beta", "residue")

    code, _, _ = go(tmp_path, extract(server, "--archive", "relevant", "--limit", "9", "--write"))

    assert code == ExitCode.OK and Fake.seen == ["r1"]
    conn = open_database(tmp_path / "infovore.db")
    assert conn.execute("SELECT selection FROM claim_runs").fetchone()[0] == "archive=relevant"


def test_archive_excludes_the_other_selectors(tmp_path: Path, server: str) -> None:
    with pytest.raises(SystemExit):
        go(tmp_path, extract(server, "--archive", "relevant", "--channels", "alpha"))


def test_resume_skips_done_conversations_and_retries_failures(tmp_path: Path, server: str) -> None:
    seed(tmp_path, ["a1", "a2", "a3"], "alpha")
    Fake.fail_on = {"a2"}
    args = extract(server, "--channels", "alpha", "--limit", "9", "--write", "--resume")

    code, _, _ = go(tmp_path, args)
    assert code == ExitCode.OK and sorted(Fake.seen) == ["a1", "a2", "a3"]

    Fake.seen.clear()
    Fake.fail_on = set()
    code, out, _ = go(tmp_path, args)

    assert code == ExitCode.OK and Fake.seen == ["a2"]
    assert "skipping 2" in out
    assert [s for _, s in stored(tmp_path)] == ["claim a1", "claim a2", "claim a3"]

    Fake.seen.clear()
    code, out, _ = go(tmp_path, args)
    assert code == ExitCode.OK and Fake.seen == [] and "nothing to do" in out
    conn = open_database(tmp_path / "infovore.db")
    assert conn.execute("SELECT COUNT(*) FROM claim_runs").fetchone()[0] == 2


def test_resume_limit_counts_only_conversations_still_to_do(tmp_path: Path, server: str) -> None:
    seed(tmp_path, ["a1", "a2", "a3"], "alpha")
    go(tmp_path, extract(server, "--channels", "alpha", "--limit", "1", "--write"))
    Fake.seen.clear()

    go(tmp_path, extract(server, "--channels", "alpha", "--limit", "1", "--write", "--resume"))

    assert Fake.seen == ["a2"]


def test_resume_needs_write(tmp_path: Path, server: str) -> None:
    seed(tmp_path, ["a1"], "alpha")

    code, _, err = go(tmp_path, extract(server, "--channels", "alpha", "--limit", "1", "--resume"))

    assert code == ExitCode.CONFIG and "--resume needs --write" in err


def test_processed_ok_matches_model_id_and_prompt_hash_and_ignores_failures(
    tmp_path: Path,
) -> None:
    conn = db(tmp_path)
    ids = [conversation(conn, [(1, "a", "x")], n, None)[0] for n in (1, 2, 3)]

    def make(model_id: str, phash: str) -> int:
        return create_run(
            conn,
            endpoint="e",
            model_alias="m",
            model_id=model_id,
            model_id_source="alias",
            prompt_hash=phash,
            selection="s",
            recipe={},
            now=NOW,
        )

    ok = ExchangeOutcome("ok", None, 1, 1, 1, 1.0)
    bad = ExchangeOutcome("failed", "boom", 0, 0, 0, 0.0)
    record_exchange(conn, make("m1", "h1"), ids[0], ok, [], [])
    record_exchange(conn, make("m1", "h1"), ids[1], bad, [], [])
    record_exchange(conn, make("m2", "h1"), ids[2], ok, [], [])
    record_exchange(conn, make("m1", "h2"), ids[2], ok, [], [])

    assert processed_ok(conn, "m1", "h1") == {ids[0]}
    assert processed_ok(conn, "m2", "h1") == {ids[2]}
    assert processed_ok(conn, "m3", "h1") == set()


def test_requests_really_overlap_and_every_conversation_is_committed(
    tmp_path: Path, server: str
) -> None:
    names = ["a1", "a2", "a3", "a4", "a5", "a6"]
    ids = seed(tmp_path, names, "alpha")
    Fake.delays = {"a1": 0.6, "a2": 0.3}

    code, out, _ = go(
        tmp_path,
        extract(server, "--channels", "alpha", "--limit", "9", "--write", "--concurrency", "3"),
    )

    assert code == ExitCode.OK
    assert Fake.peak == 3
    order = sent(out)
    assert sorted(order) == ids and order.index(ids[0]) > order.index(ids[1]) > 0
    assert stored(tmp_path) == [(eid, f"claim {n}") for eid, n in zip(ids, names, strict=True)]


def test_concurrent_and_serial_runs_store_the_same_results(tmp_path: Path, server: str) -> None:
    results = []
    for name, workers in (("one", "1"), ("four", "4")):
        home = tmp_path / name
        home.mkdir()
        seed(home, ["a1", "a2", "a3", "a4"], "alpha")
        Fake.delays = {"a1": 0.3}
        common = ["--channels", "alpha", "--limit", "9", "--write", "--concurrency", workers]
        go(home, extract(server, *common))
        results.append(stored(home))

    assert results[0] == results[1] and len(results[0]) == 4


def test_default_is_eight_in_flight_and_only_absurd_requests_are_capped(
    tmp_path: Path, server: str
) -> None:
    seed(tmp_path, [f"a{n}" for n in range(40)], "alpha")
    Fake.delays = {f"a{n}": 0.3 for n in range(40)}

    go(tmp_path, extract(server, "--channels", "alpha", "--limit", "10"))
    assert Fake.peak == 8

    Fake.peak, Fake.seen = 0, []
    _, out, _ = go(
        tmp_path, extract(server, "--channels", "alpha", "--limit", "12", "--concurrency", "12")
    )
    assert Fake.peak == 12 and "capped" not in out

    Fake.peak, Fake.seen = 0, []
    _, out, _ = go(
        tmp_path, extract(server, "--channels", "alpha", "--limit", "40", "--concurrency", "99")
    )
    assert Fake.peak == 32 and "capped at 32" in out


def test_shuffle_orders_the_selection_by_seed_and_resume_continues_it(
    tmp_path: Path, server: str
) -> None:
    names = [f"a{n}" for n in range(12)]
    seed(tmp_path, names, "alpha")
    base = ["--channels", "alpha", "--concurrency", "1", "--write", "--resume"]

    go(tmp_path, extract(server, *base, "--limit", "4", "--shuffle", "7"))
    first = list(Fake.seen)
    Fake.seen = []
    go(tmp_path, extract(server, *base, "--limit", "12", "--shuffle", "7"))
    rest = list(Fake.seen)

    assert first != names[:4] and sorted(first + rest) == sorted(names)
    assert not set(first) & set(rest)
    conn = open_database(tmp_path / "infovore.db")
    selections = [r[0] for r in conn.execute("SELECT selection FROM claim_runs ORDER BY id")]
    assert selections == ["channels=alpha shuffle=7"] * 2


def test_the_same_seed_gives_the_same_order(tmp_path: Path, server: str) -> None:
    seed(tmp_path, [f"a{n}" for n in range(12)], "alpha")
    args = ["--channels", "alpha", "--concurrency", "1", "--limit", "12", "--shuffle", "3"]

    go(tmp_path, extract(server, *args))
    once = list(Fake.seen)
    Fake.seen = []
    go(tmp_path, extract(server, *args))

    assert Fake.seen == once and sorted(once) != once


def test_concurrency_must_be_positive(tmp_path: Path, server: str) -> None:
    seed(tmp_path, ["a1"], "alpha")

    code, _, err = go(
        tmp_path, extract(server, "--channels", "alpha", "--limit", "9", "--concurrency", "0")
    )

    assert code == ExitCode.CONFIG and "--concurrency must be positive" in err


def test_progress_every_must_be_positive(tmp_path: Path, server: str) -> None:
    seed(tmp_path, ["a1"], "alpha")

    code, _, err = go(
        tmp_path, extract(server, "--channels", "alpha", "--limit", "9", "--progress-every", "0")
    )

    assert code == ExitCode.CONFIG and "--progress-every must be positive" in err


def test_timeouts_are_failures_and_are_retried_on_resume(tmp_path: Path, server: str) -> None:
    seed(tmp_path, ["a1", "a2"], "alpha")
    Fake.delays = {"a1": 1.0, "a2": 1.0}
    args = extract(server, "--channels", "alpha", "--limit", "9", "--write", "--resume")

    code, out, _ = go(tmp_path, [*args, "--timeout", "0.2"])

    assert code == ExitCode.BACKEND and "failed=2" in out
    conn = open_database(tmp_path / "infovore.db")
    assert [r[0] for r in conn.execute("SELECT outcome FROM claim_run_exchanges")] == ["failed"] * 2

    Fake.delays, Fake.seen = {}, []
    code, _, _ = go(tmp_path, args)

    assert code == ExitCode.OK and sorted(Fake.seen) == ["a1", "a2"]


def test_the_default_timeout_is_300_seconds() -> None:
    root = argparse.ArgumentParser()
    command.ClaimsCommand().configure(root)
    top = next(a for a in root._actions if isinstance(a, argparse._SubParsersAction))
    sub = top.choices["extract"]
    defaults = [a.default for a in sub._actions if "--timeout" in a.option_strings]

    assert TIMEOUT == 300.0 and defaults == [300.0]


def test_progress_goes_to_stderr_every_n_conversations(
    tmp_path: Path, server: str, capsys: pytest.CaptureFixture[str]
) -> None:
    seed(tmp_path, ["a1", "a2", "a3", "a4", "a5"], "alpha")

    _, out, _ = go(
        tmp_path, extract(server, "--channels", "alpha", "--limit", "9", "--progress-every", "2")
    )
    err = capsys.readouterr().err

    lines = [line for line in err.splitlines() if line.startswith("progress:")]
    assert len(lines) == 2
    assert re.fullmatch(
        r"progress: 2/5 conversations, claims=2, [\d.]+ conv/hour, eta [\d:]+", lines[0]
    )
    assert lines[1].startswith("progress: 4/5 conversations, claims=4")
    assert "progress:" not in out


def test_progress_line_numbers() -> None:
    assert progress_line(10, 100, 25, 36.0) == (
        "progress: 10/100 conversations, claims=25, 1000.0 conv/hour, eta 0:05:24"
    )
    assert progress_line(0, 5, 0, 0.0) == (
        "progress: 0/5 conversations, claims=0, 0.0 conv/hour, eta n/a"
    )
    assert progress_line(2, 2000, 0, 3600.0).endswith("eta 999:00:00")


def test_a_stop_request_drains_in_flight_commits_them_and_resumes_cleanly(
    tmp_path: Path, server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(tmp_path, ["a1", "a2", "a3", "a4", "a5"], "alpha")
    flags: list[StopFlag] = []

    class Capturing(StopFlag):
        def __init__(self) -> None:
            super().__init__()
            flags.append(self)

    monkeypatch.setattr(command, "StopFlag", Capturing)
    Fake.on_request = [lambda name: flags[0].request() if name == "a2" else None]
    args = extract(server, "--channels", "alpha", "--limit", "9", "--write", "--resume")
    before = signal.getsignal(signal.SIGTERM)

    code, out, _ = go(tmp_path, [*args, "--concurrency", "2"])

    assert code == 130 and "interrupted" in out and "--resume" in out
    assert signal.getsignal(signal.SIGTERM) is before
    assert sorted(Fake.seen) == ["a1", "a2"]
    assert [s for _, s in stored(tmp_path)] == ["claim a1", "claim a2"]

    Fake.on_request, Fake.seen = [], []
    code, _, _ = go(tmp_path, args)
    assert code == ExitCode.OK and sorted(Fake.seen) == ["a3", "a4", "a5"]


def test_signals_are_handled_while_extracting(tmp_path: Path, server: str) -> None:
    seed(tmp_path, ["a1"], "alpha")
    handlers: list[object] = []
    Fake.on_request = [lambda name: handlers.append(signal.getsignal(signal.SIGTERM))]
    before = signal.getsignal(signal.SIGTERM)

    go(tmp_path, extract(server, "--channels", "alpha", "--limit", "9"))

    assert handlers and handlers[0] != before


def test_stop_flag_stops_once_then_raises() -> None:
    flag = StopFlag()
    assert flag.requested is False

    flag.handle(signal.SIGTERM, None)
    assert flag.requested is True

    with pytest.raises(KeyboardInterrupt):
        flag.handle(signal.SIGINT, None)


def test_extract_concurrently_yields_errors_per_conversation() -> None:
    def item(eid: int) -> tuple[int, Redacted]:
        return eid, Redacted([RenderedLine(1, 1, "user-a", "x")], {"user-a"}, [])

    def post(payload: Any) -> tuple[Any, float]:
        raise ClaimsReplyError("down")

    results = list(extract_concurrently([item(1), item(2)], post, "m", 100, 2, lambda: False))

    assert sorted(eid for eid, _ in results) == [1, 2]
    assert all(isinstance(r, ClaimsReplyError) for _, r in results)
