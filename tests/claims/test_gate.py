import io
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from infovore.claims.gate import (
    ConversationGate,
    claim_has_tech,
    conversation_stats,
    gated_ids,
    reduction,
)
from infovore.cli import ExitCode, main
from infovore.db.claims_v2 import ClaimIn, ExchangeOutcome, record_exchange, record_review
from infovore.triage.lexicon import load_lexicon
from infovore.wiki.topics import load_topics
from tests.claims.seed import conversation, db, environment, messages_of
from tests.claims.test_store import AT, make_run

OK = ExchangeOutcome("ok", None, 1, 1, 1, 1.0)
TECH = [(1, "ann", "my Indy runs IRIX 6.5 fine"), (2, "bob", "try the PROM first")]
CHAT = [(1, "ann", "hi everyone"), (2, "bob", "lol"), (3, "cy", "hello there friends")]
LEXICON = load_lexicon()


def run(tmp_path: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), environ=environment(tmp_path), dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def seeded_reviews(tmp_path: Path) -> None:
    conn = db(tmp_path)
    run_id = make_run(conn)
    tech, _ = conversation(conn, TECH, 1)
    chat, _ = conversation(conn, CHAT, 2)
    both = [ClaimIn("u", "The Indy runs IRIX", ()), ClaimIn("u", "People are friendly", ())]
    record_exchange(conn, run_id, tech, OK, both, [])
    record_exchange(conn, run_id, chat, OK, [ClaimIn("u", "Everyone said hello", ())], [])
    for claim_id, verdict in ((1, "good"), (2, "not_useful"), (3, "not_useful")):
        record_review(conn, claim_id, verdict, AT)
    conn.close()


def test_stats_are_tech_hits_per_hundred_words_and_substantive_share(tmp_path: Path) -> None:
    conn = db(tmp_path)
    tech, _ = conversation(conn, TECH, 1)
    chat, _ = conversation(conn, CHAT, 2)

    density, share = conversation_stats(LEXICON, messages_of(conn, tech), 4)
    assert density > 20 and share == 1.0
    density, share = conversation_stats(LEXICON, messages_of(conn, chat), 3)
    assert density == 0.0 and share == pytest.approx(1 / 3)
    assert conversation_stats(LEXICON, [], 4) == (0.0, 0.0)


def test_gate_requires_both_thresholds_and_default_passes_everything() -> None:
    assert ConversationGate().passes(0.0, 0.0)
    gate = ConversationGate(min_density=2.0, min_share=0.5)
    assert gate.passes(2.0, 0.5)
    assert not gate.passes(1.9, 0.9)
    assert not gate.passes(5.0, 0.4)
    assert not ConversationGate().active and gate.active


def test_claim_needs_a_lexicon_or_topic_term() -> None:
    topics = load_topics()
    assert claim_has_tech(LEXICON, topics, "The Indy runs IRIX")
    assert claim_has_tech(LEXICON, topics, "The O2 is quiet")
    assert not claim_has_tech(LEXICON, topics, "People are friendly")


def test_gated_ids_keep_order_and_drop_chatter(tmp_path: Path) -> None:
    conn = db(tmp_path)
    tech, _ = conversation(conn, TECH, 1)
    chat, _ = conversation(conn, CHAT, 2)
    again, _ = conversation(conn, TECH, 3)
    gate = ConversationGate(min_density=1.0, min_share=0.5, word_floor=4)

    assert gated_ids(conn, [again, chat, tech], gate, LEXICON) == [again, tech]
    assert gated_ids(conn, [chat], ConversationGate(), LEXICON) == [chat]


def test_reduction_counts_good_lost_and_removed_verdicts() -> None:
    row = reduction({"good": 3, "not_useful": 2, "wrong": 1, "made_up": 1}, {"not_useful": 2})
    assert (row.good_lost, row.not_useful_removed, row.bad_removed) == (0, 2, 0)
    assert row.good_rate == (3, 5)
    cut = reduction({"good": 3, "wrong": 1, "made_up": 2}, {"good": 1, "wrong": 1, "made_up": 2})
    assert (cut.good_lost, cut.bad_removed, cut.good_rate) == (1, 3, (2, 2))


def test_gate_score_reports_conversation_and_claim_tables(tmp_path: Path) -> None:
    seeded_reviews(tmp_path)

    code, out, _ = run(
        tmp_path, "claims", "gate-score", "--runs", "1", "--densities", "0,1", "--shares", "0,0.5"
    )

    assert code == ExitCode.OK
    lines = out.splitlines()
    assert "runs 1: conversations 2, reviewed claims 3 (good 1, not_useful 2)" in out
    assert "good rate 33.3%" in lines[0] or "good rate 33.3%" in out
    hit = [x for x in lines if x.startswith("  density>=1 share>=0.5")]
    assert len(hit) == 1
    assert "dropped 1 (50.0%)" in hit[0] and "good lost 0" in hit[0]
    assert "not_useful removed 1" in hit[0] and "good rate 50.0%" in hit[0]
    base = [x for x in lines if x.startswith("  density>=0 share>=0 ")]
    assert len(base) == 1 and "dropped 0" in base[0]
    claim = [x for x in lines if x.startswith("  no tech term")]
    assert len(claim) == 1
    assert "dropped 2 of 3 reviewed" in claim[0] and "not_useful removed 2" in claim[0]


def test_gate_score_with_nothing_reviewed_says_so(tmp_path: Path) -> None:
    conn = db(tmp_path)
    make_run(conn)
    conn.close()

    code, out, _ = run(tmp_path, "claims", "gate-score", "--runs", "1")

    assert code == ExitCode.OK and "reviewed claims 0" in out and "good rate n/a" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--runs", "x"],
        ["--runs", "9"],
        ["--runs", "1", "--densities", "a"],
        ["--runs", "1", "--shares", "2"],
        ["--runs", "1", "--word-floor", "0"],
    ],
)
def test_gate_score_refuses_bad_arguments(tmp_path: Path, argv: list[str]) -> None:
    seeded_reviews(tmp_path)

    code, _, err = run(tmp_path, "claims", "gate-score", *argv)

    assert code == ExitCode.CONFIG and err


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        return

    def do_GET(self) -> None:
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()
    httpd.server_close()


def extract_argv(endpoint: str, *more: str) -> list[str]:
    return [
        *("claims", "extract", "--endpoint", endpoint, "--model", "m", "--slices", "gold"),
        *("--limit", "5", "--dry-run", *more),
    ]


def test_extract_gate_is_off_by_default(tmp_path: Path, server: str) -> None:
    seeded_reviews(tmp_path)

    _, out, _ = run(tmp_path, *extract_argv(server))

    assert "2 requests for 2 conversations" in out and "gate" not in out


def test_extract_gate_drops_chatter_before_the_limit(tmp_path: Path, server: str) -> None:
    seeded_reviews(tmp_path)

    gate = ("--gate-density", "1", "--gate-share", "0.5")
    code, out, _ = run(tmp_path, *extract_argv(server, *gate))

    assert code == ExitCode.OK
    assert "gate: dropped 1 of 2 conversations" in out
    assert "1 requests for 1 conversations" in out


def test_extract_gate_applies_to_real_runs_too(tmp_path: Path, server: str) -> None:
    seeded_reviews(tmp_path)

    code, out, _ = run(
        tmp_path,
        *("claims", "extract", "--endpoint", server, "--model", "m", "--ids", "2"),
        *("--limit", "1", "--gate-share", "0.9"),
    )

    assert code == ExitCode.OK and "nothing to do" in out


@pytest.mark.parametrize(
    "extra", [["--gate-density", "-1"], ["--gate-share", "1.5"], ["--gate-word-floor", "0"]]
)
def test_extract_gate_refuses_bad_thresholds(tmp_path: Path, server: str, extra: list[str]) -> None:
    seeded_reviews(tmp_path)

    code, _, err = run(tmp_path, *extract_argv(server, *extra))

    assert code == ExitCode.CONFIG and "gate" in err


def test_value_claim_gate_filters_claims_and_reviews(tmp_path: Path) -> None:
    seeded_reviews(tmp_path)

    _, plain, _ = run(tmp_path, "claims", "value", "--runs", "1")
    _, gated, _ = run(tmp_path, "claims", "value", "--runs", "1", "--claim-gate")

    assert "claims 3 " in plain and "good 1 of 3" in plain
    assert "claims 1 " in gated and "good 1 of 1" in gated
