import io
import json
import sqlite3
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from infovore.claims import novelty
from infovore.claims.novelty import (
    BROAD_CANDIDATES,
    CANDIDATES,
    DESCRIBE_BATCH,
    FOCUS_WEIGHT,
    PAUSE,
    RETRIES,
    TITLE_WEIGHT,
    Candidate,
    Report,
    co_topics,
    common_tokens,
    coverage,
    fetch_descriptions,
    fetch_extract,
    fetch_references,
    focus,
    generic_share,
    match,
    overlap,
    pick_reference,
    queries,
    render,
    score,
    search_titles,
    sentences,
    summarise,
    title_match,
    wiki_get,
)
from infovore.cli import ExitCode, main
from infovore.db.novelty import Reference, novelty_for, references, store_reference
from infovore.db.wiki_tags import create_tag_run, record_tags
from infovore.wiki.build import WikiClaim
from tests.claims.seed import NOW, db, environment
from tests.wiki.seed import add_claim, add_exchange, wiki_db

WORDS = ("octane", "irix", "runs", "psu", "fan", "capacitor", "fails", "workstation")
OCTANE_TEXT = (
    "== Overview ==\n"
    "The Octane series of IRIX workstations was developed and sold by SGI in the 1990s.\n"
    "Octane and Octane2 are two-way multiprocessing-capable workstations based on R10000.\n"
    "Short.\n"
    "== See also ==\n"
)


class Vocab:
    model_id = "fake"
    revision = "r"
    max_tokens = 512

    def token_count(self, text: str) -> int:
        return len(text.split())

    def split(self, text: str, limit: int) -> list[str]:
        return [text]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(w in t.lower()) for w in WORDS] or [1.0] for t in texts]


def claim(i: int, statement: str, topics: set[str]) -> WikiClaim:
    return WikiClaim(i, 1, "2026-01-01", "user-aaaa", statement, frozenset(topics))


def params(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).items()}


class FakeWiki:
    def __init__(
        self,
        search: dict[str, list[str]],
        pages: dict[str, str],
        descriptions: dict[str, str] | None = None,
    ) -> None:
        self.search, self.pages, self.descriptions = search, pages, descriptions or {}
        self.urls: list[str] = []

    def __call__(self, url: str) -> Any:
        self.urls.append(url)
        q = params(url)
        if q.get("list") == "search":
            hits = self.search.get(q["srsearch"], [])
            assert int(q["srlimit"]) in (CANDIDATES, BROAD_CANDIDATES)
            return {"query": {"search": [{"title": t} for t in hits]}}
        if q["prop"] == "description":
            pages = {
                str(i): {"title": t, "description": self.descriptions.get(t, "")}
                for i, t in enumerate(q["titles"].split("|"))
            }
            return {"query": {"pages": pages}}
        title = q["titles"]
        if title not in self.pages:
            return {"query": {"pages": {"-1": {"title": title, "missing": ""}}}}
        page = {"title": title.upper(), "extract": self.pages[title]}
        if title in self.descriptions:
            page["description"] = self.descriptions[title]
        return {"query": {"pages": {"7": page}}}


def test_sentences_drop_headings_and_short_fragments() -> None:
    out = sentences(OCTANE_TEXT)
    assert len(out) == 2 and out[0].startswith("The Octane series") and "==" not in "".join(out)
    assert sentences("") == []


def test_search_and_extract_parse_the_api_shapes() -> None:
    wiki = FakeWiki({"Octane": ["Octane rating", "SGI Octane"]}, {"SGI Octane": OCTANE_TEXT})
    assert search_titles(wiki, "Octane") == ["Octane rating", "SGI Octane"]
    assert fetch_extract(wiki, "SGI Octane") == ("SGI OCTANE", OCTANE_TEXT, "")
    assert fetch_extract(wiki, "Nothing") == ("Nothing", "", "")
    assert search_titles(lambda url: {}, "x") == []
    wiki = FakeWiki({}, {"SGI Octane": OCTANE_TEXT}, {"SGI Octane": "Computer series"})
    assert fetch_extract(wiki, "SGI Octane") == ("SGI OCTANE", OCTANE_TEXT, "Computer series")


def test_fetch_descriptions_fills_found_references_in_batches(tmp_path: Path) -> None:
    conn = wiki_db(tmp_path)
    for i in range(DESCRIBE_BATCH + 1):
        store_reference(conn, Reference(f"T{i}", f"Title {i}", "u", "text", 0.5), NOW)
    store_reference(conn, Reference("Done", "Known", "u", "text", 0.5, "Already"), NOW)
    store_reference(conn, Reference("None", "", "", "", 0.0), NOW)
    wiki = FakeWiki({}, {}, {"Title 0": "Computer series", "Title 20": "MIPS microprocessor"})
    log = io.StringIO()
    assert fetch_descriptions(conn, wiki, log) == 2
    assert len(wiki.urls) == 2 and params(wiki.urls[0])["titles"].count("|") == DESCRIBE_BATCH - 1
    refs = references(conn)
    assert refs["T0"].description == "Computer series" and refs["T20"].description.startswith(
        "MIPS"
    )
    assert refs["T1"].description == "" and refs["Done"].description == "Already"
    assert log.getvalue() == f"described 2/{DESCRIBE_BATCH + 1} references\n"


def test_pick_reference_takes_a_fitting_direct_hit_unless_another_fits_far_better() -> None:
    claims = frozenset({"octane", "irix", "workstations", "sgi", "r10000", "fan"})
    co = ["IRIX", "R10000", "PSU"]
    sgi = Candidate("SGI Octane", OCTANE_TEXT)
    fuel = Candidate("Octane", "Octane is a hydrocarbon; IRIX engines. " * 20, direct=True)
    chosen = pick_reference("Octane", [fuel, sgi], claims, co)
    assert chosen.title == "SGI Octane" and chosen.match == pytest.approx(2 / 3)
    assert chosen.url == "https://en.wikipedia.org/wiki/SGI_Octane"
    linux = Candidate("Linux", "Linux runs IRIX R10000 PSU software. " * 20, direct=True)
    games = Candidate("Video games and Linux", "Linux games on IRIX R10000 PSU. " * 30)
    assert pick_reference("Linux", [games, linux], claims, co).title == "Linux"
    page = Candidate("Octane", "Octane may refer to: fuel; an SGI workstation.", direct=True)
    assert pick_reference("Octane", [page, sgi], claims, co).title == "SGI Octane"
    assert not pick_reference("Octane", [page], claims, co).found
    assert not pick_reference("Octane", [], claims).found


def test_match_weighs_title_focus_and_coverage_and_gates_unrelated_pages() -> None:
    claims = frozenset({"octane", "irix", "fan"})
    co = ["IRIX", "R10000", "PSU"]
    sgi = Candidate("SGI Octane", OCTANE_TEXT)
    expect = TITLE_WEIGHT * (2 / 3) + 2 / 3 + FOCUS_WEIGHT * focus("Octane", OCTANE_TEXT)
    assert match("Octane", sgi, co, claims) == pytest.approx(expect)
    assert match("Octane", Candidate("E", ""), co, claims) == 0.0
    unrelated = Candidate("Pennsylvania", "IRIX R10000 PSU appear here but no topic word.")
    assert match("Octane", unrelated, co, claims) == 0.0
    assert match("Octane", Candidate("Pennsylvania", unrelated.text, direct=True), co, claims) > 0
    assert title_match("IRIX 6.5", "IRIX") == pytest.approx(2 * (1 * (1 / 3)) / (1 + 1 / 3))
    assert title_match("Octane", "Octane") == 1.0 and title_match("Octane", "Fuel") == 0.0
    assert focus("Octane", "octane " * 10 + "x " * 90) == 1.0
    assert focus("Octane", "octane " + "x " * 999) == pytest.approx(0.1)
    assert focus("", "x") == 0.0 and focus("Octane", "") == 0.0
    assert coverage("", co) == 0.0 and coverage("irix", []) == 0.0
    assert overlap("anything", frozenset()) == 0.0 and overlap("", claims) == 0.0
    assert overlap("octane fan", claims) == pytest.approx(2 / (2 * 3) ** 0.5)
    assert pick_reference("Octane", [Candidate("Octane rating", "octane rating fan")], claims).found
    assert not pick_reference("Octane", [Candidate("Octane rating", "zebra")], claims).found


def test_queries_cover_direct_title_co_topics_and_domain_hints() -> None:
    assert queries("Indy", ["O2", "IRIX", "SGI", "PSU"]) == [
        ("Indy", CANDIDATES),
        ('intitle:"Indy" O2', CANDIDATES),
        ("Indy O2 IRIX SGI", BROAD_CANDIDATES),
        ('intitle:"Indy" computer', CANDIDATES),
    ]
    assert queries("Indy", []) == [("Indy", CANDIDATES), ('intitle:"Indy" computer', CANDIDATES)]


def test_co_topics_are_the_most_frequent_other_topics() -> None:
    claims = [claim(1, "a", {"Octane", "IRIX"}), claim(2, "b", {"Octane", "IRIX", "PSU"})]
    assert co_topics("Octane", claims) == ["IRIX", "PSU"]
    assert co_topics("Octane", claims, n=1) == ["IRIX"]
    assert co_topics("Octane", [claim(3, "c", {"Octane"})]) == []


def test_fetch_references_queries_bare_and_co_topic_and_skips_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(novelty, "pause", waits.append)
    conn = db(tmp_path)
    wiki = FakeWiki(
        {"Octane": ["Octane rating"], 'intitle:"Octane" IRIX': ["SGI Octane", "Octane rating"]},
        {"SGI Octane": OCTANE_TEXT, "Octane rating": "Fuel octane rating for engines and cars."},
    )
    by_topic = {
        "Octane": [claim(1, "The Octane runs IRIX workstations", {"Octane", "IRIX"})] * 3,
        "IRIX": [claim(2, "IRIX is an SGI operating system", {"IRIX"})] * 2,
    }
    log = io.StringIO()
    assert fetch_references(conn, wiki, by_topic, NOW, log, limit=1) == 1
    refs = references(conn)
    assert refs["Octane"].title == "SGI OCTANE" and refs["Octane"].found
    searches = [params(u)["srsearch"] for u in wiki.urls if "srsearch" in params(u)]
    assert searches == [
        "Octane",
        'intitle:"Octane" IRIX',
        "Octane IRIX",
        'intitle:"Octane" computer',
    ]
    fetched = [params(u)["titles"] for u in wiki.urls if "titles" in params(u)]
    assert fetched == ["Octane", "Octane rating", "SGI Octane"]
    assert len(waits) == 7
    assert fetch_references(conn, wiki, by_topic, NOW, log) == 1
    assert not references(conn)["IRIX"].found
    assert fetch_references(conn, wiki, by_topic, NOW, log) == 0


def test_fetch_references_logs_progress_every_25(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(novelty, "pause", lambda s: None)
    conn = db(tmp_path)
    by_topic = {f"T{i}": [claim(i, "x", {f"T{i}"})] for i in range(25)}
    log = io.StringIO()
    fetch_references(conn, FakeWiki({}, {}), by_topic, NOW, log)
    assert log.getvalue() == "fetched 25/25 references\n"


def test_generic_share_and_common_tokens() -> None:
    claims = [claim(i, "The Octane runs IRIX", {"Octane"}) for i in range(3)]
    claims.append(claim(9, "A capacitor in the PSU fails", {"PSU"}))
    common = common_tokens(claims, n=2)
    assert len(common) == 2 and common <= {"octane", "runs", "irix"}
    generic = frozenset({"octane", "irix", "runs"})
    assert generic_share("The Octane runs IRIX", generic) == 1.0
    assert generic_share("A capacitor in the PSU fails", generic) == 0.0
    assert generic_share("the", generic) == 1.0


def test_score_finds_the_nearest_reference_sentence_per_claim() -> None:
    refs = {
        "Octane": Reference("Octane", "SGI Octane", "u", OCTANE_TEXT, 0.5),
        "PSU": Reference("PSU", "", "", "", 0.0),
    }
    claims = [
        claim(1, "The Octane runs IRIX", {"Octane", "IRIX"}),
        claim(2, "The PSU fan fails", {"PSU"}),
        claim(3, "A capacitor fails", {"Octane"}),
    ]
    rows = score(Vocab(), claims, refs, frozenset({"octane"}))
    by_id = {r.claim_id: r for r in rows}
    assert by_id[1].reference_title == "SGI Octane"
    assert by_id[1].max_cosine == pytest.approx(2 / 3)
    assert (by_id[1].nearest or "").startswith("The Octane series")
    assert by_id[1].generic_share == 1 / 3
    assert by_id[2].max_cosine is None and by_id[2].nearest is None
    assert by_id[3].max_cosine == 0.0


def test_summarise_and_render_report_distribution_and_examples() -> None:
    refs = {"Octane": Reference("Octane", "SGI Octane", "u", OCTANE_TEXT, 0.5)}
    claims = [claim(i, f"octane irix workstation {i}", {"Octane", "PSU"}) for i in range(4)]
    claims.append(claim(9, "lonely", {"PSU"}))
    scored = score(Vocab(), claims, refs, frozenset())
    report = summarise(claims, scored, refs)
    assert isinstance(report, Report)
    assert report.topics == 1 and report.with_reference == 1
    assert report.scored == 4 and report.unscored == 1
    assert report.by_topic == [("Octane", 4, 1.0)]
    assert report.established[0.9] == 4
    text = render(report)
    assert "claims scored: 4 (no reference for their topics: 1)" in text
    assert "Octane: 100.0% of 4" in text and "least novel:" in text and "most novel:" in text
    assert summarise(claims[:1], scored[:1], refs).quantiles == (0.0, 0.0, 0.0)


def test_wiki_get_sets_a_user_agent_and_retries_with_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    failures = [OSError("no route"), ValueError("bad json")]
    waits: list[float] = []
    monkeypatch.setattr(novelty, "pause", waits.append)

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"ok": True}).encode()

    def urlopen(request: Any, timeout: float) -> Response:
        seen["agent"] = request.get_header("User-agent")
        seen["url"] = request.full_url
        if failures:
            raise failures.pop(0)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert wiki_get("https://en.wikipedia.org/w/api.php?x=1") == {"ok": True}
    assert seen["agent"].startswith("infovore/") and seen["url"].endswith("x=1")
    assert waits == [PAUSE * 4, PAUSE * 8]
    failures[:] = [OSError("flaky")] * (RETRIES - 1)
    assert wiki_get("https://en.wikipedia.org/w/api.php?x=3") == {"ok": True}
    assert len(waits) == 2 + RETRIES - 1
    failures[:] = [OSError("down")] * RETRIES
    with pytest.raises(OSError, match="down"):
        wiki_get("https://en.wikipedia.org/w/api.php?x=2")
    assert len(waits) == 2 * (RETRIES - 1) + 2


def seed(tmp_path: Path) -> sqlite3.Connection:
    conn = wiki_db(tmp_path)
    exchange = add_exchange(conn, 1, "2026-02-01")
    ids = [
        add_claim(conn, exchange, "user-aaaa", "The Octane runs IRIX"),
        add_claim(conn, exchange, "user-bbbb", "The Octane PSU fan fails"),
        add_claim(conn, exchange, "user-cccc", "A capacitor on the Octane fails"),
        add_claim(conn, exchange, "user-dddd", "Indy is small"),
    ]
    run = create_tag_run(
        conn, endpoint="e", model_alias="m", model_id="m", prompt_hash="p", now=NOW
    )
    tags = {ids[0]: ["Octane"], ids[1]: ["Octane"], ids[2]: ["Octane"], ids[3]: ["Indy"]}
    record_tags(conn, run, tags)
    conn.commit()
    return conn


def run_cli(tmp_path: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["claims", "novelty", *argv],
        environ=environment(tmp_path),
        dotenv_path=None,
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


def test_cli_fetches_scores_and_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = seed(tmp_path)
    conn.close()
    wiki = FakeWiki({"Octane": ["SGI Octane"]}, {"SGI Octane": OCTANE_TEXT})
    wiki.descriptions["SGI OCTANE"] = "Computer series"
    monkeypatch.setattr(novelty, "get", wiki)
    monkeypatch.setattr(novelty, "pause", lambda s: None)
    monkeypatch.setattr(novelty, "load_embedder", lambda m, r: Vocab())
    code, out, _ = run_cli(tmp_path, "--run", "1", "--tag-run", "1", "--min-claims", "2")
    assert code == ExitCode.OK and out == "no reference texts; run with --fetch\n"
    code, out, _ = run_cli(
        tmp_path, "--run", "1", "--tag-run", "1", "--min-claims", "2", "--fetch", "--write"
    )
    assert code == ExitCode.OK
    assert "fetched: 1 references\n" in out and "stored: 3 novelty rows\n" in out
    assert "described: 1 references\n" in out
    assert references(db(tmp_path))["Octane"].description == "Computer series"
    assert "topics: 1 with a reference article: 1" in out
    rows = novelty_for(db(tmp_path), [1, 2, 3, 4])
    assert set(rows) == {1, 2, 3}
    assert (rows[1].max_cosine or 0) > (rows[3].max_cosine or 0)
    assert rows[1].reference_title == "SGI OCTANE"
    code, out, _ = run_cli(tmp_path, "--run", "1", "--tag-run", "1", "--min-claims", "2")
    assert code == ExitCode.OK and "stored:" not in out and "least novel:" in out


def test_cli_refuses_unknown_runs_and_bad_floors(tmp_path: Path) -> None:
    seed(tmp_path).close()
    code, _, err = run_cli(tmp_path, "--run", "9", "--tag-run", "1")
    assert code == ExitCode.CONFIG and "unknown run 9" in err
    code, _, err = run_cli(tmp_path, "--run", "1", "--tag-run", "1", "--min-claims", "0")
    assert code == ExitCode.CONFIG and "--min-claims" in err
