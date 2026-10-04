import io
from pathlib import Path

import pytest

from infovore.cli import ExitCode, main
from infovore.db.connection import open_database
from infovore.db.reviewed_words import approved_words
from tests.cascade_marks import mark
from tests.triage.test_human import seed


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_JUDGE_BACKEND": "fake",
    }


def run(argv: list[str], tmp_path: Path) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=environment(tmp_path), dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue() + err.getvalue()


def populate(tmp_path: Path) -> None:
    run(["status"], tmp_path)
    conn = open_database(tmp_path / "infovore.db")
    for index, text in enumerate(("ubr ubr g5 zxqv hello", "ubr blorp", "ubr")):
        mark(conn, seed(conn, index + 1, text), "residue")


def test_report_on_an_empty_review(tmp_path: Path) -> None:
    populate(tmp_path)

    code, out = run(["words", "report"], tmp_path)

    assert code == ExitCode.OK
    assert "reviewed: 0 (approved 0, not tech 0)" in out
    assert "undecided conversations: 3" in out
    assert "with an approved word: 0" in out
    assert "candidates: 4" in out


def test_report_shows_the_top_candidates_on_request(tmp_path: Path) -> None:
    populate(tmp_path)

    code, out = run(["words", "report", "--show", "2"], tmp_path)

    assert code == ExitCode.OK
    assert "ubr\t4" in out
    assert out.count("\t") == 2


def test_report_counts_decisions_and_projected_gain(tmp_path: Path) -> None:
    populate(tmp_path)
    conn = open_database(tmp_path / "infovore.db")
    conn.execute(
        "INSERT INTO reviewed_words (word, tech, decided_at)"
        " VALUES ('g5', 1, 'x'), ('blorp', 0, 'x')"
    )

    code, out = run(["words", "report"], tmp_path)

    assert code == ExitCode.OK
    assert "reviewed: 2 (approved 1, not tech 1)" in out
    assert "with an approved word: 1" in out
    assert "candidates: 2" in out


def test_report_top_n_controls_the_common_word_list(tmp_path: Path) -> None:
    populate(tmp_path)

    _, small = run(["words", "report", "--top-n", "10"], tmp_path)
    _, default = run(["words", "report"], tmp_path)

    assert "candidates: 5" in small
    assert "candidates: 4" in default


def test_serve_listens_and_decisions_land_in_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import infovore.sift.httpd

    populate(tmp_path)
    monkeypatch.setattr(infovore.sift.httpd, "block_until_interrupted", lambda event: None)

    code, out = run(["words", "serve", "--port", "0"], tmp_path)

    assert code == ExitCode.OK
    assert "4 candidate words" in out
    assert "listening on http://127.0.0.1:" in out
    assert approved_words(open_database(tmp_path / "infovore.db")) == frozenset()


def test_serve_accepts_hosts_and_a_common_word_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import infovore.sift.httpd

    populate(tmp_path)
    monkeypatch.setattr(infovore.sift.httpd, "block_until_interrupted", lambda event: None)

    code, out = run(
        ["words", "serve", "--port", "0", "--host", "127.0.0.1", "--top-n", "10"], tmp_path
    )

    assert code == ExitCode.OK
    assert "listening on" in out
