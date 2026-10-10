from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.cli import ExitCode
from infovore.db.author_ratings import record_rating
from infovore.db.connection import open_database
from infovore.reputation.evidence import SIGNALS, Evidence
from infovore.reputation.people import People
from infovore.reputation.score import BAN_MARGIN, build_reputation
from infovore.reputation.short import ShortMessage, by_rating
from tests.claims.seed import environment
from tests.reputation.test_command import populated, run

NOW = datetime(2026, 1, 1, tzinfo=UTC)
EVIDENCE = Evidence({"1": {"replies": (1.0, 2.0)}}, dict.fromkeys(SIGNALS, 0.5), {})
PEOPLE = People({7: "x", 8: "x"}, frozenset({"y"}), {"x": (7, 8)})


def test_rated_reputation_is_the_rating_with_the_mean_for_the_unrated() -> None:
    reputation = build_reputation(EVIDENCE, PEOPLE, ratings={"1": 3, "x": 1})

    assert reputation.of("1") == 3.0
    assert reputation.of("x") == 1.0
    assert reputation.of("stranger") == 2.0
    assert reputation.of("y") == 1 - BAN_MARGIN


def test_no_ratings_at_all_score_everyone_zero() -> None:
    reputation = build_reputation(EVIDENCE, PEOPLE, ratings={})

    assert reputation.of("1") == 0.0
    assert reputation.floor == -BAN_MARGIN


def test_short_messages_are_bucketed_by_the_authors_rating_through_the_people_file() -> None:
    reputation = build_reputation(EVIDENCE, PEOPLE, ratings={"x": 3, "1": 0})
    messages = [
        ShortMessage(1, 1, 7, keep=True),
        ShortMessage(2, 2, 8, keep=True),
        ShortMessage(3, 3, 1, keep=False),
        ShortMessage(4, 4, 99, keep=True),
    ]

    levels = {level.level: level for level in by_rating(reputation, messages)}

    assert set(levels) == {"0", "1", "2", "3", "unrated"}
    assert (levels["3"].n, levels["3"].kept, levels["3"].rate) == (2, 2, 1.0)
    assert (levels["0"].n, levels["0"].kept) == (1, 0)
    assert levels["unrated"].n == 1
    assert levels["2"].n == 0 and (levels["2"].low, levels["2"].high) == (0.0, 1.0)


def test_without_ratings_there_are_no_levels() -> None:
    assert by_rating(build_reputation(EVIDENCE, PEOPLE), [ShortMessage(1, 1, 1, True)]) == []


def test_eval_with_ratings_reports_the_per_rating_table(tmp_path: Path) -> None:
    conn = populated(tmp_path)
    record_rating(conn, 1, 3, NOW)
    record_rating(conn, 7, 0, NOW)
    conn.commit()
    conn.close()

    code, out, _ = run(["reputation", "eval", "--no-embed", "--ratings"], environment(tmp_path))

    assert code == ExitCode.OK
    assert "hand ratings: 2 persons rated, unrated score 1.500" in out
    assert "  short low-hit by rating 3: kept " in out
    assert "  short low-hit by rating unrated: kept " in out


def test_eval_without_ratings_prints_no_rating_table(tmp_path: Path) -> None:
    populated(tmp_path)

    _, out, _ = run(["reputation", "eval", "--no-embed"], environment(tmp_path))

    assert "hand ratings:" not in out


def test_rate_serves_the_top_authors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from infovore.reputation import command as reputation_command

    populated(tmp_path)
    monkeypatch.setattr(reputation_command, "block_until_interrupted", lambda event: None)
    monkeypatch.setattr(reputation_command, "default_hosts", lambda: ["127.0.0.1"])

    code, out, _ = run(["reputation", "rate", "--top", "3", "--port", "0"], environment(tmp_path))

    assert code == ExitCode.OK
    assert "listening on http://127.0.0.1:" in out
    assert "rating 3 authors (Ctrl-C to stop)" in out


def test_rate_refuses_a_bad_top_or_a_taken_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from infovore.reputation import command as reputation_command

    populated(tmp_path)
    monkeypatch.setattr(reputation_command, "block_until_interrupted", lambda event: None)

    code, _, err = run(["reputation", "rate", "--top", "0"], environment(tmp_path))
    assert code == ExitCode.CONFIG and "--top" in err

    def fail(hosts: list[str], port: int, app: object) -> list[object]:
        raise OSError("in use")

    monkeypatch.setattr(reputation_command, "start_all", fail)
    code, _, err = run(
        ["reputation", "rate", "--host", "127.0.0.1", "--port", "1"], environment(tmp_path)
    )
    assert code == ExitCode.CONFIG and "could not bind port 1" in err


def test_the_migration_creates_the_view(tmp_path: Path) -> None:
    from infovore.db.connection import migrate

    conn = open_database(tmp_path / "m.db")
    migrate(conn)

    assert conn.execute("SELECT COUNT(*) FROM current_author_ratings").fetchone()[0] == 0
