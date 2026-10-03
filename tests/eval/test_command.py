import io
from pathlib import Path

import pytest

from infovore.cli import ExitCode, main
from infovore.db.connection import open_database


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_JUDGE_BACKEND": "fake",
        "INFOVORE_TRIAGE_MIN_P_LORE": "0.9",
    }


def run(argv: list[str], tmp_path: Path) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=environment(tmp_path), dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue() + err.getvalue()


def _seed(tmp_path: Path) -> None:
    run(["status"], tmp_path)
    conn = open_database(tmp_path / "infovore.db")
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'c', 'text')"
    )
    sizes = (1, 2, 4, 9, 20, 60)
    for index in range(1, 1201):
        conn.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id, started_at,"
            " ended_at, message_count, grouping_rule, content_hash, p_lore, triage_score)"
            " VALUES (?, 1, 1, 1, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', ?,"
            " 'quiet_gap', ?, ?, 0.5)",
            (index, sizes[index % len(sizes)], f"h{index}", 0.95 if index <= 900 else 0.1),
        )
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 1, 5, 'hal', '2026-01-01T00:00:00+00:00', 'hi',"
            " '2026-01-01T00:00:00+00:00', '{}')",
            (index,),
        )
        conn.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
            (index, index),
        )
    conn.close()


def test_show_before_freezing_says_so(tmp_path: Path) -> None:
    code, out = run(["slice", "show"], tmp_path)

    assert code == ExitCode.OK
    assert "no slices frozen yet" in out


def test_freeze_then_show_reports_every_slice(tmp_path: Path) -> None:
    _seed(tmp_path)

    code, out = run(["slice", "freeze"], tmp_path)
    assert code == ExitCode.OK
    assert "froze s1: 200 exchanges" in out
    assert "froze gold-repeats: 5 exchanges" in out

    code, out = run(["slice", "show"], tmp_path)
    assert code == ExitCode.OK
    assert out.startswith("s1: 200 exchanges")
    assert "50+" in out

    code, out = run(["slice", "show", "c1"], tmp_path)
    assert out.startswith("c1: 50 exchanges")


def test_a_second_freeze_is_a_configuration_error(tmp_path: Path) -> None:
    _seed(tmp_path)
    run(["slice", "freeze"], tmp_path)

    code, out = run(["slice", "freeze"], tmp_path)

    assert code == ExitCode.CONFIG
    assert "already frozen" in out


def test_report_before_any_judging(tmp_path: Path) -> None:
    _seed(tmp_path)
    run(["slice", "freeze"], tmp_path)

    code, out = run(["judge", "report"], tmp_path)

    assert code == ExitCode.OK
    assert "relevant: 0" in out
    assert "bad_grouping: 0" in out
    assert "slice gold: 0 of 50 judged" in out
    assert "slice gold-repeats: 0 of 5 judged" in out
    assert "uncertain: 0 judged" in out
    assert "self-agreement: n/a" in out
    assert "needed: 200 more relevant to reach 200" in out
    assert "needed: 200 more irrelevant to reach 200" in out


def test_report_shows_counts_and_agreement_once_a_repeat_has_both_passes(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from infovore.eval.judge import frozen_queue, submit

    _seed(tmp_path)
    run(["slice", "freeze"], tmp_path)
    conn = open_database(tmp_path / "infovore.db")
    queue = frozen_queue(conn)
    at = datetime(2026, 10, 2, tzinfo=UTC)
    repeat = next(i for i, item in enumerate(queue) if item.slice_name == "gold-repeats")
    first = next(i for i, item in enumerate(queue) if item.exchange_id == queue[repeat].exchange_id)
    for index in (first, repeat):
        submit(conn, queue, index, queue[index].exchange_id, "relevant", at)
    conn.close()

    _, out = run(["judge", "report"], tmp_path)

    assert "relevant: 1" in out
    assert "self-agreement: 100.0% (1 of 1 repeated exchanges)" in out
    assert "needed: 199 more relevant" in out


def test_serve_refuses_without_a_gold_set(tmp_path: Path) -> None:
    code, out = run(["judge", "serve"], tmp_path)

    assert code == ExitCode.CONFIG
    assert "slice freeze" in out


def test_serve_listens_then_returns_when_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import infovore.sift.httpd

    monkeypatch.setattr(infovore.sift.httpd, "block_until_interrupted", lambda event: None)
    _seed(tmp_path)
    run(["slice", "freeze"], tmp_path)

    code, out = run(["judge", "serve", "--port", "0"], tmp_path)

    assert code == ExitCode.OK
    assert "listening on http://127.0.0.1:" in out


def test_serve_uncertain_needs_no_frozen_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import infovore.sift.httpd

    monkeypatch.setattr(infovore.sift.httpd, "block_until_interrupted", lambda event: None)

    code, out = run(["judge", "serve", "--port", "0", "--queue", "uncertain"], tmp_path)

    assert code == ExitCode.OK
    assert "listening on http://127.0.0.1:" in out


def test_serve_uncertain_accepts_a_scorer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import infovore.sift.httpd

    monkeypatch.setattr(infovore.sift.httpd, "block_until_interrupted", lambda event: None)

    code, _ = run(
        ["judge", "serve", "--port", "0", "--queue", "uncertain", "--scorer", "local-model"],
        tmp_path,
    )

    assert code == ExitCode.OK


def test_builtin_commands_include_slice_and_judge() -> None:
    from infovore.cli import builtin_commands

    names = [command.name for command in builtin_commands()]
    assert "slice" in names
    assert "judge" in names
