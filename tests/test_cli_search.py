import io
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from infovore.cli import ExitCode, main

AT = datetime(2026, 1, 1, tzinfo=UTC)


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "secret-token",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1,2",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_JUDGE_BACKEND": "fake",
    }


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def seed(tmp_path: Path, messages: list[tuple[int, str]], author: str = "hal") -> None:
    """Writes through a plain connection AFTER the schema exists, so the
    triggers installed by 0021 are what indexes these rows."""
    run(["status"], environment(tmp_path))
    conn = sqlite3.connect(tmp_path / "infovore.db")
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 9, NULL, 'hardware', 'text')"
    )
    for index, (message_id, content) in enumerate(messages):
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 9, 5, ?, ?, ?, ?, '{}')",
            (
                message_id,
                author,
                (AT + timedelta(minutes=index)).isoformat(),
                content,
                AT.isoformat(),
            ),
        )
    conn.commit()
    conn.close()


def test_search_finds_a_message_by_a_distinctive_term(tmp_path: Path) -> None:
    seed(tmp_path, [(1, "the Octane2 PSU is rated at 330W"), (2, "unrelated lunch chatter")])

    code, out, _ = run(["search", "Octane2"], environment(tmp_path))

    assert code == ExitCode.OK
    assert "Octane2 PSU is rated at 330W" in out
    assert "lunch" not in out


def test_search_reports_where_the_hit_came_from(tmp_path: Path) -> None:
    """A hit nobody can place in a conversation is not useful (issue #182)."""
    seed(tmp_path, [(1, "the Tezro front panel button can be disabled")])

    _, out, _ = run(["search", "Tezro"], environment(tmp_path))

    assert "#hardware" in out
    assert "hal" in out
    assert "2026-01-01" in out


def test_search_takes_several_terms_without_shell_quoting(tmp_path: Path) -> None:
    seed(tmp_path, [(1, "the Octane2 takes V12 graphics"), (2, "the Octane2 alone")])

    _, out, _ = run(["search", "Octane2", "V12"], environment(tmp_path))

    assert "V12 graphics" in out
    assert "Octane2 alone" not in out


def test_search_honours_the_limit(tmp_path: Path) -> None:
    seed(tmp_path, [(index, "repeated Onyx mention") for index in range(1, 11)])

    _, out, _ = run(["search", "Onyx", "--limit", "3"], environment(tmp_path))

    assert out.count("repeated Onyx mention") == 3


def test_search_says_so_when_nothing_matches(tmp_path: Path) -> None:
    seed(tmp_path, [(1, "ordinary text")])

    code, out, _ = run(["search", "Nonexistentterm"], environment(tmp_path))

    assert code == ExitCode.OK
    assert "no matches" in out


def test_an_all_punctuation_query_matches_nothing_rather_than_everything(tmp_path: Path) -> None:
    seed(tmp_path, [(1, "ordinary text")])

    code, out, _ = run(["search", "-"], environment(tmp_path))

    assert code == ExitCode.OK
    assert "no matches" in out


def test_search_shows_surrounding_messages_with_context(tmp_path: Path) -> None:
    seed(
        tmp_path,
        [
            (1, "anybody know the part number"),
            (2, "the board is 030-1234-567 rev B"),
            (3, "thanks, ordering one now"),
        ],
    )

    _, out, _ = run(["search", "030-1234-567", "--context", "1"], environment(tmp_path))

    assert "anybody know the part number" in out
    assert "thanks, ordering one now" in out


def test_context_does_not_reach_into_another_channel(tmp_path: Path) -> None:
    seed(tmp_path, [(1, "the board is 030-1234-567 rev B")])
    conn = sqlite3.connect(tmp_path / "infovore.db")
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (2, 9, NULL, 'offtopic', 'text')"
    )
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (2, 2, 9, 5, 'hal', ?, 'completely unrelated', ?, '{}')",
        ((AT + timedelta(minutes=1)).isoformat(), AT.isoformat()),
    )
    conn.commit()
    conn.close()

    _, out, _ = run(["search", "030-1234-567", "--context", "3"], environment(tmp_path))

    # The positive half matters: a bare "not in" assertion passes vacuously on
    # empty output, so it could never fail for the right reason.
    assert "030-1234-567 rev B" in out
    assert "completely unrelated" not in out


def test_context_marks_which_message_was_the_hit(tmp_path: Path) -> None:
    seed(tmp_path, [(1, "some preamble"), (2, "the Indigo2 Impact board")])

    _, out, _ = run(["search", "Indigo2", "--context", "1"], environment(tmp_path))

    hit_line = next(line for line in out.splitlines() if "Indigo2 Impact" in line)
    preamble_line = next(line for line in out.splitlines() if "some preamble" in line)
    assert hit_line.startswith(">")
    assert not preamble_line.startswith(">")


def test_a_redacted_message_is_not_searchable_from_the_cli(tmp_path: Path) -> None:
    """The privacy seam, driven through the real entry point rather than the
    query helper: `redact_stored` clears content AND author_name_at_time, and
    an external-content FTS5 index keeps its own tokens (issue #182)."""
    seed(tmp_path, [(1, "contact me at averydistinctivehandle")])
    conn = sqlite3.connect(tmp_path / "infovore.db")
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (5, ?)", (AT.isoformat(),))
    conn.commit()
    conn.close()

    env = environment(tmp_path)
    from infovore.db.connection import open_database
    from infovore.privacy.optout import redact_stored

    live = open_database(tmp_path / "infovore.db")
    redact_stored(live, frozenset({5}))
    live.close()

    code, out, _ = run(["search", "averydistinctivehandle"], env)

    assert code == ExitCode.OK
    assert "no matches" in out
    # The echoed query holds the term because the user typed it; what must
    # never come back is the stored message it was indexed from.
    assert "contact me at" not in out


def test_builtin_commands_include_search() -> None:
    from infovore.cli import builtin_commands

    assert "search" in [command.name for command in builtin_commands()]
