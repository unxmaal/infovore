import io
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.cli import ExitCode, main
from infovore.db import archive
from tests.cascade_marks import mark

AT = datetime(2026, 1, 1, tzinfo=UTC)
OPTED_OUT = 66


def environment(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "secret-token",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1,2",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_JUDGE_BACKEND": "fake",
        **extra,
    }


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


class Archive:
    def __init__(self, tmp_path: Path) -> None:
        run(["status"], environment(tmp_path))
        self.conn = sqlite3.connect(tmp_path / "infovore.db")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
            " VALUES (1, 9, NULL, 'hardware', 'text'), (2, 9, NULL, 'offtopic', 'text')"
        )
        self.next_id = 100

    def exchange(
        self,
        messages: list[tuple[int, str]],
        cascade: str | None = "residue",
        channel: int = 1,
        start: int = 0,
    ) -> int:
        ids = []
        for index, (author, content) in enumerate(messages):
            self.next_id += 1
            ids.append(self.next_id)
            self.conn.execute(
                "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
                " created_at, content, ingested_at, raw_json)"
                " VALUES (?, ?, 9, ?, ?, ?, ?, ?, '{}')",
                (
                    self.next_id,
                    channel,
                    author,
                    f"user{author}",
                    (AT + timedelta(days=start, minutes=index)).isoformat(),
                    content,
                    AT.isoformat(),
                ),
            )
        cursor = self.conn.execute(
            "INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,"
            " ended_at, message_count, grouping_rule, content_hash)"
            " VALUES (?, ?, ?, ?, ?, ?, 'quiet_gap', ?)",
            (
                channel,
                ids[0],
                ids[-1],
                (AT + timedelta(days=start)).isoformat(),
                (AT + timedelta(days=start, minutes=len(ids))).isoformat(),
                len(ids),
                f"hash{ids[0]}",
            ),
        )
        exchange_id = cursor.lastrowid
        assert exchange_id is not None
        for position, message_id in enumerate(ids):
            self.conn.execute(
                "INSERT INTO exchange_messages (exchange_id, message_id, position)"
                " VALUES (?, ?, ?)",
                (exchange_id, message_id, position),
            )
        if cascade is not None:
            mark(self.conn, exchange_id, cascade)
        self.conn.commit()
        return exchange_id

    def opt_out(self, user_id: int) -> None:
        self.conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (?, ?)", (user_id, "x"))
        self.conn.commit()


def test_exchanges_rank_by_summed_relevance_then_recency(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "Octane PROM flash"), (2, "the Octane PROM needs 4.3")], start=0)
    a.exchange([(1, "Octane PROM flash")], start=5)
    a.exchange([(1, "Octane PROM flash")], start=9)

    _, out, _ = run(["search", "--exchanges", "Octane", "PROM"], environment(tmp_path))

    ids = [line.split("[ex:")[1].split("]")[0] for line in out.splitlines() if "[ex:" in line]
    assert ids == ["1", "3", "2"]
    assert "3 exchange(s)" in out


def test_exchange_line_has_channel_span_participants_snippet_and_jump_link(
    tmp_path: Path,
) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "Tezro  panel\nbutton"), (2, "agreed"), (1, "Tezro again")])

    _, out, _ = run(["search", "--exchanges", "Tezro"], environment(tmp_path))

    assert "#hardware [ex:1]" in out
    assert "2026-01-01T00:00:00+00:00 .. 2026-01-01T00:03:00+00:00" in out
    assert "2 participant(s), 2 matching message(s)" in out
    assert "> Tezro panel button" in out
    assert "https://discord.com/channels/9/1/101" in out


def test_unarchived_exchanges_are_hidden_unless_all(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "IRIX gate pass")], cascade="lexicon")
    a.exchange([(1, "IRIX gate fail")], cascade="bayes_irrelevant")
    a.exchange([(1, "IRIX no p_lore pass")], cascade="residue")
    a.exchange([(1, "IRIX untriaged")], cascade=None)
    a.exchange([(1, "IRIX denied")], cascade="denylist")
    a.exchange([(1, "IRIX textless")], cascade="no_text")

    _, default, _ = run(["search", "--exchanges", "IRIX"], environment(tmp_path))
    _, everything, _ = run(["search", "--exchanges", "--all", "IRIX"], environment(tmp_path))

    assert "gate pass" in default and "no p_lore pass" in default
    assert "gate fail" not in default and "untriaged" not in default
    assert "REJECTED" not in default
    assert "gate fail" in everything and "untriaged" in everything
    assert "gate pass" in everything
    assert "denied" not in default and "textless" not in default
    assert everything.count("REJECTED") == 4


def test_opted_out_author_never_appears_even_before_redaction(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "Octane secret handshake"), (OPTED_OUT, "Octane private remark")])
    a.opt_out(OPTED_OUT)

    _, out, _ = run(["search", "--exchanges", "Octane"], environment(tmp_path))

    assert "private remark" not in out
    assert "secret handshake" in out
    assert "1 participant(s), 1 matching message(s)" in out


def test_exchange_search_honours_the_channel_denylist(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "Octane in hardware")], channel=1)
    a.exchange([(1, "Octane in offtopic")], channel=2)

    _, out, _ = run(
        ["search", "--exchanges", "Octane"],
        environment(tmp_path, INFOVORE_EXCLUDE_CHANNELS="offtopic"),
    )

    assert "in hardware" in out
    assert "offtopic" not in out


def test_exchange_search_limits_and_truncates(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "Octane " + "x" * 500)], start=0)
    a.exchange([(1, "Octane short")], start=1)

    _, out, _ = run(["search", "--exchanges", "--limit", "1", "Octane"], environment(tmp_path))
    _, full, _ = run(["search", "--exchanges", "Octane"], environment(tmp_path))

    assert "1 exchange(s)" in out
    assert "x" * 201 not in full
    assert "..." in full


def test_exchange_search_reports_no_match(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "something else")])

    _, none, _ = run(["search", "--exchanges", "Octane"], environment(tmp_path))
    _, empty, _ = run(["search", "--exchanges", "-"], environment(tmp_path))

    assert "no exchanges for 'Octane'" in none
    assert "no exchanges for '-'" in empty


def test_export_keeps_only_archived_non_opted_out_content(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "Octane keep me"), (OPTED_OUT, "Octane opted out words")], start=0)
    a.exchange([(1, "Octane rejected exchange")], cascade="bayes_irrelevant", start=1)
    a.exchange([(1, "Octane [redacted]"), (1, "[redacted]")], start=2, channel=2)
    a.conn.execute("UPDATE messages SET deleted_at = 'x' WHERE content = 'Octane [redacted]'")
    a.conn.commit()
    a.opt_out(OPTED_OUT)
    dest = tmp_path / "out" / "archive.db"

    code, out, _ = run(["export-archive", str(dest)], environment(tmp_path))

    assert code == ExitCode.OK
    assert "(1 exchanges, 1 messages" in out
    shared = sqlite3.connect(dest)
    tables = {r[0] for r in shared.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"channels", "exchanges", "messages", "messages_fts"} <= tables
    assert not tables & {"opt_outs", "claims", "message_revisions", "attachments"}
    assert [r[0] for r in shared.execute("SELECT content FROM messages")] == ["Octane keep me"]
    assert shared.execute("SELECT message_count FROM exchanges").fetchall() == [(1,)]
    assert shared.execute("SELECT name FROM channels").fetchall() == [("hardware",)]
    columns = {r[1] for r in shared.execute("PRAGMA table_info(messages)")}
    assert "author_id" not in columns and "raw_json" not in columns
    opted = shared.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH '\"opted\"'")
    assert opted.fetchall() == []
    kept = shared.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH '\"Octane\"'")
    assert len(kept.fetchall()) == 1
    shared.close()
    blob = dest.read_bytes()
    assert b"opted out words" not in blob and b"rejected exchange" not in blob


def test_search_and_export_agree_on_which_messages_are_visible(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "Octane keep me"), (1, "Octane deleted words"), (1, "[redacted]")])
    a.conn.execute("UPDATE messages SET deleted_at = 'x' WHERE content = 'Octane deleted words'")
    a.conn.commit()
    dest = tmp_path / "agree.db"
    run(["export-archive", str(dest)], environment(tmp_path))
    exported = {r[0] for r in sqlite3.connect(dest).execute("SELECT content FROM messages")}

    _, octane, _ = run(["search", "--exchanges", "Octane"], environment(tmp_path))
    _, redacted, _ = run(["search", "--exchanges", "redacted"], environment(tmp_path))

    assert exported == {"Octane keep me"}
    assert "deleted words" not in octane and "keep me" in octane
    assert "no exchanges" in redacted


def test_export_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "Octane")])
    dest = tmp_path / "archive.db"
    dest.write_text("precious")

    code, out, _ = run(["export-archive", str(dest)], environment(tmp_path))
    assert code == ExitCode.FAILURE
    assert "already exists" in out
    assert dest.read_text() == "precious"

    code, _, _ = run(["export-archive", "--force", str(dest)], environment(tmp_path))
    assert code == ExitCode.OK
    assert sqlite3.connect(dest).execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)


def test_export_honours_the_channel_denylist(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "Octane hardware")], channel=1)
    a.exchange([(1, "Octane offtopic")], channel=2)
    dest = tmp_path / "archive.db"

    run(
        ["export-archive", str(dest)],
        environment(tmp_path, INFOVORE_EXCLUDE_CHANNELS="offtopic"),
    )

    contents = [r[0] for r in sqlite3.connect(dest).execute("SELECT content FROM messages")]
    assert contents == ["Octane hardware"]


def test_failed_export_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "Octane")])
    dest = tmp_path / "archive.db"

    def boom(*args: object) -> tuple[int, int]:
        raise RuntimeError("disk full")

    monkeypatch.setattr(archive, "_write", boom)
    code, _, err = run(["export-archive", str(dest)], environment(tmp_path))

    assert code == ExitCode.FAILURE
    assert "disk full" in err
    assert list(tmp_path.glob(".archive.db.*")) == []
    assert not dest.exists()


def test_snippets_are_capped_per_exchange(tmp_path: Path) -> None:
    a = Archive(tmp_path)
    a.exchange([(1, "Octane one"), (1, "Octane two"), (1, "Octane three")])

    _, out, _ = run(["search", "--exchanges", "Octane"], environment(tmp_path))

    assert "3 matching message(s)" in out
    assert out.count("  > ") == 2
