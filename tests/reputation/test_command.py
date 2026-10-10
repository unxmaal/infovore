import io
import json
import sqlite3
from pathlib import Path

import pytest

from infovore.claims.redact import pseudonym
from infovore.cli import ExitCode, builtin_commands, main
from infovore.reputation import compare as compare_module
from infovore.reputation.people import ENV
from tests.claims.seed import SALT, environment
from tests.reputation.world import exchange, label_exchange, label_message, react, reply, world
from tests.triage.test_embed import FakeEmbedder

LORE = "my Indy runs IRIX 6.5.22, PROM says 030-1234-001, hinv shows nothing"
NOISE = "lol gg"


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def populated(tmp_path: Path) -> sqlite3.Connection:
    conn = world(tmp_path)
    base = 0
    for relevant in (True, False):
        for index in range(12):
            base += 1
            author = index % 6 + (1 if relevant else 7)
            text = f"{LORE} {index}" if relevant else f"{NOISE} {index}"
            eid, (message, *_) = exchange(
                conn, [(author, "n", text), (author + 20, "m", NOISE)], base
            )
            label_exchange(conn, eid, relevant)
            if relevant:
                react(conn, message, 2)
                reply(conn, message, author + 1)
            else:
                label_message(conn, message, keep=False)
    for relevant in (True, True, False, False):
        base += 1
        eid, _ = exchange(
            conn, [(1 if relevant else 7, "n", LORE if relevant else NOISE)], base, held_out=True
        )
        label_exchange(conn, eid, relevant)
    conn.commit()
    return conn


def test_reputation_is_a_builtin_command() -> None:
    assert "reputation" in [command.name for command in builtin_commands()]


def test_eval_prints_every_table_without_the_embedding(tmp_path: Path) -> None:
    populated(tmp_path)

    code, out, _ = run(["reputation", "eval", "--no-embed", "--seed", "3"], environment(tmp_path))

    assert code == ExitCode.OK
    assert "reputation eval: seed=3 k_response=20 k_label=10" in out
    assert "held-out: n=4 relevant=2 irrelevant=2" in out
    assert "held-out accuracy at youden threshold" in out
    assert "  held-out lexicon-undecided:" in out
    assert "secondary (leave-one-out): n=24 relevant=12 irrelevant=12" in out
    assert "embed comparison: skipped" in out
    assert "short low-hit messages: n=" in out
    assert "reputation vs message volume (spearman):" in out


def test_eval_json_carries_the_same_numbers(tmp_path: Path) -> None:
    populated(tmp_path)

    code, out, _ = run(["reputation", "eval", "--no-embed", "--json"], environment(tmp_path))

    report = json.loads(out)
    assert code == ExitCode.OK
    assert report["held_out"]["overall"]["n"] == 4
    assert report["embed"] is None
    assert report["short"]["n"] >= 1
    assert set(report["secondary"]["subsets"]) >= {"denylist-wrong", "3+ text messages"}


def test_eval_compares_with_the_embedding_when_it_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    populated(tmp_path)
    monkeypatch.setattr(compare_module, "load_embedder", lambda model, revision: FakeEmbedder())

    code, out, _ = run(["reputation", "eval", "--k-response", "5"], environment(tmp_path))

    assert code == ExitCode.OK
    assert "k_response=5" in out
    for population in ("held_out", "secondary"):
        for name in ("embed", "reputation", "combined"):
            assert f"embed comparison {population} {name}: n=" in out


def test_a_sparse_archive_reports_what_it_cannot_compute(tmp_path: Path) -> None:
    conn = world(tmp_path)
    exchange(conn, [(1, "a", NOISE)], 1)
    conn.commit()

    code, out, _ = run(["reputation", "eval"], environment(tmp_path))

    assert code == ExitCode.OK
    assert "held-out accuracy: n/a (no threshold)" in out
    assert "embed comparison: skipped" in out
    assert "short low-hit messages: none labelled" in out
    assert "auc=n/a [n/a, n/a]" in out
    assert "spearman): n/a" in out


def test_a_people_file_from_the_flag_or_the_environment_is_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    populated(tmp_path)
    people = tmp_path / "people.toml"
    people.write_text('[[person]]\nid = "x"\naccounts = [1, 2]\nbanned = true\n')

    _, flagged, _ = run(
        ["reputation", "eval", "--no-embed", "--people", str(people)], environment(tmp_path)
    )
    monkeypatch.setenv(ENV, str(people))
    _, from_env, _ = run(["reputation", "eval", "--no-embed"], environment(tmp_path))
    monkeypatch.delenv(ENV)
    _, plain, _ = run(["reputation", "eval", "--no-embed"], environment(tmp_path))

    assert "listed=1 banned=1" in flagged
    assert "listed=1 banned=1" in from_env
    assert "listed=0 banned=0" in plain


def test_top_lists_pseudonyms_with_their_evidence(tmp_path: Path) -> None:
    populated(tmp_path)
    people = tmp_path / "people.toml"
    people.write_text('[[person]]\nid = "x"\naccounts = [1, 2]\nbanned = true\n')

    code, out, _ = run(
        ["reputation", "top", "--n", "5", "--people", str(people)], environment(tmp_path)
    )

    rows = [line.split("\t") for line in out.splitlines()]
    assert code == ExitCode.OK
    assert [row[0] for row in rows] == ["1", "2", "3", "4", "5"]
    allowed = {pseudonym(a, SALT) for a in range(1, 40)}
    assert {row[1] for row in rows} <= allowed
    assert pseudonym(1, SALT) not in {row[1] for row in rows}
    assert all(row[3].startswith("messages=") and "replies=" in row[4] for row in rows)


def test_top_lists_everyone_when_there_are_fewer_than_thirty(tmp_path: Path) -> None:
    populated(tmp_path)

    _, out, _ = run(["reputation", "top"], environment(tmp_path))

    assert len(out.splitlines()) == 24


@pytest.mark.parametrize("action", ["eval", "top"])
def test_the_pseudonym_salt_is_required(tmp_path: Path, action: str) -> None:
    world(tmp_path)

    code, _, err = run(["reputation", action], environment(tmp_path, salt=None))

    assert code == ExitCode.CONFIG
    assert "INFOVORE_PSEUDONYM_SALT" in err


def test_a_negative_smoothing_constant_is_refused(tmp_path: Path) -> None:
    world(tmp_path)

    code, _, err = run(["reputation", "eval", "--k-label", "-1"], environment(tmp_path))

    assert code == ExitCode.CONFIG
    assert "must not be negative" in err
