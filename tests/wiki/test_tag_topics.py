import io
from pathlib import Path

from infovore.cli import ExitCode, main
from infovore.db.wiki_tags import create_tag_run, record_tags
from infovore.wiki.build import load_claims
from tests.claims.seed import NOW, db, environment
from tests.wiki.seed import add_claim, add_exchange, wiki_db


def seed(tmp_path: Path) -> dict[str, str]:
    conn = wiki_db(tmp_path)
    exchange = add_exchange(conn, 1, "2026-02-01")
    ids = [
        add_claim(conn, exchange, "user-aaaa", "The SGI Indigo 2 has a boot rom"),
        add_claim(conn, exchange, "user-bbbb", "Indigo2 uses R4400"),
        add_claim(conn, exchange, "user-cccc", "Nothing tagged here"),
        add_claim(conn, exchange, "user-dddd", "Foo Widget is rare"),
        add_claim(conn, exchange, "user-eeee", "Foo widget again"),
    ]
    run = create_tag_run(
        conn, endpoint="e", model_alias="m", model_id="m", prompt_hash="p", now=NOW
    )
    record_tags(
        conn,
        run,
        {
            ids[0]: ["SGI Indigo 2", " "],
            ids[1]: ["Indigo2", "R4400"],
            ids[2]: [],
            ids[3]: ["Foo Widget"],
            ids[4]: ["foo widget", "Foo widget"],
        },
    )
    conn.commit()
    conn.close()
    return environment(tmp_path)


def test_variant_tags_land_on_one_page_with_the_topics_toml_name(tmp_path: Path) -> None:
    seed(tmp_path)
    claims, _ = load_claims(db(tmp_path), tag_runs=[1])
    by_id = {c.claim_id: c.topics for c in claims}
    assert by_id[1] == frozenset({"Indigo2"})
    assert by_id[2] == frozenset({"Indigo2", "R4400"})
    assert by_id[3] == frozenset()
    assert by_id[4] == by_id[5] == frozenset({"Foo Widget"})


def test_without_a_tag_run_topics_toml_is_used(tmp_path: Path) -> None:
    seed(tmp_path)
    claims, _ = load_claims(db(tmp_path))
    assert all("Foo Widget" not in c.topics for c in claims)


def test_cli_build_and_stats_take_a_tag_run(tmp_path: Path) -> None:
    env = seed(tmp_path)
    out = io.StringIO()
    argv = ["wiki", "stats", "--min-claims", "2", "--tag-run", "1"]
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=io.StringIO())
    assert code == ExitCode.OK
    assert "pages: 2" in out.getvalue() and "unassigned: 1" in out.getvalue()
    site = tmp_path / "site"
    argv = ["wiki", "build", "--out", str(site), "--min-claims", "2", "--tag-run", "1"]
    code = main(argv, environ=env, dotenv_path=None, stdout=io.StringIO(), stderr=io.StringIO())
    assert code == ExitCode.OK
    assert (site / "indigo2.md").exists() and (site / "foo-widget.md").exists()
