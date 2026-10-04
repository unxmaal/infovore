import argparse
import dataclasses
import http.server
import io
import json
import sqlite3
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from infovore import doctor
from infovore.cli import main
from infovore.db.connection import load_migrations, open_database
from infovore.doctor import Check, DoctorCommand, Host, Status

GOOD_ENV = {
    "INFOVORE_DB_PATH": "",
    "INFOVORE_PSEUDONYM_SALT": "salt-value",
    "INFOVORE_DISCORD_TOKEN": "tok",
    "INFOVORE_GUILD_ID": "1",
    "INFOVORE_EXCLUDE_CHANNELS": "trains",
}


def fake_run(
    uv: bool = True, sync_ok: bool = True, sysctl: dict[str, str] | None = None
) -> "Callable[[Sequence[str]], tuple[int, str] | None]":
    facts = {"machdep.cpu.brand_string": "Apple M5 Ultra", "hw.memsize": str(256 * 1024**3)}
    facts.update(sysctl or {})

    def run(argv: Sequence[str]) -> tuple[int, str] | None:
        if argv[0] == "uv":
            if not uv:
                return None
            return (0, "uv 1") if argv[1] == "--version" else (0 if sync_ok else 1, "")
        key = argv[-1]
        return (0, facts[key]) if key in facts else (1, "")

    return run


def seeded_db(path: Path, empty: Sequence[str] = ()) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)")
    for migration in load_migrations():
        conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (migration.version,))
    conn.execute("CREATE TABLE exchanges (id INTEGER PRIMARY KEY, superseded_by_recipe INTEGER)")
    conn.execute(
        "CREATE VIEW current_exchanges AS"
        " SELECT * FROM exchanges WHERE superseded_by_recipe IS NULL"
    )
    conn.execute("CREATE TABLE annotations (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE claims_v2 (id INTEGER PRIMARY KEY)")
    for table in ("exchanges", "annotations", "claims_v2"):
        if table not in empty:
            conn.execute(f"INSERT INTO {table} (id) VALUES (1)")
    conn.commit()
    conn.close()


def host(tmp_path: Path, env: dict[str, str] | None = None, **kwargs: object) -> Host:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    merged = dict(GOOD_ENV)
    merged["INFOVORE_DB_PATH"] = str(tmp_path / "x.db")
    merged.update(env or {})
    lockf = tmp_path / "lockf"
    lockf.write_text("")
    defaults: dict[str, object] = {
        "run": fake_run(),
        "hostname": lambda: "Terminus.local",
        "free_bytes": lambda path: 900 * 1024**3,
        "has_module": lambda name: True,
        "lockf": lockf,
        "project_root": tmp_path,
    }
    defaults.update(kwargs)
    return Host(environ=merged, home=home, **defaults)  # type: ignore[arg-type]


def by_name(results: list[Check]) -> dict[str, Check]:
    return {result.name: result for result in results}


def test_env_all_good_and_values_never_printed(tmp_path: Path) -> None:
    h = host(tmp_path)
    (h.home / ".infovore.env").write_text("INFOVORE_PSEUDONYM_SALT=from-file\n")
    results = doctor.check_env(h)
    assert {r.status for r in results} == {Status.PASS}
    assert "salt-value" not in "".join(r.detail for r in results)


def test_env_file_missing_and_vars_missing(tmp_path: Path) -> None:
    h = host(tmp_path, {"INFOVORE_PSEUDONYM_SALT": "", "INFOVORE_EXCLUDE_CHANNELS": ""})
    results = by_name(doctor.check_env(h))
    assert results["env.file"].status is Status.FAIL
    assert "INFOVORE_PSEUDONYM_SALT" in results["env.required"].detail
    assert results["env.exclude"].status is Status.FAIL


def test_env_read_from_dotenv_and_export_source(tmp_path: Path) -> None:
    h = host(tmp_path, {"INFOVORE_SOURCE": "export"})
    environ = {k: v for k, v in h.environ.items() if k != "INFOVORE_PSEUDONYM_SALT"}
    h = dataclasses.replace(h, environ=environ)
    (h.home / ".infovore.env").write_text("INFOVORE_PSEUDONYM_SALT=x\n")
    results = by_name(doctor.check_env(h))
    assert results["env.required"].status is Status.FAIL
    assert results["env.required"].detail == "missing INFOVORE_EXPORT_DIR"


def test_env_retired_var_fails(tmp_path: Path) -> None:
    h = host(tmp_path, {"INFOVORE_TRIAGE_MIN_P_LORE": "0.5"})
    result = by_name(doctor.check_env(h))["env.retired"]
    assert result.status is Status.FAIL
    assert "INFOVORE_TRIAGE_MIN_P_LORE" in result.detail


def test_db_unset_and_missing_and_not_created(tmp_path: Path) -> None:
    unset = host(tmp_path, {"INFOVORE_DB_PATH": ""})
    assert doctor.check_db(unset, deep=False)[0].detail == "INFOVORE_DB_PATH unset"
    missing = host(tmp_path)
    result = doctor.check_db(missing, deep=False)
    assert [r.status for r in result] == [Status.FAIL]
    assert not (tmp_path / "x.db").exists()


def test_db_healthy_quick_and_deep(tmp_path: Path) -> None:
    seeded_db(tmp_path / "x.db")
    h = host(tmp_path)
    quick = doctor.check_db(h, deep=False)
    assert {r.status for r in quick} == {Status.PASS}
    assert by_name(quick)["db.integrity"].detail == "quick_check ok"
    assert by_name(doctor.check_db(h, deep=True))["db.integrity"].detail == "integrity_check ok"


def test_db_check_is_read_only(tmp_path: Path) -> None:
    seeded_db(tmp_path / "x.db")
    before = (tmp_path / "x.db").stat().st_mtime_ns
    doctor.check_db(host(tmp_path), deep=True)
    assert (tmp_path / "x.db").stat().st_mtime_ns == before


def test_db_empty_tables_fail(tmp_path: Path) -> None:
    seeded_db(tmp_path / "x.db", empty=["annotations"])
    results = by_name(doctor.check_db(host(tmp_path), deep=False))
    assert results["db.rows.annotations"].status is Status.FAIL
    assert results["db.rows.claims_v2"].status is Status.PASS


def test_db_old_schema_and_missing_table(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)")
    conn.execute("INSERT INTO schema_migrations (version) VALUES (3)")
    conn.close()
    results = by_name(doctor.check_db(host(tmp_path), deep=False))
    assert results["db.schema"].status is Status.FAIL
    assert "at 3" in results["db.schema"].detail
    assert results["db.rows.claims_v2"].detail == "table missing"


def test_db_without_migration_table(tmp_path: Path) -> None:
    open_database(tmp_path / "x.db").close()
    results = by_name(doctor.check_db(host(tmp_path), deep=False))
    assert "at None" in results["db.schema"].detail


def test_db_not_a_database(tmp_path: Path) -> None:
    (tmp_path / "x.db").write_bytes(b"this is not sqlite" * 100)
    results = doctor.check_db(host(tmp_path), deep=False)
    assert results[-1].name == "db.open"
    assert results[-1].status is Status.FAIL


def test_db_integrity_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seeded_db(tmp_path / "x.db")

    class Corrupt:
        def execute(self, sql: str) -> "Corrupt":
            return self

        def fetchone(self) -> list[str]:
            return ["row 1 missing"]

        def close(self) -> None:
            return None

    monkeypatch.setattr(doctor, "_open_readonly", lambda path: Corrupt())
    monkeypatch.setattr(doctor, "_schema_check", lambda conn: Check(Status.PASS, "s", ""))
    monkeypatch.setattr(doctor, "_count_checks", lambda conn: [])
    results = by_name(doctor.check_db(host(tmp_path), deep=False))
    assert results["db.integrity"].status is Status.FAIL


def test_python_all_good(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    results = doctor.check_python(host(tmp_path))
    assert {r.status for r in results} == {Status.PASS}


def test_python_uv_missing(tmp_path: Path) -> None:
    results = doctor.check_python(host(tmp_path, run=fake_run(uv=False)))
    assert [(r.name, r.status) for r in results] == [("py.uv", Status.FAIL)]


def test_python_no_venv_drift_no_embed(tmp_path: Path) -> None:
    h = host(tmp_path, run=fake_run(sync_ok=False), has_module=lambda name: False)
    results = by_name(doctor.check_python(h))
    assert results["py.venv"].status is Status.FAIL
    assert results["py.lock"].status is Status.FAIL
    assert results["py.embed"].status is Status.WARN


def test_python_sync_check_unavailable(tmp_path: Path) -> None:
    calls: list[Sequence[str]] = []

    def run(argv: Sequence[str]) -> tuple[int, str] | None:
        calls.append(argv)
        return (0, "") if argv[1] == "--version" else None

    results = by_name(doctor.check_python(host(tmp_path, run=run)))
    assert results["py.lock"].status is Status.FAIL


def test_gpu_lock_present_and_absent(tmp_path: Path) -> None:
    h = host(tmp_path)
    lock = h.home / "localharness" / "queue" / "generation.lock"
    assert by_name(doctor.check_gpu_lock(h))["gpu.lock"].status is Status.WARN
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    assert {r.status for r in doctor.check_gpu_lock(h)} == {Status.PASS}


def test_gpu_lockf_missing(tmp_path: Path) -> None:
    h = host(tmp_path, lockf=tmp_path / "nope")
    assert by_name(doctor.check_gpu_lock(h))["gpu.lockf"].status is Status.FAIL


def test_gateway_variants(tmp_path: Path) -> None:
    def serving(body: bytes) -> "Callable[[str], bytes]":
        def fetch(url: str) -> bytes:
            assert url == "http://gw:1/v1/models"
            return body

        return fetch

    def down(url: str) -> bytes:
        raise OSError("refused")

    ok = host(tmp_path, fetch=serving(json.dumps({"data": [{"id": "eval-7b"}]}).encode()))
    assert doctor.check_gateway(ok, "http://gw:1/").status is Status.PASS
    other = host(tmp_path, fetch=serving(json.dumps({"data": [{"id": "x"}]}).encode()))
    assert doctor.check_gateway(other, "http://gw:1").status is Status.WARN
    bad = host(tmp_path, fetch=serving(b"not json"))
    assert doctor.check_gateway(bad, "http://gw:1").status is Status.WARN
    shape = host(tmp_path, fetch=serving(b"[]"))
    assert doctor.check_gateway(shape, "http://gw:1").status is Status.WARN
    assert doctor.check_gateway(host(tmp_path, fetch=down), "http://gw:1").status is Status.WARN


def test_machine_facts_pass(tmp_path: Path) -> None:
    results = by_name(doctor.check_machine(host(tmp_path)))
    assert {r.status for r in results.values()} == {Status.PASS}
    assert results["host.chip"].detail == "Apple M5 Ultra"
    assert results["host.memory"].detail == "256 GiB"
    assert results["host.disk"].detail == "900 GiB free"


def test_machine_wrong_host_and_missing_facts(tmp_path: Path) -> None:
    h = host(
        tmp_path,
        run=fake_run(sysctl={"hw.memsize": "garbage"}),
        hostname=lambda: "Monolith",
        free_bytes=lambda path: 1024**3,
    )
    results = by_name(doctor.check_machine(h))
    assert results["host.name"].status is Status.WARN
    assert results["host.memory"].status is Status.WARN
    assert results["host.disk"].status is Status.WARN
    nothing = host(tmp_path, run=lambda argv: None)
    assert by_name(doctor.check_machine(nothing))["host.chip"].status is Status.WARN


def test_run_checks_and_render_exit_codes(tmp_path: Path) -> None:
    seeded_db(tmp_path / "x.db")
    (tmp_path / ".venv").mkdir()
    h = host(tmp_path, fetch=lambda url: b'{"data": [{"id": "eval-7b"}]}')
    (h.home / ".infovore.env").write_text("")
    out = io.StringIO()
    results = doctor.run_checks(h, gateway="http://gw")
    assert "gateway" in by_name(results)
    code = doctor.render(results, out)
    assert code == 0
    assert "0 fail" in out.getvalue()
    bad = io.StringIO()
    failing = [Check(Status.FAIL, "a", "broke", "do x"), Check(Status.WARN, "b", "meh")]
    assert doctor.render(failing, bad) == 1
    assert "fix: do x" in bad.getvalue()
    assert "fix:" not in bad.getvalue().split("fix: do x")[1]


def test_run_checks_without_gateway_skips_probe(tmp_path: Path) -> None:
    assert "gateway" not in by_name(doctor.run_checks(host(tmp_path)))


def test_default_probes_work() -> None:
    assert doctor.run_command(["true"]) == (0, "")
    assert doctor.run_command(["definitely-not-a-binary-xyz"]) is None
    assert doctor.has_module("json")
    assert not doctor.has_module("no_such_module_xyz")
    assert doctor.free_bytes(Path(".")) > 0


def test_fetch_url_reads_local_server() -> None:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"hi")

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert doctor.fetch_url(f"http://127.0.0.1:{server.server_port}/") == b"hi"
    finally:
        server.shutdown()
        server.server_close()


def stub_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    real = doctor.run_checks

    def stubbed(h: Host, deep: bool = False, gateway: str | None = None) -> list[Check]:
        quiet = dataclasses.replace(h, run=fake_run(), fetch=lambda url: b"{}")
        return real(quiet, deep=deep, gateway=gateway)

    monkeypatch.setattr(doctor, "run_checks", stubbed)


def test_cli_doctor_runs_without_creating_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    stub_probes(monkeypatch)
    db = tmp_path / "sub" / "never.db"
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["doctor"],
        environ={"INFOVORE_DB_PATH": str(db)},
        dotenv_path=None,
        stdout=out,
        stderr=err,
    )
    assert code == 1
    assert "FAIL  db.file" in out.getvalue()
    assert not db.parent.exists()


def test_cli_doctor_uses_process_environment_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    stub_probes(monkeypatch)
    monkeypatch.setenv("INFOVORE_DB_PATH", str(tmp_path / "absent.db"))
    out = io.StringIO()
    assert main(["doctor", "--deep", "--gateway", "http://127.0.0.1:9"], stdout=out) == 1
    assert "gateway" in out.getvalue()


def test_doctor_command_run_is_not_the_entry_point() -> None:
    import asyncio

    parser = argparse.ArgumentParser()
    DoctorCommand().configure(parser)
    assert parser.parse_args(["--deep"]).deep is True
    with pytest.raises(RuntimeError):
        asyncio.run(DoctorCommand().run(object(), parser.parse_args([])))
