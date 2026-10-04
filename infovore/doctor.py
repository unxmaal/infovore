import argparse
import importlib.util
import json
import shutil
import socket
import sqlite3
import subprocess
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TextIO

from infovore.config import read_dotenv
from infovore.db.connection import load_migrations

ENV_FILE = ".infovore.env"
REQUIRED_VARS = ("INFOVORE_DB_PATH", "INFOVORE_PSEUDONYM_SALT")
RETIRED_VARS = ("INFOVORE_TRIAGE_MIN_P_LORE",)
COUNTED_TABLES = ("current_exchanges", "annotations", "claims_v2")
GATEWAY_MODEL = "eval-7b"
LOCKF = Path("/usr/bin/lockf")
LOCK_PATH = Path("localharness/queue/generation.lock")
MIN_FREE_GIB = 50
GIB = 1024**3
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Check:
    status: Status
    name: str
    detail: str
    hint: str = ""


def run_command(argv: Sequence[str]) -> tuple[int, str] | None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.returncode, (done.stdout + done.stderr).strip()


def fetch_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=5) as response:
        data: bytes = response.read()
    return data


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


@dataclass(frozen=True)
class Host:
    environ: Mapping[str, str]
    home: Path
    run: Callable[[Sequence[str]], tuple[int, str] | None] = run_command
    fetch: Callable[[str], bytes] = fetch_url
    hostname: Callable[[], str] = socket.gethostname
    free_bytes: Callable[[Path], int] = free_bytes
    has_module: Callable[[str], bool] = has_module
    lockf: Path = LOCKF
    project_root: Path = PROJECT_ROOT


def _merged_env(host: Host) -> dict[str, str]:
    merged = read_dotenv(host.home / ENV_FILE)
    merged.update(host.environ)
    return merged


def check_env(host: Host) -> list[Check]:
    env_file = host.home / ENV_FILE
    if env_file.is_file():
        results = [Check(Status.PASS, "env.file", f"{env_file} loads")]
    else:
        results = [Check(Status.FAIL, "env.file", f"{env_file} missing", f"create {env_file}")]
    env = _merged_env(host)
    required = list(REQUIRED_VARS)
    if env.get("INFOVORE_SOURCE") == "export":
        required.append("INFOVORE_EXPORT_DIR")
    else:
        required += ["INFOVORE_DISCORD_TOKEN", "INFOVORE_GUILD_ID"]
    missing = [name for name in required if not env.get(name)]
    if missing:
        results.append(
            Check(Status.FAIL, "env.required", f"missing {', '.join(missing)}", "set in env file")
        )
    else:
        results.append(Check(Status.PASS, "env.required", f"{len(required)} present"))
    if env.get("INFOVORE_EXCLUDE_CHANNELS"):
        results.append(Check(Status.PASS, "env.exclude", "INFOVORE_EXCLUDE_CHANNELS set"))
    else:
        results.append(
            Check(
                Status.FAIL,
                "env.exclude",
                "INFOVORE_EXCLUDE_CHANNELS unset",
                "set it, or excluded channels leak into the archive",
            )
        )
    retired = [name for name in RETIRED_VARS if name in env]
    if retired:
        results.append(
            Check(Status.FAIL, "env.retired", f"retired set: {', '.join(retired)}", "unset them")
        )
    else:
        results.append(Check(Status.PASS, "env.retired", "no retired vars"))
    return results


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def check_db(host: Host, deep: bool) -> list[Check]:
    raw = _merged_env(host).get("INFOVORE_DB_PATH")
    if not raw:
        return [Check(Status.FAIL, "db.file", "INFOVORE_DB_PATH unset", "set INFOVORE_DB_PATH")]
    path = Path(raw).expanduser()
    if not path.is_file():
        return [Check(Status.FAIL, "db.file", f"{path} missing", "restore or copy the database")]
    results = [Check(Status.PASS, "db.file", f"{path} ({path.stat().st_size / GIB:.2f} GiB)")]
    pragma = "integrity_check" if deep else "quick_check"
    try:
        conn = _open_readonly(path)
        verdict = conn.execute(f"PRAGMA {pragma}").fetchone()[0]
    except sqlite3.Error as error:
        results.append(Check(Status.FAIL, "db.open", str(error), "restore from backup"))
        return results
    try:
        if verdict == "ok":
            results.append(Check(Status.PASS, "db.integrity", f"{pragma} ok"))
        else:
            results.append(
                Check(Status.FAIL, "db.integrity", f"{pragma}: {verdict}", "restore from backup")
            )
        results.append(_schema_check(conn))
        results.extend(_count_checks(conn))
    finally:
        conn.close()
    return results


def _schema_check(conn: sqlite3.Connection) -> Check:
    latest = max(migration.version for migration in load_migrations())
    try:
        current = conn.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
    except sqlite3.Error:
        current = None
    if current == latest:
        return Check(Status.PASS, "db.schema", f"at migration {latest}")
    return Check(
        Status.FAIL,
        "db.schema",
        f"at {current}, code expects {latest}",
        "run any infovore command (e.g. infovore status) to migrate",
    )


def _count_checks(conn: sqlite3.Connection) -> list[Check]:
    results: list[Check] = []
    for table in COUNTED_TABLES:
        try:
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        except sqlite3.Error:
            results.append(Check(Status.FAIL, f"db.rows.{table}", "table missing", "migrate"))
            continue
        if count:
            results.append(Check(Status.PASS, f"db.rows.{table}", f"{count} rows"))
        else:
            results.append(
                Check(Status.FAIL, f"db.rows.{table}", "0 rows", "database looks empty or stale")
            )
    return results


def check_python(host: Host) -> list[Check]:
    root = host.project_root
    if host.run(["uv", "--version"]) is None:
        return [Check(Status.FAIL, "py.uv", "uv not found", "install uv")]
    results = [Check(Status.PASS, "py.uv", "uv available")]
    if (root / ".venv").is_dir():
        results.append(Check(Status.PASS, "py.venv", ".venv present"))
    else:
        results.append(Check(Status.FAIL, "py.venv", ".venv missing", "run uv sync"))
    synced = host.run(["uv", "sync", "--check", "--project", str(root)])
    if synced is not None and synced[0] == 0:
        results.append(Check(Status.PASS, "py.lock", "environment matches uv.lock"))
    else:
        results.append(Check(Status.FAIL, "py.lock", "environment differs from lock", "uv sync"))
    if host.has_module("sentence_transformers"):
        results.append(Check(Status.PASS, "py.embed", "embed extra importable"))
    else:
        results.append(
            Check(Status.WARN, "py.embed", "embed extra missing", "uv sync --extra embed")
        )
    return results


def check_gpu_lock(host: Host) -> list[Check]:
    if host.lockf.exists():
        results = [Check(Status.PASS, "gpu.lockf", f"{host.lockf} exists")]
    else:
        results = [
            Check(Status.FAIL, "gpu.lockf", f"{host.lockf} missing", "lockf ships with macOS")
        ]
    lock = host.home / LOCK_PATH
    if lock.exists():
        results.append(Check(Status.PASS, "gpu.lock", f"{lock} exists"))
    else:
        results.append(
            Check(Status.WARN, "gpu.lock", f"{lock} missing", "localharness not yet installed")
        )
    return results


def check_gateway(host: Host, url: str) -> Check:
    endpoint = url.rstrip("/") + "/v1/models"
    try:
        payload = json.loads(host.fetch(endpoint))
        ids = [str(item["id"]) for item in payload["data"]]
    except (OSError, ValueError, KeyError, TypeError):
        return Check(Status.WARN, "gateway", f"{endpoint} unreachable or malformed", "start it")
    if GATEWAY_MODEL in ids:
        return Check(Status.PASS, "gateway", f"{GATEWAY_MODEL} listed")
    return Check(Status.WARN, "gateway", f"{GATEWAY_MODEL} not listed", f"load {GATEWAY_MODEL}")


def _sysctl(host: Host, key: str) -> str | None:
    done = host.run(["sysctl", "-n", key])
    return done[1] if done is not None and done[0] == 0 else None


def check_machine(host: Host) -> list[Check]:
    name = host.hostname()
    if "terminus" in name.lower():
        results = [Check(Status.PASS, "host.name", name)]
    else:
        results = [Check(Status.WARN, "host.name", name, "expected Terminus; wrong machine?")]
    chip = _sysctl(host, "machdep.cpu.brand_string")
    if chip is None:
        results.append(Check(Status.WARN, "host.chip", "unknown", "sysctl failed"))
    else:
        results.append(Check(Status.PASS, "host.chip", chip))
    memory = _sysctl(host, "hw.memsize")
    if memory is None or not memory.isdigit():
        results.append(Check(Status.WARN, "host.memory", "unknown", "sysctl failed"))
    else:
        results.append(Check(Status.PASS, "host.memory", f"{int(memory) // GIB} GiB"))
    free = host.free_bytes(host.home) // GIB
    if free >= MIN_FREE_GIB:
        results.append(Check(Status.PASS, "host.disk", f"{free} GiB free"))
    else:
        results.append(Check(Status.WARN, "host.disk", f"{free} GiB free", "free disk space"))
    return results


def run_checks(host: Host, deep: bool = False, gateway: str | None = None) -> list[Check]:
    results = [*check_env(host), *check_db(host, deep), *check_python(host)]
    results.extend(check_gpu_lock(host))
    if gateway:
        results.append(check_gateway(host, gateway))
    results.extend(check_machine(host))
    return results


def render(results: Sequence[Check], stdout: TextIO) -> int:
    for result in results:
        stdout.write(f"{result.status.value:<4}  {result.name:<22} {result.detail}\n")
        if result.status is not Status.PASS and result.hint:
            stdout.write(f"      fix: {result.hint}\n")
    counts = {status: sum(r.status is status for r in results) for status in Status}
    stdout.write(
        f"{counts[Status.PASS]} pass, {counts[Status.WARN]} warn, {counts[Status.FAIL]} fail\n"
    )
    return 1 if counts[Status.FAIL] else 0


class DoctorCommand:
    name = "doctor"
    standalone = True
    help = "read-only post-migration health check (env, db, python, gpu lock, gateway, host)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--deep", action="store_true", help="full integrity_check (slow)")
        parser.add_argument("--gateway", metavar="URL", help="probe URL/v1/models for eval-7b")

    def run_standalone(
        self, args: argparse.Namespace, environ: Mapping[str, str], stdout: TextIO
    ) -> int:
        host = Host(environ=environ, home=Path.home())
        return render(run_checks(host, deep=args.deep, gateway=args.gateway), stdout)

    async def run(self, context: object, args: argparse.Namespace) -> int:
        raise RuntimeError("doctor runs standalone, without opening the database")
