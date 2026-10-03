import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

REBUILD_MARKER = "-- rebuild: foreign_keys off"
MIGRATION_NAME = re.compile(r"^(\d+)_[A-Za-z0-9_]+\.sql$")


class MigrationError(Exception):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


def open_database(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def load_migrations(directory: Traversable | Path | None = None) -> list[Migration]:
    source = directory if directory is not None else resources.files(__package__) / "migrations"
    found: dict[int, Migration] = {}
    for entry in source.iterdir():
        match = MIGRATION_NAME.match(entry.name)
        if match is None:
            continue
        version = int(match.group(1))
        if version in found:
            raise MigrationError(f"duplicate migration version {version}: {entry.name}")
        found[version] = Migration(version, entry.name.removesuffix(".sql"), entry.read_text())
    return [found[version] for version in sorted(found)]


def applied_versions(conn: sqlite3.Connection) -> list[int]:
    ensure_migration_table(conn)
    rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    return [row[0] for row in rows]


def ensure_migration_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version INTEGER PRIMARY KEY,"
        " applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))"
    )


def migrate(conn: sqlite3.Connection, migrations: Sequence[Migration] | None = None) -> list[int]:
    pending_source = migrations if migrations is not None else load_migrations()
    done = set(applied_versions(conn))
    newly_applied: list[int] = []
    for migration in pending_source:
        if migration.version in done:
            continue
        rebuild = REBUILD_MARKER in migration.sql
        if rebuild:
            conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.executescript(
                "BEGIN IMMEDIATE;\n"
                f"{migration.sql}\n;\n"
                f"INSERT INTO schema_migrations (version) VALUES ({migration.version});\n"
                f"PRAGMA user_version = {migration.version};\n"
                "COMMIT;"
            )
            if rebuild and conn.execute("PRAGMA foreign_key_check").fetchall():
                raise MigrationError(f"{migration.name} left dangling foreign keys")
        except sqlite3.Error as error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise MigrationError(f"{migration.name} failed: {error}") from error
        finally:
            if rebuild:
                conn.execute("PRAGMA foreign_keys = ON")
        newly_applied.append(migration.version)
    return newly_applied
