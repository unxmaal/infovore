import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SnapshotReport:
    dest: Path
    size_bytes: int
    user_version: int


def snapshot(conn: sqlite3.Connection, dest: Path | str, force: bool = False) -> SnapshotReport:
    dest_path = Path(dest)
    if dest_path.exists() and not force:
        raise FileExistsError(f"{dest_path} already exists; pass force=True to overwrite")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=dest_path.parent, prefix=f".{dest_path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        dest_conn = sqlite3.connect(tmp_path)
        try:
            conn.backup(dest_conn)
            user_version = int(dest_conn.execute("PRAGMA user_version").fetchone()[0])
        finally:
            dest_conn.close()
        os.replace(tmp_path, dest_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return SnapshotReport(
        dest=dest_path, size_bytes=dest_path.stat().st_size, user_version=user_version
    )
