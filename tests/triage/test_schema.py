from pathlib import Path

from infovore.db.connection import migrate, open_database


def test_exchanges_carry_triage_columns_and_index(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(exchanges)")}
    assert {"triage_score", "triage_reasons", "triage_version"} <= columns
    indexes = {row["name"] for row in conn.execute("PRAGMA index_list(exchanges)")}
    assert "exchanges_triage_score" in indexes
