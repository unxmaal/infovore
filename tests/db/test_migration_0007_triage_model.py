from pathlib import Path

from infovore.db.connection import migrate, open_database


def test_triage_model_and_tokens_tables_and_exchange_columns(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    names = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"triage_model", "triage_tokens"} <= names

    model_columns = {row["name"] for row in conn.execute("PRAGMA table_info(triage_model)")}
    assert model_columns == {"version", "trained_at", "labels_used", "holdout_size", "params_json"}

    token_columns = {row["name"] for row in conn.execute("PRAGMA table_info(triage_tokens)")}
    assert token_columns == {"model_version", "token", "lore_count", "noise_count"}

    exchange_columns = {row["name"] for row in conn.execute("PRAGMA table_info(exchanges)")}
    assert {"p_lore", "p_lore_model"} <= exchange_columns


def test_triage_model_version_autoincrements(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO triage_model (trained_at, labels_used, holdout_size, params_json)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}')"
    )
    first = conn.execute("SELECT version FROM triage_model").fetchone()["version"]
    conn.execute(
        "INSERT INTO triage_model (trained_at, labels_used, holdout_size, params_json)"
        " VALUES ('2026-01-02T00:00:00Z', 25, 5, '{}')"
    )
    second = conn.execute(
        "SELECT version FROM triage_model ORDER BY version DESC LIMIT 1"
    ).fetchone()["version"]
    assert second == first + 1


def test_triage_tokens_reference_triage_model(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO triage_model (trained_at, labels_used, holdout_size, params_json)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}')"
    )
    version = conn.execute("SELECT version FROM triage_model").fetchone()["version"]
    conn.execute(
        "INSERT INTO triage_tokens (model_version, token, lore_count, noise_count)"
        " VALUES (?, 'prom', 3, 1)",
        (version,),
    )
    row = conn.execute("SELECT * FROM triage_tokens WHERE model_version = ?", (version,)).fetchone()
    assert (row["token"], row["lore_count"], row["noise_count"]) == ("prom", 3, 1)
