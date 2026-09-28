from pathlib import Path

from infovore.db.connection import migrate, open_database


def test_message_model_and_tokens_tables_exist(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    names = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"message_model", "message_tokens"} <= names

    model_columns = {row["name"] for row in conn.execute("PRAGMA table_info(message_model)")}
    assert model_columns == {"version", "trained_at", "labels_used", "holdout_size", "params_json"}

    token_columns = {row["name"] for row in conn.execute("PRAGMA table_info(message_tokens)")}
    assert token_columns == {"model_version", "token", "trash_count", "keep_count"}


def test_message_model_version_autoincrements(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}')"
    )
    first = conn.execute("SELECT version FROM message_model").fetchone()["version"]
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json)"
        " VALUES ('2026-01-02T00:00:00Z', 25, 5, '{}')"
    )
    second = conn.execute(
        "SELECT version FROM message_model ORDER BY version DESC LIMIT 1"
    ).fetchone()["version"]
    assert second == first + 1


def test_message_tokens_reference_message_model(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}')"
    )
    version = conn.execute("SELECT version FROM message_model").fetchone()["version"]
    conn.execute(
        "INSERT INTO message_tokens (model_version, token, trash_count, keep_count)"
        " VALUES (?, 'lol', 3, 1)",
        (version,),
    )
    row = conn.execute(
        "SELECT * FROM message_tokens WHERE model_version = ?", (version,)
    ).fetchone()
    assert (row["token"], row["trash_count"], row["keep_count"]) == ("lol", 3, 1)
