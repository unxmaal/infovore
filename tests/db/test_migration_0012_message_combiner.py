from pathlib import Path

from infovore.db.connection import migrate, open_database


def test_message_model_gets_a_kind_column_defaulting_to_legacy(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}')"
    )
    row = conn.execute("SELECT kind FROM message_model").fetchone()
    assert row["kind"] == "legacy"


def test_message_model_accepts_citation_and_human_kinds(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json, kind)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}', 'citation')"
    )
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json, kind)"
        " VALUES ('2026-01-01T00:00:00Z', 5, 0, '{}', 'human')"
    )
    kinds = {row["kind"] for row in conn.execute("SELECT kind FROM message_model")}
    assert kinds == {"citation", "human"}


def test_message_combiner_table_exists_with_expected_columns(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    names = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "message_combiner" in names

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(message_combiner)")}
    assert columns == {
        "version",
        "trained_at",
        "citation_model_version",
        "human_model_version",
        "fallback",
        "params_json",
        # `feature_set_version` (default 1) is added by migration 0013
        # (issue #141), additive to this one.
        "feature_set_version",
    }


def test_message_combiner_version_autoincrements(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json, kind)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}', 'citation')"
    )
    citation_version = conn.execute("SELECT version FROM message_model").fetchone()["version"]
    conn.execute(
        "INSERT INTO message_combiner (trained_at, citation_model_version, human_model_version,"
        " fallback, params_json)"
        " VALUES ('2026-01-01T00:00:00Z', ?, NULL, 1, '{}')",
        (citation_version,),
    )
    first = conn.execute("SELECT version FROM message_combiner").fetchone()["version"]
    conn.execute(
        "INSERT INTO message_combiner (trained_at, citation_model_version, human_model_version,"
        " fallback, params_json)"
        " VALUES ('2026-01-02T00:00:00Z', ?, NULL, 1, '{}')",
        (citation_version,),
    )
    second = conn.execute(
        "SELECT version FROM message_combiner ORDER BY version DESC LIMIT 1"
    ).fetchone()["version"]
    assert second == first + 1


def test_message_combiner_human_model_version_is_nullable_for_fallback(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json, kind)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}', 'citation')"
    )
    citation_version = conn.execute("SELECT version FROM message_model").fetchone()["version"]
    conn.execute(
        "INSERT INTO message_combiner (trained_at, citation_model_version, human_model_version,"
        " fallback, params_json)"
        " VALUES ('2026-01-01T00:00:00Z', ?, NULL, 1, '{}')",
        (citation_version,),
    )
    row = conn.execute("SELECT human_model_version, fallback FROM message_combiner").fetchone()
    assert row["human_model_version"] is None
    assert row["fallback"] == 1


def test_migration_0012_applies_cleanly_to_a_populated_message_model_table(
    tmp_path: Path,
) -> None:
    """Additive migration: existing rows written before #135 (no `kind`, no
    `message_combiner`) must still be readable after migrating forward."""
    from infovore.db.connection import load_migrations

    conn = open_database(tmp_path / "x.db")
    all_migrations = load_migrations()
    pre_0012 = [m for m in all_migrations if m.version < 12]
    migrate(conn, pre_0012)
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{\"human_weight\": 5}')"
    )

    migrate(conn)  # apply everything, including 0012

    row = conn.execute("SELECT labels_used, kind FROM message_model").fetchone()
    assert row["labels_used"] == 20
    assert row["kind"] == "legacy"
