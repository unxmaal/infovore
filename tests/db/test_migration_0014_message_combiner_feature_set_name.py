from pathlib import Path

from infovore.db.connection import load_migrations, migrate, open_database


def test_message_combiner_gets_a_feature_set_name_column_defaulting_to_null(
    tmp_path: Path,
) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json, kind)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}', 'citation')"
    )
    citation_version = conn.execute("SELECT version FROM message_model").fetchone()["version"]
    conn.execute(
        "INSERT INTO message_combiner (trained_at, citation_model_version, human_model_version,"
        " fallback, params_json) VALUES ('2026-01-01T00:00:00Z', ?, NULL, 1, '{}')",
        (citation_version,),
    )
    row = conn.execute("SELECT feature_set_name FROM message_combiner").fetchone()
    assert row["feature_set_name"] is None


def test_message_combiner_accepts_an_explicit_feature_set_name(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json, kind)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}', 'citation')"
    )
    citation_version = conn.execute("SELECT version FROM message_model").fetchone()["version"]
    conn.execute(
        "INSERT INTO message_combiner (trained_at, citation_model_version, human_model_version,"
        " fallback, params_json, feature_set_name)"
        " VALUES ('2026-01-01T00:00:00Z', ?, NULL, 1, '{}', 'structural')",
        (citation_version,),
    )
    row = conn.execute("SELECT feature_set_name FROM message_combiner").fetchone()
    assert row["feature_set_name"] == "structural"


def test_migration_0014_applies_cleanly_to_a_populated_message_combiner_table(
    tmp_path: Path,
) -> None:
    """Additive migration: existing rows written before #144 (no
    `feature_set_name`) must still be readable after migrating forward,
    defaulting to `NULL` so a pre-existing stored ensemble falls back to
    resolving its named feature set from `feature_set_version` alone (see
    `infovore.sift.train.load_latest_message_model`)."""
    conn = open_database(tmp_path / "x.db")
    all_migrations = load_migrations()
    pre_0014 = [m for m in all_migrations if m.version < 14]
    migrate(conn, pre_0014)
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json, kind)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}', 'citation')"
    )
    citation_version = conn.execute("SELECT version FROM message_model").fetchone()["version"]
    conn.execute(
        "INSERT INTO message_combiner (trained_at, citation_model_version, human_model_version,"
        " fallback, params_json) VALUES ('2026-01-01T00:00:00Z', ?, NULL, 1, '{}')",
        (citation_version,),
    )

    migrate(conn)  # apply everything, including 0014

    row = conn.execute("SELECT feature_set_name FROM message_combiner").fetchone()
    assert row["feature_set_name"] is None
