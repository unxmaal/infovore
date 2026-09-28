from pathlib import Path

from infovore.db.connection import load_migrations, migrate, open_database


def test_message_combiner_gets_a_feature_set_version_column_defaulting_to_1(
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
    row = conn.execute("SELECT feature_set_version FROM message_combiner").fetchone()
    assert row["feature_set_version"] == 1


def test_message_combiner_accepts_an_explicit_feature_set_version(tmp_path: Path) -> None:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO message_model (trained_at, labels_used, holdout_size, params_json, kind)"
        " VALUES ('2026-01-01T00:00:00Z', 20, 4, '{}', 'citation')"
    )
    citation_version = conn.execute("SELECT version FROM message_model").fetchone()["version"]
    conn.execute(
        "INSERT INTO message_combiner (trained_at, citation_model_version, human_model_version,"
        " fallback, params_json, feature_set_version)"
        " VALUES ('2026-01-01T00:00:00Z', ?, NULL, 1, '{}', 2)",
        (citation_version,),
    )
    row = conn.execute("SELECT feature_set_version FROM message_combiner").fetchone()
    assert row["feature_set_version"] == 2


def test_migration_0013_applies_cleanly_to_a_populated_message_combiner_table(
    tmp_path: Path,
) -> None:
    """Additive migration: existing rows written before #141 (no
    `feature_set_version`) must still be readable after migrating forward,
    defaulting to feature set version 1 (no context features) so a
    pre-existing stored ensemble is treated as needing a retrain."""
    conn = open_database(tmp_path / "x.db")
    all_migrations = load_migrations()
    pre_0013 = [m for m in all_migrations if m.version < 13]
    migrate(conn, pre_0013)
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

    migrate(conn)  # apply everything, including 0013

    row = conn.execute("SELECT feature_set_version FROM message_combiner").fetchone()
    assert row["feature_set_version"] == 1
