import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from infovore.db.annotations import (
    MATERIALISED,
    Annotation,
    AnnotationConflictError,
    activate_scorer,
    active_version,
    annotation_history,
    drop_derived,
    record_annotation,
    version_active_at,
)
from infovore.db.connection import migrate, open_database

AT = datetime(2026, 1, 1, tzinfo=UTC)
LATER = AT + timedelta(days=30)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    connection.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (7, 1, 1, 1, ?, ?, 1, 'quiet_gap', 'h7')",
        (AT.isoformat(), AT.isoformat()),
    )
    # exchanges.p_lore_model is a foreign key onto triage_model(version), so
    # materialising into the legacy column needs the model row to exist.
    for version in (4, 5):
        connection.execute(
            "INSERT INTO triage_model (version, trained_at, labels_used, holdout_size,"
            " params_json) VALUES (?, ?, 0, 0, '{}')",
            (version, (AT + timedelta(days=version)).isoformat()),
        )
    return connection


def a_score(version: int, score: float) -> Annotation:
    return Annotation(
        subject_kind="exchange",
        subject_id=7,
        scorer="p_lore",
        scorer_version=version,
        reproducibility="derived",
        score=score,
        recipe={"model_table": "triage_model", "model_version": version},
    )


def test_two_versions_of_one_scorer_coexist_for_one_subject(conn: sqlite3.Connection) -> None:
    """The entire point. `exchanges.p_lore` held one answer, so four triage
    models were trained and three left no recoverable trace (issue #176)."""
    record_annotation(conn, a_score(4, 0.9), AT)
    record_annotation(conn, a_score(5, 0.3), LATER)

    history = annotation_history(conn, "exchange", 7, "p_lore")

    assert [(row["scorer_version"], row["score"]) for row in history] == [(4, 0.9), (5, 0.3)]


def test_a_derived_annotation_is_refused_without_its_recipe(conn: sqlite3.Connection) -> None:
    """RULE #317: a derived layer that does not record what produced it cannot
    be reproduced or compared, which is the whole defect."""
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO annotations (subject_kind, subject_id, scorer, scorer_version,"
            " reproducibility, score, created_at)"
            " VALUES ('exchange', 7, 'p_lore', 4, 'derived', 0.9, ?)",
            (AT.isoformat(),),
        )


def test_a_recorded_annotation_needs_no_recipe(conn: sqlite3.Connection) -> None:
    """A human label has no recipe to record: it is not a computation."""
    record_annotation(
        conn,
        Annotation(
            subject_kind="exchange",
            subject_id=7,
            scorer="human_lore",
            scorer_version=1,
            reproducibility="recorded",
            label="lore",
        ),
        AT,
    )

    assert len(annotation_history(conn, "exchange", 7, "human_lore")) == 1


def test_an_annotation_cannot_be_updated(conn: sqlite3.Connection) -> None:
    record_annotation(conn, a_score(4, 0.9), AT)

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE annotations SET score = 0.1 WHERE scorer_version = 4")


def test_a_recorded_annotation_cannot_be_deleted(conn: sqlite3.Connection) -> None:
    """It cannot be recomputed, so deleting it destroys the only copy. This
    session lost prompt v6's 31 claims by treating a record as a computation."""
    record_annotation(
        conn,
        Annotation(
            subject_kind="exchange",
            subject_id=7,
            scorer="human_lore",
            scorer_version=1,
            reproducibility="recorded",
            label="lore",
        ),
        AT,
    )

    with pytest.raises(sqlite3.IntegrityError, match="cannot be recomputed"):
        conn.execute("DELETE FROM annotations WHERE scorer = 'human_lore'")


def test_a_derived_annotation_can_be_dropped(conn: sqlite3.Connection) -> None:
    """The negative control for the test above: derived values are a droppable
    cache, and a blanket append-only rule would have made the distinction
    decorative."""
    record_annotation(conn, a_score(4, 0.9), AT)

    assert drop_derived(conn, "p_lore", 4) == 1
    assert annotation_history(conn, "exchange", 7, "p_lore") == []


def test_rescoring_the_same_version_is_a_conflict_not_an_overwrite(
    conn: sqlite3.Connection,
) -> None:
    record_annotation(conn, a_score(4, 0.9), AT)

    with pytest.raises(AnnotationConflictError):
        record_annotation(conn, a_score(4, 0.4), LATER)


def test_a_recorded_annotation_may_repeat_for_the_same_subject(conn: sqlite3.Connection) -> None:
    """Repeat human judgments on one item are the self-consistency measurement
    (RULE #305), so the uniqueness rule must NOT apply to records."""
    judgment = Annotation(
        subject_kind="exchange",
        subject_id=7,
        scorer="human_lore",
        scorer_version=1,
        reproducibility="recorded",
        label="lore",
    )
    record_annotation(conn, judgment, AT)
    record_annotation(conn, judgment, LATER)

    assert len(annotation_history(conn, "exchange", 7, "human_lore")) == 2


def test_activation_is_a_fact_with_a_time(conn: sqlite3.Connection) -> None:
    activate_scorer(conn, "p_lore", 4, AT)
    activate_scorer(conn, "p_lore", 5, LATER)

    assert active_version(conn, "p_lore") == 5
    assert version_active_at(conn, "p_lore", AT + timedelta(days=1)) == 4
    assert version_active_at(conn, "p_lore", LATER + timedelta(days=1)) == 5


def test_an_activation_cannot_be_rewritten(conn: sqlite3.Connection) -> None:
    activate_scorer(conn, "p_lore", 4, AT)

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE scorer_activations SET scorer_version = 9")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM scorer_activations")


def test_a_scorer_with_no_activation_has_no_active_version(conn: sqlite3.Connection) -> None:
    assert active_version(conn, "p_lore") is None
    assert version_active_at(conn, "p_lore", AT) is None


def test_the_active_version_materialises_into_the_legacy_column(
    conn: sqlite3.Connection,
) -> None:
    """The columns stay as a materialised current view so no existing reader
    breaks (issue #176)."""
    activate_scorer(conn, "p_lore", 4, AT)
    record_annotation(conn, a_score(4, 0.75), AT)

    row = conn.execute("SELECT p_lore, p_lore_model FROM exchanges WHERE id = 7").fetchone()

    assert row["p_lore"] == 0.75
    assert row["p_lore_model"] == 4


def test_a_non_active_version_does_not_touch_the_legacy_column(
    conn: sqlite3.Connection,
) -> None:
    """A shadow run must be able to score the whole corpus without changing
    what the live gate reads. That is what makes two scorers comparable
    without refitting either."""
    activate_scorer(conn, "p_lore", 4, AT)
    record_annotation(conn, a_score(4, 0.75), AT)
    record_annotation(conn, a_score(5, 0.11), LATER)

    row = conn.execute("SELECT p_lore, p_lore_model FROM exchanges WHERE id = 7").fetchone()

    assert row["p_lore"] == 0.75
    assert row["p_lore_model"] == 4


def test_the_current_view_follows_activation(conn: sqlite3.Connection) -> None:
    activate_scorer(conn, "p_lore", 4, AT)
    record_annotation(conn, a_score(4, 0.75), AT)
    record_annotation(conn, a_score(5, 0.11), LATER)

    assert [row["score"] for row in conn.execute("SELECT score FROM current_annotations")] == [0.75]

    activate_scorer(conn, "p_lore", 5, LATER)

    assert [row["score"] for row in conn.execute("SELECT score FROM current_annotations")] == [0.11]


def test_every_materialised_scorer_names_real_columns(conn: sqlite3.Connection) -> None:
    """The seam: this mapping duplicates column names that live in the schema,
    and nothing else would notice a rename."""
    for scorer, (table, score_column, version_column) in MATERIALISED.items():
        columns = {
            row["name"] for row in conn.execute(f"SELECT name FROM pragma_table_info('{table}')")
        }
        assert score_column in columns, f"{scorer}: {table}.{score_column}"
        assert version_column in columns, f"{scorer}: {table}.{version_column}"


def test_scores_stay_continuous_with_no_band_column(conn: sqlite3.Connection) -> None:
    """A low/med/high column discards ordering within each band and re-invites
    the AUC-shaped error that already cost a day: the gate scored 0.917 AUC
    while delivering no economic advantage, because ranking is what pays."""
    columns = {
        row["name"] for row in conn.execute("SELECT name FROM pragma_table_info('annotations')")
    }

    assert "score" in columns
    assert not {"band", "rating", "tier"} & columns


def test_an_annotation_must_carry_a_score_or_a_label(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO annotations (subject_kind, subject_id, scorer, scorer_version,"
            " reproducibility, recipe_json, created_at)"
            " VALUES ('exchange', 7, 'p_lore', 4, 'derived', '{}', ?)",
            (AT.isoformat(),),
        )


def test_an_unknown_subject_kind_is_refused(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO annotations (subject_kind, subject_id, scorer, scorer_version,"
            " reproducibility, score, recipe_json, created_at)"
            " VALUES ('banana', 7, 'p_lore', 4, 'derived', 0.5, '{}', ?)",
            (AT.isoformat(),),
        )


def _at_migration_21(tmp_path: Path) -> sqlite3.Connection:
    from infovore.db.connection import load_migrations

    connection = open_database(tmp_path / "b.db")
    migrate(connection, [m for m in load_migrations() if m.version <= 21])
    return connection


def _apply_22(connection: sqlite3.Connection) -> None:
    from infovore.db.connection import load_migrations

    migrate(connection, [m for m in load_migrations() if m.version == 22])


def test_the_backfill_preserves_the_scores_already_in_the_columns(tmp_path: Path) -> None:
    """203,634 exchange scores and 1,381,131 message scores exist only in the
    columns. A migration that installed the table without carrying them over
    would leave the history starting from empty."""
    conn = _at_migration_21(tmp_path)
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    conn.execute(
        "INSERT INTO triage_model (version, trained_at, labels_used, holdout_size, params_json)"
        " VALUES (4, ?, 600, 0, '{\"w\": 1}')",
        (AT.isoformat(),),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash, p_lore, p_lore_model)"
        " VALUES (7, 1, 1, 1, ?, ?, 1, 'quiet_gap', 'h7', 0.9994, 4)",
        (AT.isoformat(), AT.isoformat()),
    )

    _apply_22(conn)

    history = annotation_history(conn, "exchange", 7, "p_lore")
    assert len(history) == 1
    assert history[0]["score"] == 0.9994
    assert history[0]["scorer_version"] == 4
    assert history[0]["reproducibility"] == "derived"
    assert history[0]["source_ref"] == "migration:0022"


def test_the_backfilled_recipe_points_at_the_surviving_params(tmp_path: Path) -> None:
    """The scores for triage models 1-3 are gone, but their `params_json`
    survives in `triage_model`, which is what makes them recomputable rather
    than lost. The recipe has to say where to look."""
    import json

    conn = _at_migration_21(tmp_path)
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    conn.execute(
        "INSERT INTO triage_model (version, trained_at, labels_used, holdout_size, params_json)"
        " VALUES (4, ?, 600, 0, '{}')",
        (AT.isoformat(),),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash, p_lore, p_lore_model)"
        " VALUES (7, 1, 1, 1, ?, ?, 1, 'quiet_gap', 'h7', 0.5, 4)",
        (AT.isoformat(), AT.isoformat()),
    )

    _apply_22(conn)

    recipe = json.loads(annotation_history(conn, "exchange", 7, "p_lore")[0]["recipe_json"])
    assert recipe["model_table"] == "triage_model"
    assert recipe["model_version"] == 4
    assert recipe["params_in"] == "triage_model.params_json"
    assert recipe["value_predates_row"] is True


def test_the_backfill_activates_the_version_the_scores_came_from(tmp_path: Path) -> None:
    conn = _at_migration_21(tmp_path)
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    conn.execute(
        "INSERT INTO triage_model (version, trained_at, labels_used, holdout_size, params_json)"
        " VALUES (4, ?, 600, 0, '{}')",
        (AT.isoformat(),),
    )
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash, p_lore, p_lore_model)"
        " VALUES (7, 1, 1, 1, ?, ?, 1, 'quiet_gap', 'h7', 0.5, 4)",
        (AT.isoformat(), AT.isoformat()),
    )

    _apply_22(conn)

    assert active_version(conn, "p_lore") == 4


def test_an_unscored_row_is_not_invented(tmp_path: Path) -> None:
    """342 messages have no p_trash. A backfill that wrote a row for them
    would be asserting a score nothing produced."""
    conn = _at_migration_21(tmp_path)
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'general', 'text')"
    )
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (3, 1, 1, 1, 'a', ?, 'hi', ?, '{}')",
        (AT.isoformat(), AT.isoformat()),
    )

    _apply_22(conn)

    assert annotation_history(conn, "message", 3, "p_trash") == []
    assert conn.execute("SELECT COUNT(*) AS n FROM annotations").fetchone()["n"] == 0
