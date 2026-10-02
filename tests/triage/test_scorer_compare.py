import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infovore.db.annotations import activate_scorer
from infovore.db.connection import migrate, open_database
from infovore.triage.bayes import Model
from infovore.triage.gain_curve import Candidate
from infovore.triage.rules import DEFAULT_RULES, TriageRules
from infovore.triage.scorer_compare import (
    RulesVersionMismatchError,
    UnknownModelVersionError,
    bootstrap_difference,
    candidates_for,
    compare_scorers,
    format_comparison,
    load_model,
    scores_from_annotations,
    shadow_score,
)

AT = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "x.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'hardware', 'text')"
    )
    for version, rules_version in ((3, DEFAULT_RULES.version), (4, None)):
        params = {"lore_documents": 10, "noise_documents": 20}
        if rules_version is not None:
            params["rules_version"] = rules_version  # type: ignore[assignment]
        import json

        connection.execute(
            "INSERT INTO triage_model (version, trained_at, labels_used, holdout_size,"
            " params_json) VALUES (?, ?, 30, 0, ?)",
            (version, AT.isoformat(), json.dumps(params)),
        )
        connection.execute(
            "INSERT INTO triage_tokens (model_version, token, lore_count, noise_count)"
            " VALUES (?, 'octane', ?, ?)",
            (version, 9 if version == 4 else 1, 1 if version == 4 else 9),
        )
    for exchange_id in (1, 2):
        connection.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 1, 1, 'a', ?, ?, ?, '{}')",
            (
                exchange_id,
                AT.isoformat(),
                "octane" if exchange_id == 1 else "lunch",
                AT.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
            " started_at, ended_at, message_count, grouping_rule, content_hash)"
            " VALUES (?, 1, ?, ?, ?, ?, 1, 'quiet_gap', ?)",
            (
                exchange_id,
                exchange_id,
                exchange_id,
                AT.isoformat(),
                AT.isoformat(),
                f"h{exchange_id}",
            ),
        )
        connection.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
            (exchange_id, exchange_id),
        )
    return connection


def test_a_model_loads_by_version_not_just_the_latest(conn: sqlite3.Connection) -> None:
    """Comparing versions is impossible if only the newest can be rebuilt."""
    model, rules_version = load_model(conn, 3)

    assert isinstance(model, Model)
    assert model.lore_documents == 10
    assert rules_version == DEFAULT_RULES.version


def test_an_unknown_version_is_refused(conn: sqlite3.Connection) -> None:
    with pytest.raises(UnknownModelVersionError):
        load_model(conn, 99)


def test_scoring_with_the_wrong_rules_is_refused(conn: sqlite3.Connection) -> None:
    """`rules_version` is a content hash with no registry to resolve it, so
    the wrong rules silently change the SIG_* features and produce numbers
    that look like the model's and are not."""
    other = TriageRules(**{**DEFAULT_RULES.__dict__, "version": "r-deadbeef"})

    with pytest.raises(RulesVersionMismatchError, match="r-deadbeef"):
        shadow_score(conn, 3, other, [1, 2], AT)


def test_a_model_recording_no_rules_version_is_scored_anyway(conn: sqlite3.Connection) -> None:
    """Models 1 and 2 predate rules versioning. Refusing them would make the
    oldest scorers permanently uncomparable for a reason that is not a
    mismatch."""
    assert shadow_score(conn, 4, DEFAULT_RULES, [1, 2], AT) == 2


def test_a_shadow_score_does_not_touch_the_live_column(conn: sqlite3.Connection) -> None:
    """The property the whole table was built for: a scorer can be measured
    over the corpus without moving the gate that production reads."""
    conn.execute("UPDATE exchanges SET p_lore = 0.5, p_lore_model = 4")
    activate_scorer(conn, "p_lore", 4, AT)

    shadow_score(conn, 3, DEFAULT_RULES, [1, 2], AT)

    rows = conn.execute("SELECT p_lore, p_lore_model FROM exchanges ORDER BY id").fetchall()
    assert [row["p_lore"] for row in rows] == [0.5, 0.5]
    assert [row["p_lore_model"] for row in rows] == [4, 4]


def test_a_shadow_score_is_readable_back_per_version(conn: sqlite3.Connection) -> None:
    shadow_score(conn, 3, DEFAULT_RULES, [1, 2], AT)
    shadow_score(conn, 4, DEFAULT_RULES, [1, 2], AT)

    v3 = scores_from_annotations(conn, "p_lore", 3)
    v4 = scores_from_annotations(conn, "p_lore", 4)

    assert set(v3) == set(v4) == {1, 2}
    # The two models were fitted with opposite evidence for 'octane', so they
    # must disagree about exchange 1 or the fixture proves nothing.
    assert v3[1] != v4[1]


def test_the_shadow_recipe_names_every_input(conn: sqlite3.Connection) -> None:
    """A recipe naming only `params_json` is incomplete: the priors live
    there and the per-token counts live in `triage_tokens`."""
    import json

    shadow_score(conn, 3, DEFAULT_RULES, [1], AT)
    row = conn.execute("SELECT recipe_json FROM annotations WHERE scorer_version = 3").fetchone()
    recipe = json.loads(row["recipe_json"])

    assert recipe["priors_in"] == "triage_model.params_json"
    assert recipe["token_counts_in"] == "triage_tokens"
    assert recipe["rules_version"] == DEFAULT_RULES.version


def _outcome(conn: sqlite3.Connection, exchange_id: int, claims: int, tokens: int) -> None:
    # extraction_runs.prompt_version is a foreign key onto prompt_versions.
    conn.execute(
        "INSERT OR IGNORE INTO prompt_versions (version, text_sha256, created_at)"
        " VALUES ('v5', 'sha', ?)",
        (AT.isoformat(),),
    )
    cursor = conn.execute(
        "INSERT INTO extraction_runs (exchange_id, mode, outcome, model, prompt_version,"
        " started_at, finished_at, input_tokens, output_tokens)"
        " VALUES (?, 'trial', 'ok', 'm', 'v5', ?, ?, ?, 0)",
        (exchange_id, AT.isoformat(), AT.isoformat(), tokens),
    )
    for _ in range(claims):
        conn.execute(
            "INSERT INTO claims (exchange_id, extraction_run_id, statement, subject, kind,"
            " confidence, probe_question, novelty, permalink)"
            " VALUES (?, ?, 's', 'x', 'fact', 0.9, 'q', 'unprobed', 'https://x/1')",
            (exchange_id, cursor.lastrowid),
        )


def test_two_versions_are_ranked_over_one_population(conn: sqlite3.Connection) -> None:
    _outcome(conn, 1, claims=5, tokens=1000)
    _outcome(conn, 2, claims=0, tokens=1000)
    shadow_score(conn, 3, DEFAULT_RULES, [1, 2], AT)
    shadow_score(conn, 4, DEFAULT_RULES, [1, 2], AT)

    comparison = compare_scorers(conn, "trial", [3, 4])
    labels = [point.label for point in comparison.report.points]

    assert "p_lore v3" in labels
    assert "p_lore v4" in labels
    assert "oracle (true claims/token)" in labels
    assert "random order (control)" in labels


def test_the_better_ranker_scores_higher_on_the_curve(conn: sqlite3.Connection) -> None:
    """The fixture is rigged: v4 treats 'octane' as lore and v3 treats it as
    noise, and exchange 1 (the one containing 'octane') is the one that yields
    claims. v4 must therefore win, or the comparison is not measuring the
    ranking at all.

    The token sizes matter: a 25% budget over 2000 tokens is 500, which admits
    exactly one 500-token exchange. Sized equally, neither arm could fit any
    candidate and both would score zero regardless of their ranking."""
    _outcome(conn, 1, claims=5, tokens=500)
    _outcome(conn, 2, claims=0, tokens=1500)
    shadow_score(conn, 3, DEFAULT_RULES, [1, 2], AT)
    shadow_score(conn, 4, DEFAULT_RULES, [1, 2], AT)

    points = {p.label: p for p in compare_scorers(conn, "trial", [3, 4]).report.points}

    assert points["p_lore v4"].claims_at_25_percent > points["p_lore v3"].claims_at_25_percent


def test_an_arm_that_scored_nothing_is_reported_not_hidden(conn: sqlite3.Connection) -> None:
    """An arm ranked over a population it only partly scored looks worse for a
    reason that is not about the scorer, so the gap has to be printed."""
    _outcome(conn, 1, claims=5, tokens=1000)
    _outcome(conn, 2, claims=0, tokens=1000)
    shadow_score(conn, 3, DEFAULT_RULES, [1], AT)

    comparison = compare_scorers(conn, "trial", [3])

    assert comparison.unscored[3] == 1
    assert any("1 of the population unscored" in line for line in format_comparison(comparison))


def test_implausible_token_counts_are_still_excluded(conn: sqlite3.Connection) -> None:
    """RULE #315: a run billed less input than the system prompt occupies is
    free, so it sorts to the front of any claims-per-token ordering."""
    _outcome(conn, 1, claims=5, tokens=1000)
    _outcome(conn, 2, claims=1, tokens=2)
    shadow_score(conn, 3, DEFAULT_RULES, [1, 2], AT)

    comparison = compare_scorers(conn, "trial", [3])

    assert comparison.report.excluded_implausible == 1
    assert comparison.report.exchanges == 1


def test_candidates_exclude_implausible_token_counts(conn: sqlite3.Connection) -> None:
    _outcome(conn, 1, claims=5, tokens=1000)
    _outcome(conn, 2, claims=1, tokens=2)

    assert [c.exchange_id for c in candidates_for(conn, "trial")] == [1]


def test_a_candidate_with_no_live_score_is_kept(conn: sqlite3.Connection) -> None:
    """An exchange the live column never scored still has an outcome, so it
    belongs in the population every arm is ranked over."""
    _outcome(conn, 1, claims=5, tokens=1000)

    assert candidates_for(conn, "trial")[0].p_lore == 0.0


def _ranked(n: int) -> list[Candidate]:
    # Candidate i yields claims only if i is even; all cost the same.
    return [
        Candidate(exchange_id=i, tokens=100, claims=5 if i % 2 == 0 else 0, p_lore=0.0)
        for i in range(n)
    ]


def test_the_bootstrap_separates_a_perfect_ranker_from_a_useless_one() -> None:
    """The positive control: if a perfect ordering and a constant ordering are
    not separable, the interval is not measuring anything."""
    candidates = _ranked(200)
    arms = {
        "perfect": {c.exchange_id: float(c.claims) for c in candidates},
        "constant": {c.exchange_id: 1.0 for c in candidates},
    }

    intervals = {
        iv.label: iv
        for iv in bootstrap_difference(
            candidates, arms, baseline="constant", fraction=0.25, resamples=300
        )
    }

    assert intervals["perfect"].point > 0
    assert intervals["perfect"].separable_from_zero


def test_the_bootstrap_does_not_separate_an_arm_from_itself() -> None:
    """The negative control, and the one that matters: a method that reports a
    difference between identical arms would call anything separable."""
    candidates = _ranked(200)
    scores = {c.exchange_id: float(c.claims) for c in candidates}
    arms = {"a": dict(scores), "b": dict(scores)}

    intervals = {
        iv.label: iv
        for iv in bootstrap_difference(
            candidates, arms, baseline="a", fraction=0.25, resamples=300, include_random=False
        )
    }

    assert intervals["b"].point == pytest.approx(0.0)
    assert not intervals["b"].separable_from_zero


def test_the_random_control_is_redrawn_each_resample() -> None:
    """One fixed shuffle is a single draw from the distribution of random
    orderings; pinning it makes the verdict depend on which shuffle was
    seeded (RULE #217). Re-drawing widens the interval, which is the honest
    width."""
    candidates = _ranked(120)
    arms = {"perfect": {c.exchange_id: float(c.claims) for c in candidates}}

    intervals = bootstrap_difference(
        candidates, arms, baseline="perfect", fraction=0.25, resamples=300
    )

    assert any(iv.label.startswith("random") for iv in intervals)


def test_a_missing_baseline_yields_no_intervals() -> None:
    assert bootstrap_difference(_ranked(10), {"a": {}}, baseline="absent") == []


def test_no_candidates_yields_no_intervals() -> None:
    assert bootstrap_difference([], {"a": {}}, baseline="a") == []


def test_recovery_is_zero_when_nothing_yielded() -> None:
    barren = [Candidate(exchange_id=i, tokens=100, claims=0, p_lore=0.0) for i in range(5)]
    arms = {"a": {c.exchange_id: 1.0 for c in barren}, "b": {c.exchange_id: 2.0 for c in barren}}

    intervals = bootstrap_difference(barren, arms, baseline="a", resamples=50)

    assert all(iv.point == 0.0 for iv in intervals)


def _cli(argv: list[str], tmp_path: Path) -> tuple[int, str]:
    import io

    from infovore.cli import main

    out, err = io.StringIO(), io.StringIO()
    env = {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "1",
        "INFOVORE_CHANNEL_IDS": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
        "INFOVORE_JUDGE_BACKEND": "fake",
    }
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue() + err.getvalue()


def _seed_cli_db(tmp_path: Path) -> None:
    import json

    _cli(["status"], tmp_path)
    conn = open_database(tmp_path / "infovore.db")
    conn.execute(
        "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
        " VALUES (1, 1, NULL, 'hardware', 'text')"
    )
    conn.execute(
        "INSERT INTO triage_model (version, trained_at, labels_used, holdout_size, params_json)"
        " VALUES (3, ?, 30, 0, ?)",
        (AT.isoformat(), json.dumps({"lore_documents": 10, "noise_documents": 20})),
    )
    conn.execute(
        "INSERT INTO triage_tokens (model_version, token, lore_count, noise_count)"
        " VALUES (3, 'octane', 9, 1)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO prompt_versions (version, text_sha256, created_at)"
        " VALUES ('v5', 'sha', ?)",
        (AT.isoformat(),),
    )
    for exchange_id in (1, 2):
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, 1, 1, 1, 'a', ?, ?, ?, '{}')",
            (exchange_id, AT.isoformat(), "octane", AT.isoformat()),
        )
        conn.execute(
            "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
            " started_at, ended_at, message_count, grouping_rule, content_hash)"
            " VALUES (?, ?, ?, ?, ?, 1, 'quiet_gap', ?)".replace(
                "VALUES (?, ?, ?, ?, ?, 1,", "VALUES (?, 1, ?, ?, ?, ?, 1,"
            ),
            (
                exchange_id,
                exchange_id,
                exchange_id,
                AT.isoformat(),
                AT.isoformat(),
                f"h{exchange_id}",
            ),
        )
        conn.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, 1)",
            (exchange_id, exchange_id),
        )
        cursor = conn.execute(
            "INSERT INTO extraction_runs (exchange_id, mode, outcome, model, prompt_version,"
            " started_at, finished_at, input_tokens, output_tokens)"
            " VALUES (?, 'trial', 'ok', 'm', 'v5', ?, ?, 1000, 0)",
            (exchange_id, AT.isoformat(), AT.isoformat()),
        )
        if exchange_id == 1:
            conn.execute(
                "INSERT INTO claims (exchange_id, extraction_run_id, statement, subject, kind,"
                " confidence, probe_question, novelty, permalink)"
                " VALUES (?, ?, 's', 'x', 'fact', 0.9, 'q', 'unprobed', 'https://x/1')",
                (exchange_id, cursor.lastrowid),
            )
    conn.commit()
    conn.close()


def test_the_cli_reports_the_comparison_and_its_intervals(tmp_path: Path) -> None:
    """The durable artifact: a command anyone can re-run, not a number in a
    chat message that is stale the moment anything changes."""
    _seed_cli_db(tmp_path)
    conn = open_database(tmp_path / "infovore.db")
    shadow_score(conn, 3, DEFAULT_RULES, [1, 2], AT)
    conn.commit()
    conn.close()

    code, out = _cli(["triage", "--compare-scorers", "3", "--compare-mode", "trial"], tmp_path)

    assert code == 0
    assert "p_lore v3" in out
    assert "oracle" in out
    assert "paired against p_lore v3" in out
    assert "95% CI over 2000 resamples" in out


def test_the_cli_refuses_a_version_list_that_is_not_versions(tmp_path: Path) -> None:
    _seed_cli_db(tmp_path)

    code, out = _cli(["triage", "--compare-scorers", "v3,banana"], tmp_path)

    assert code != 0
    assert "wants versions" in out


def test_the_cli_refuses_an_empty_version_list(tmp_path: Path) -> None:
    _seed_cli_db(tmp_path)

    code, out = _cli(["triage", "--compare-scorers", ","], tmp_path)

    assert code != 0
    assert "at least one version" in out


def test_at_the_full_budget_every_ordering_recovers_everything() -> None:
    """The loop only ever exits by exhausting the budget in the other tests.
    At 100% the whole population fits, so ordering cannot matter and every
    arm must tie the baseline at zero difference."""
    candidates = _ranked(40)
    arms = {
        "perfect": {c.exchange_id: float(c.claims) for c in candidates},
        "inverted": {c.exchange_id: -float(c.claims) for c in candidates},
    }

    intervals = bootstrap_difference(
        candidates, arms, baseline="perfect", fraction=1.0, resamples=50, include_random=False
    )

    assert [iv.point for iv in intervals] == [0.0]


def test_the_live_order_breaks_p_lore_ties_by_triage_score_then_time() -> None:
    """Ranking by p_lore alone is NOT the live gate and understates it. The
    real key is `p_lore IS NULL, p_lore DESC, triage_score DESC, started_at,
    id`, so saturated ties are already broken (PR #186 measured p_lore alone
    and drew the wrong conclusion from it)."""
    import tempfile

    from infovore.db.connection import migrate, open_database
    from infovore.triage.scorer_compare import live_order_scores

    with tempfile.TemporaryDirectory() as directory:
        conn = open_database(Path(directory) / "x.db")
        migrate(conn)
        conn.execute(
            "INSERT INTO channels (id, guild_id, parent_id, name, kind)"
            " VALUES (1, 1, NULL, 'c', 'text')"
        )
        # Three exchanges tied on p_lore: the saturated case.
        for exchange_id, triage_score, day in ((1, 0.1, "03"), (2, 0.9, "02"), (3, 0.1, "01")):
            conn.execute(
                "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
                " started_at, ended_at, message_count, grouping_rule, content_hash, p_lore,"
                " triage_score) VALUES (?, 1, 1, 1, ?, ?, 1, 'quiet_gap', ?, 1.0, ?)",
                (
                    exchange_id,
                    f"2026-01-{day}T00:00:00+00:00",
                    f"2026-01-{day}T00:00:00+00:00",
                    f"h{exchange_id}",
                    triage_score,
                ),
            )

        scores = live_order_scores(conn, [1, 2, 3])

    # 2 first on triage_score; then 3 before 1 because it is older.
    assert sorted(scores, key=lambda i: -scores[i]) == [2, 3, 1]


def test_an_empty_population_has_no_live_order() -> None:
    import tempfile

    from infovore.db.connection import migrate, open_database
    from infovore.triage.scorer_compare import live_order_scores

    with tempfile.TemporaryDirectory() as directory:
        conn = open_database(Path(directory) / "x.db")
        migrate(conn)
        assert live_order_scores(conn, []) == {}


def test_the_comparison_always_includes_the_live_order_arm(conn: sqlite3.Connection) -> None:
    """Every comparison must show what production actually does, or an arm
    gets judged against a gate that is not the one running."""
    from infovore.triage.scorer_compare import LIVE_ORDER

    _outcome(conn, 1, claims=5, tokens=1000)
    _outcome(conn, 2, claims=0, tokens=1000)
    shadow_score(conn, 3, DEFAULT_RULES, [1, 2], AT)

    labels = [p.label for p in compare_scorers(conn, "trial", [3]).report.points]

    assert LIVE_ORDER in labels
