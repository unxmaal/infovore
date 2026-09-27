import io
import re
import sqlite3
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from infovore.cli import ExitCode, builtin_commands, main
from infovore.db.connection import migrate, open_database
from infovore.db.exchanges import insert_exchange
from infovore.db.labels import set_label
from infovore.rows import ExchangeRow, ExtractionStatus, GroupingRule, Label, LabelSource
from infovore.triage.score import TRIAGE_VERSION

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def seed_linked_exchange(
    conn: sqlite3.Connection, message_id: int, channel_id: int, content: str, label: Label
) -> int:
    """Unlike `seed_labeled` below, this properly links the message to the
    exchange via `insert_exchange` (populating `exchange_messages`), so
    `infovore.triage.bayes.features` sees real word tokens rather than just
    the CHAN_/LEN_ virtual ones — needed for anything that inspects specific
    token content (e.g. `--suggest-terms`)."""
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " author_is_bot, created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 9, 1, 'alice', 0, ?, ?, ?, '{}')",
        (message_id, channel_id, _NOW.isoformat(), content, _NOW.isoformat()),
    )
    row = ExchangeRow(
        id=None,
        channel_id=channel_id,
        thread_id=None,
        first_message_id=message_id,
        last_message_id=message_id,
        started_at=_NOW,
        ended_at=_NOW,
        message_count=1,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash=f"linked-{message_id}",
        parent_exchange_id=None,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )
    exchange_id = insert_exchange(conn, row, [message_id])
    set_label(conn, exchange_id, label, LabelSource.HUMAN, None, _NOW)
    return exchange_id


def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "9",
        "INFOVORE_CHANNEL_IDS": "1",
        "INFOVORE_DB_PATH": str(tmp_path / "infovore.db"),
    }


def custom_rules_path(tmp_path: Path) -> Path:
    """A copy of the shipped rules.toml with one value changed, so its version differs."""
    shipped = resources.files("infovore.triage").joinpath("rules.toml").read_text(encoding="utf-8")
    changed = shipped.replace("domain_term_weight = 0.15", "domain_term_weight = 0.16", 1)
    assert changed != shipped
    path = tmp_path / "custom_rules.toml"
    path.write_text(changed)
    return path


def seed(db_path: str, content: str = "just chatting, nothing to see") -> int:
    conn = open_database(db_path)
    migrate(conn)
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (1, 1, 9, 1, 'a', '2026-01-01T00:00:00+00:00', ?,"
        " '2026-01-01T00:00:00+00:00', '{}')",
        (content,),
    )
    conn.execute(
        "INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash, extraction_status)"
        " VALUES (1, 1, 1, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 1,"
        " 'quiet_gap', 'h1', 'pending')"
    )
    exchange_id = conn.execute("SELECT id FROM exchanges").fetchone()["id"]
    conn.close()
    return int(exchange_id)


def run(argv: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, environ=env, dotenv_path=None, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_triage_is_a_builtin_command() -> None:
    assert "triage" in [command.name for command in builtin_commands()]


def test_triage_scores_pending_exchanges(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["triage"], env)

    assert code == ExitCode.OK
    assert "scored=1" in out
    conn = open_database(env["INFOVORE_DB_PATH"])
    row = conn.execute("SELECT triage_version FROM exchanges").fetchone()
    assert row["triage_version"] == TRIAGE_VERSION


def test_triage_streams_progress_lines(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["triage"], env)

    assert code == ExitCode.OK
    lines = out.splitlines()
    assert lines[0] == "triage: 1 exchanges to score"
    assert any(line.startswith("exchange 1: score=") for line in lines)


class FlushCountingIO(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes_at: list[int] = []

    def flush(self) -> None:
        self.flushes_at.append(self.getvalue().count("\n"))
        super().flush()


def test_triage_flushes_progress_lines(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out = FlushCountingIO()
    code = main(["triage"], environ=env, dotenv_path=None, stdout=out, stderr=io.StringIO())

    assert code == ExitCode.OK
    assert len(out.flushes_at) >= 3
    assert out.flushes_at == sorted(out.flushes_at)


def test_triage_report_prints_histogram_channels_thresholds_and_reasons(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"], content="Octane needs 6.5.22, check /usr/var/log")

    code, out, _ = run(["triage", "--report"], env)

    assert code == ExitCode.OK
    assert "score histogram:" in out
    assert "channels:" in out
    assert "above threshold:" in out
    assert "below threshold:" in out
    assert "top reasons:" in out


def test_triage_without_report_flag_omits_report(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["triage"], env)

    assert code == ExitCode.OK
    assert "score histogram:" not in out


def test_triage_is_idempotent_over_two_cli_runs(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    run(["triage"], env)

    code, out, _ = run(["triage"], env)

    assert code == ExitCode.OK
    assert "scored=0" in out


LORE_CONTENT = "PROM 6.5.22 part 030-1234-001 /usr/sbin/inst"
NOISE_CONTENT = "lol gg no cap"


def _insert_labeled_exchange(
    conn: sqlite3.Connection, message_id: int, channel_id: int, content: str, label: str
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 9, 1, 'a', '2026-01-01T00:00:00+00:00', ?,"
        " '2026-01-01T00:00:00+00:00', '{}')",
        (message_id, channel_id, content),
    )
    content_hash = f"h{message_id}"
    conn.execute(
        "INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash, extraction_status)"
        " VALUES (?, ?, ?, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 1,"
        " 'quiet_gap', ?, 'pending')",
        (channel_id, message_id, message_id, content_hash),
    )
    exchange_id = conn.execute(
        "SELECT id FROM exchanges WHERE content_hash = ?", (content_hash,)
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO exchange_labels (exchange_id, label, source, source_ref, labeled_at)"
        " VALUES (?, ?, 'human', NULL, '2026-01-01T00:00:00+00:00')",
        (exchange_id, label),
    )


def seed_labeled(db_path: str, lore_count: int, noise_count: int) -> None:
    conn = open_database(db_path)
    migrate(conn)
    message_id = 1
    for _ in range(lore_count):
        _insert_labeled_exchange(conn, message_id, channel_id=1, content=LORE_CONTENT, label="lore")
        message_id += 1
    for _ in range(noise_count):
        _insert_labeled_exchange(
            conn, message_id, channel_id=2, content=NOISE_CONTENT, label="noise"
        )
        message_id += 1
    conn.close()


def _insert_llm_labeled_exchange(
    conn: sqlite3.Connection,
    message_id: int,
    channel_id: int,
    content: str,
    label: str,
    sampled_by: str | None,
) -> None:
    """Like `_insert_labeled_exchange`, but the label is `llm`-sourced from a
    trial extraction run carrying a recorded sampling origin, so the
    uncertain-sampling-share warning (issue #107) can trace it."""
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, ingested_at, raw_json)"
        " VALUES (?, ?, 9, 1, 'a', '2026-01-01T00:00:00+00:00', ?,"
        " '2026-01-01T00:00:00+00:00', '{}')",
        (message_id, channel_id, content),
    )
    content_hash = f"h{message_id}"
    conn.execute(
        "INSERT INTO exchanges (channel_id, first_message_id, last_message_id, started_at,"
        " ended_at, message_count, grouping_rule, content_hash, extraction_status)"
        " VALUES (?, ?, ?, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 1,"
        " 'quiet_gap', ?, 'pending')",
        (channel_id, message_id, message_id, content_hash),
    )
    exchange_id = conn.execute(
        "SELECT id FROM exchanges WHERE content_hash = ?", (content_hash,)
    ).fetchone()["id"]
    conn.execute(
        "INSERT OR IGNORE INTO prompt_versions (version, text_sha256, created_at)"
        " VALUES ('v1', 'sha', '2026-01-01T00:00:00+00:00')"
    )
    cursor = conn.execute(
        "INSERT INTO extraction_runs (exchange_id, model, prompt_version, started_at, mode,"
        " outcome, sampled_by) VALUES (?, 'm', 'v1', '2026-01-01T00:00:00+00:00', 'trial',"
        " 'ok', ?)",
        (exchange_id, sampled_by),
    )
    run_id = cursor.lastrowid
    conn.execute(
        "INSERT INTO exchange_labels (exchange_id, label, source, source_ref, labeled_at)"
        " VALUES (?, ?, 'llm', ?, '2026-01-01T00:00:00+00:00')",
        (exchange_id, label, f"run:{run_id} model:m"),
    )


def seed_llm_labeled(db_path: str, origins: list[tuple[str, str]]) -> None:
    """`origins`: one `(label, sampled_by)` pair per labeled exchange."""
    conn = open_database(db_path)
    migrate(conn)
    for message_id, (label, sampled_by) in enumerate(origins, start=1):
        content = LORE_CONTENT if label == "lore" else NOISE_CONTENT
        channel_id = 1 if label == "lore" else 2
        _insert_llm_labeled_exchange(conn, message_id, channel_id, content, label, sampled_by)
    conn.close()


_MAJORITY_UNCERTAIN_ORIGINS: list[tuple[str, str]] = (
    [("lore", "uncertain")] * 3
    + [("lore", "random")]
    + [("noise", "uncertain")] * 3
    + [("noise", "random")]
)


def test_triage_signal_report_warns_on_uncertain_sampling_share(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_llm_labeled(env["INFOVORE_DB_PATH"], _MAJORITY_UNCERTAIN_ORIGINS)

    code, out, _ = run(["triage", "--signal-report"], env)

    assert code == ExitCode.OK
    assert "uncertain" in out
    assert "warning" in out


def test_triage_signal_report_does_not_warn_when_sampling_is_balanced(tmp_path: Path) -> None:
    env = environment(tmp_path)
    origins = [("lore", "random")] * 4 + [("noise", "random")] * 4
    seed_llm_labeled(env["INFOVORE_DB_PATH"], origins)

    code, out, _ = run(["triage", "--signal-report"], env)

    assert code == ExitCode.OK
    assert "warning" not in out


def test_triage_signal_report_warns_on_class_imbalance_when_provenance_unknown(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 18, 2)

    code, out, _ = run(["triage", "--signal-report"], env)

    assert code == ExitCode.OK
    assert "warning" in out
    assert "class imbalance" in out


def test_triage_fit_weights_warns_on_uncertain_sampling_share_not_class_imbalance(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    seed_llm_labeled(env["INFOVORE_DB_PATH"], _MAJORITY_UNCERTAIN_ORIGINS)
    out_path = tmp_path / "fitted.toml"

    code, out, _ = run(["triage", "--fit-weights", "--out", str(out_path)], env)

    assert code == ExitCode.OK
    assert "uncertain" in out
    assert "warning" in out


def test_triage_train_refuses_with_insufficient_labels(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 3, 3)

    code, _, err = run(["triage", "--train"], env)

    assert code == ExitCode.CONFIG
    assert "10" in err


def test_triage_train_prints_report_and_scores_every_exchange(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)

    code, out, _ = run(["triage", "--train"], env)

    assert code == ExitCode.OK
    assert "trained model v" in out
    assert "threshold" in out
    assert "confusion" in out
    assert "top tokens:" in out
    assert "scored 40 exchanges" in out

    conn = open_database(env["INFOVORE_DB_PATH"])
    rows = conn.execute("SELECT p_lore, p_lore_model FROM exchanges").fetchall()
    assert len(rows) == 40
    assert all(row["p_lore"] is not None for row in rows)


def test_triage_recommend_threshold_without_a_model_exits_config_error(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)

    code, _, err = run(["triage", "--recommend-threshold"], env)

    assert code == ExitCode.CONFIG
    assert "triage --train" in err


def test_triage_recommend_threshold_prints_recommendation(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    code, out, _ = run(["triage", "--recommend-threshold", "--min-recall", "0.5"], env)

    assert code == ExitCode.OK
    assert "INFOVORE_TRIAGE_MIN_P_LORE=" in out
    assert "expected share" in out


def test_triage_recommend_threshold_prints_a_paste_safe_assignment_line(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    code, out, _ = run(["triage", "--recommend-threshold", "--min-recall", "0.5"], env)

    assert code == ExitCode.OK
    assignments = [line for line in out.splitlines() if "INFOVORE_TRIAGE_MIN_P_LORE=" in line]
    assert len(assignments) == 1
    assert re.fullmatch(r"INFOVORE_TRIAGE_MIN_P_LORE=[0-9.eE+-]+", assignments[0])
    assert "recall=" in out
    assert "precision=" in out


def test_triage_recommend_threshold_with_unreachable_recall_reports_none(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    code, out, _ = run(["triage", "--recommend-threshold", "--min-recall", "1.01"], env)

    assert code == ExitCode.OK
    assert "no threshold meets recall" in out


def test_triage_recommend_threshold_uses_precision_beyond_the_fixed_grid(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    code, out, _ = run(["triage", "--recommend-threshold", "--min-recall", "1.0"], env)

    assert code == ExitCode.OK
    line = next(line for line in out.splitlines() if line.startswith("recommended"))
    value_str = line.split("=", 1)[1].split(" ", 1)[0]
    assert float(value_str) > 0.9
    assert value_str not in {f"{i / 10:.1f}" for i in range(1, 10)}


def test_triage_recommend_threshold_prints_a_recall_target_table(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    code, out, _ = run(["triage", "--recommend-threshold", "--min-recall", "0.6"], env)

    assert code == ExitCode.OK
    for target in ("0.95", "0.90", "0.80", "0.70", "0.50"):
        assert target in out


def test_render_recall_table_marks_unreachable_targets_plainly() -> None:
    from infovore.triage.bayes import Metrics
    from infovore.triage.command import _render_recall_table
    from infovore.triage.train import RecommendationRow

    rows = [
        RecommendationRow(min_recall=0.95, metric=None, share=None, formatted_threshold=None),
        RecommendationRow(
            min_recall=0.5,
            metric=Metrics(0.9, tp=2, fp=0, fn=0, tn=2),
            share=0.5,
            formatted_threshold="0.9",
        ),
    ]

    rendered = _render_recall_table(rows)

    assert "0.95  unreachable" in rendered
    assert "0.50  threshold=0.9 share=0.500" in rendered


# --- issue #95 PR A: staleness warning when the active rules changed -------


def test_triage_warns_when_the_trained_model_used_different_rules(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    stale_env = dict(env)
    stale_env["INFOVORE_TRIAGE_RULES"] = str(custom_rules_path(tmp_path))
    code, out, _ = run(["triage"], stale_env)

    assert code == ExitCode.OK
    assert "warning" in out
    assert "triage --train" in out


def test_triage_recommend_threshold_warns_when_the_trained_model_used_different_rules(
    tmp_path: Path,
) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    stale_env = dict(env)
    stale_env["INFOVORE_TRIAGE_RULES"] = str(custom_rules_path(tmp_path))
    code, out, _ = run(["triage", "--recommend-threshold"], stale_env)

    assert code == ExitCode.OK
    assert "warning" in out
    assert "triage --train" in out


def test_triage_does_not_warn_when_rules_match_the_trained_model(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage", "--train"], env)

    code, out, _ = run(["triage"], env)

    assert code == ExitCode.OK
    assert "warning" not in out


def test_triage_does_not_warn_without_a_trained_model(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["triage"], env)

    assert code == ExitCode.OK
    assert "warning" not in out


# --- issue #95 PR B: --signal-report, --suggest-terms, --fit-weights, report card


def test_triage_signal_report_without_labels_is_a_config_error(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])

    code, _, err = run(["triage", "--signal-report"], env)

    assert code == ExitCode.CONFIG
    assert "infovore label" in err


def test_triage_signal_report_prints_a_table_of_every_signal(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)

    code, out, _ = run(["triage", "--signal-report"], env)

    assert code == ExitCode.OK
    assert "fires_lore" in out
    assert "part_number" in out


def test_triage_suggest_terms_without_a_model_is_a_config_error(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)

    code, _, err = run(["triage", "--suggest-terms"], env)

    assert code == ExitCode.CONFIG
    assert "triage --train" in err


def seed_suggest_terms_fixture(db_path: str) -> None:
    conn = open_database(db_path)
    migrate(conn)
    for i in range(15):
        seed_linked_exchange(
            conn,
            i + 1,
            channel_id=1,
            content="Octane needs octane2000 diagnostics work",
            label=Label.LORE,
        )
    for i in range(15):
        seed_linked_exchange(
            conn, 100 + i, channel_id=2, content="lol gg no cap whatever", label=Label.NOISE
        )
    conn.close()


def test_triage_suggest_terms_prints_additions_drops_and_a_snippet(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_suggest_terms_fixture(env["INFOVORE_DB_PATH"])
    run(["triage", "--train"], env)

    code, out, _ = run(["triage", "--suggest-terms"], env)

    assert code == ExitCode.OK
    assert "candidate additions" in out
    assert "drop candidates" in out
    assert "domain_terms = [" in out
    assert "octane2000" in out


def test_triage_suggest_terms_accepts_min_support(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_suggest_terms_fixture(env["INFOVORE_DB_PATH"])
    run(["triage", "--train"], env)

    code, out, _ = run(["triage", "--suggest-terms", "--min-support", "1"], env)

    assert code == ExitCode.OK
    assert "candidate additions" in out


def test_triage_fit_weights_without_out_is_a_config_error(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)

    code, _, err = run(["triage", "--fit-weights"], env)

    assert code == ExitCode.CONFIG
    assert "--out" in err


def test_triage_fit_weights_without_labels_is_a_config_error(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])
    out_path = tmp_path / "fitted.toml"

    code, _, err = run(["triage", "--fit-weights", "--out", str(out_path)], env)

    assert code == ExitCode.CONFIG
    assert "infovore label" in err
    assert not out_path.exists()


def test_triage_fit_weights_writes_a_loadable_rules_file(tmp_path: Path) -> None:
    from infovore.triage.rules import load_rules

    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    out_path = tmp_path / "fitted.toml"

    code, out, _ = run(["triage", "--fit-weights", "--out", str(out_path)], env)

    assert code == ExitCode.OK
    assert out_path.exists()
    assert str(out_path) in out
    assert "before -> after" in out or "signal weights" in out
    rules = load_rules(out_path)
    assert rules.domain_terms  # loadable, non-empty term list preserved


def test_triage_fit_weights_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    out_path = tmp_path / "fitted.toml"
    out_path.write_text("preexisting")

    code, _, err = run(["triage", "--fit-weights", "--out", str(out_path)], env)

    assert code == ExitCode.CONFIG
    assert "--force" in err
    assert out_path.read_text() == "preexisting"


def test_triage_fit_weights_overwrites_with_force(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    out_path = tmp_path / "fitted.toml"
    out_path.write_text("preexisting")

    code, _, _ = run(["triage", "--fit-weights", "--out", str(out_path), "--force"], env)

    assert code == ExitCode.OK
    assert out_path.read_text() != "preexisting"


def test_triage_fit_weights_refuses_to_overwrite_the_shipped_rules_file(tmp_path: Path) -> None:
    from importlib import resources

    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    shipped_path = resources.files("infovore.triage").joinpath("rules.toml")

    code, _, err = run(["triage", "--fit-weights", "--out", str(shipped_path), "--force"], env)

    assert code == ExitCode.CONFIG
    assert "shipped rules.toml" in err


def test_triage_fit_weights_warns_on_class_imbalance(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 18, 2)
    out_path = tmp_path / "fitted.toml"

    code, out, _ = run(["triage", "--fit-weights", "--out", str(out_path)], env)

    assert code == ExitCode.OK
    assert "warning" in out
    assert "class imbalance" in out


def test_triage_report_includes_a_report_card_when_labels_exist(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed_labeled(env["INFOVORE_DB_PATH"], 20, 20)
    run(["triage"], env)

    code, out, _ = run(["triage", "--report"], env)

    assert code == ExitCode.OK
    assert "report card" in out
    assert "rule score" in out


def test_render_score_card_marks_unreachable_recall_plainly() -> None:
    from infovore.triage.command import _render_score_card
    from infovore.triage.tuning import ScoreCard

    card = ScoreCard(
        label="p_lore", auc=0.5, threshold=None, recall_at_threshold=None, corpus_share=None
    )

    rendered = _render_score_card(card)

    assert "unreachable" in rendered


def test_render_report_card_includes_p_lore_and_fitted_combination_when_present() -> None:
    from infovore.triage.command import _render_report_card
    from infovore.triage.tuning import ReportCard, ScoreCard

    rule = ScoreCard("rule score", 0.9, 0.5, 0.9, 0.5)
    p_lore = ScoreCard("p_lore", 0.95, 0.6, 0.9, 0.4)
    combo = ScoreCard("fitted combination", 0.97, 0.7, 0.9, 0.3)
    card = ReportCard(rule_score=rule, p_lore=p_lore, fitted_combination=combo)

    rendered = _render_report_card(card, 0.8)

    assert "rule score" in rendered
    assert "p_lore" in rendered
    assert "fitted combination" in rendered


def test_triage_report_omits_the_report_card_without_labels(tmp_path: Path) -> None:
    env = environment(tmp_path)
    seed(env["INFOVORE_DB_PATH"])

    code, out, _ = run(["triage", "--report"], env)

    assert code == ExitCode.OK
    assert "report card" not in out
