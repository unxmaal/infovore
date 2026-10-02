import json
import math
import random
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from infovore.db.annotations import Annotation, record_annotation
from infovore.db.batch import exchange_inputs_for_ids
from infovore.triage.bayes import Model, features, p_lore
from infovore.triage.gain_curve import (
    MIN_PLAUSIBLE_INPUT_TOKENS,
    RANDOM,
    Candidate,
    GainReport,
    gain_points,
)
from infovore.triage.rules import TriageRules

SCORE_BATCH = 500


class UnknownModelVersionError(ValueError):
    pass


class RulesVersionMismatchError(ValueError):
    """`rules_version` is a content hash with no registry to resolve it: the
    rules live in a TOML file that may be out of tree. Model v4 was fitted
    with `r-53559a9e0c4d`, which is not the shipped `rules.toml`. Scoring with
    the wrong rules silently changes the SIG_* features and produces numbers
    that look like the model's and are not, so this refuses instead."""


def load_model(conn: sqlite3.Connection, version: int) -> tuple[Model, str | None]:
    """Any version, not just the latest. The priors live in `triage_model`
    and the per-token counts in `triage_tokens`, so a recipe naming only
    `params_json` is incomplete."""
    row = conn.execute(
        "SELECT params_json FROM triage_model WHERE version = ?", (version,)
    ).fetchone()
    if row is None:
        raise UnknownModelVersionError(f"no triage_model row for version {version}")
    params = json.loads(row["params_json"])
    counts = {
        token_row["token"]: (token_row["lore_count"], token_row["noise_count"])
        for token_row in conn.execute(
            "SELECT token, lore_count, noise_count FROM triage_tokens WHERE model_version = ?",
            (version,),
        )
    }
    model = Model(
        lore_documents=params["lore_documents"],
        noise_documents=params["noise_documents"],
        counts=counts,
    )
    return model, params.get("rules_version")


def shadow_score(
    conn: sqlite3.Connection,
    version: int,
    rules: TriageRules,
    exchange_ids: Sequence[int],
    at: datetime,
) -> int:
    """Score exchanges under a model version and append the results as
    annotations WITHOUT activating the version, so `exchanges.p_lore` keeps
    reading whatever the live gate reads. That is the property the annotations
    table was built for: two scorers compared without refitting either."""
    model, recorded_rules = load_model(conn, version)
    if recorded_rules is not None and recorded_rules != rules.version:
        raise RulesVersionMismatchError(
            f"model {version} was fitted with rules {recorded_rules}, these are {rules.version}"
        )
    recipe = {
        "model_table": "triage_model",
        "model_version": version,
        "priors_in": "triage_model.params_json",
        "token_counts_in": "triage_tokens",
        "rules_version": rules.version,
    }
    written = 0
    ids = list(exchange_ids)
    for start in range(0, len(ids), SCORE_BATCH):
        batch = ids[start : start + SCORE_BATCH]
        inputs = exchange_inputs_for_ids(conn, batch)
        channels = {
            row["id"]: row["channel_id"]
            for row in conn.execute(
                f"SELECT id, channel_id FROM exchanges WHERE id IN ({','.join('?' * len(batch))})",
                batch,
            )
        }
        for exchange_id in batch:
            inputs_for = inputs.get(exchange_id)
            if inputs_for is None or exchange_id not in channels:  # pragma: no cover - ids are real
                continue
            tokens = features(
                inputs_for.messages,
                channels[exchange_id],
                inputs_for.reactions,
                inputs_for.attachments,
                rules,
            )
            record_annotation(
                conn,
                Annotation(
                    subject_kind="exchange",
                    subject_id=exchange_id,
                    scorer="p_lore",
                    scorer_version=version,
                    reproducibility="derived",
                    score=p_lore(model, tokens),
                    recipe=recipe,
                    source_ref="shadow-score",
                ),
                at,
            )
            written += 1
        conn.commit()
    return written


def scores_from_annotations(
    conn: sqlite3.Connection, scorer: str, version: int
) -> dict[int, float]:
    return {
        row["subject_id"]: row["score"]
        for row in conn.execute(
            "SELECT subject_id, score FROM annotations"
            " WHERE scorer = ? AND scorer_version = ? AND subject_kind = 'exchange'"
            " AND score IS NOT NULL",
            (scorer, version),
        )
    }


@dataclass(frozen=True)
class ScorerComparison:
    report: GainReport
    versions: tuple[int, ...]
    unscored: dict[int, int]


def compare_scorers(
    conn: sqlite3.Connection,
    mode: str,
    versions: Sequence[int],
    scorer: str = "p_lore",
    seed: int = 0,
    min_input_tokens: int = MIN_PLAUSIBLE_INPUT_TOKENS,
) -> ScorerComparison:
    """Rank one population by several scorer versions and report what each
    buys per token. The population is the exchanges with recorded extraction
    outcomes, so this needs no labels and makes no LLM calls."""
    rows = conn.execute(
        "SELECT e.id AS exchange_id, e.p_lore AS p_lore,"
        " r.input_tokens + r.output_tokens AS tokens,"
        " (SELECT COUNT(*) FROM claims c WHERE c.extraction_run_id = r.id) AS claims,"
        " r.input_tokens AS input_tokens"
        " FROM extraction_runs r JOIN exchanges e ON e.id = r.exchange_id"
        " WHERE r.outcome = 'ok' AND r.mode = ?"
        "   AND r.input_tokens IS NOT NULL AND r.output_tokens IS NOT NULL",
        (mode,),
    ).fetchall()
    credible = [row for row in rows if row["input_tokens"] >= min_input_tokens]
    candidates = [
        Candidate(
            exchange_id=row["exchange_id"],
            tokens=row["tokens"],
            claims=row["claims"],
            p_lore=row["p_lore"] if row["p_lore"] is not None else 0.0,
        )
        for row in credible
    ]
    wanted = {candidate.exchange_id for candidate in candidates}
    arms: dict[str, dict[int, float]] = {}
    unscored: dict[int, int] = {}
    for version in versions:
        scores = scores_from_annotations(conn, scorer, version)
        arms[f"{scorer} v{version}"] = scores
        unscored[version] = len(wanted - set(scores))
    return ScorerComparison(
        report=GainReport(
            mode=mode,
            exchanges=len(candidates),
            total_tokens=sum(candidate.tokens for candidate in candidates),
            total_claims=sum(candidate.claims for candidate in candidates),
            points=gain_points(candidates, seed, arms=arms),
            excluded_implausible=len(rows) - len(credible),
            min_input_tokens=min_input_tokens,
        ),
        versions=tuple(versions),
        unscored=unscored,
    )


def format_comparison(comparison: ScorerComparison) -> list[str]:
    from infovore.triage.gain_curve import format_gain_report

    lines = format_gain_report(comparison.report)
    missing = [
        f"v{version}: {count} of the population unscored"
        for version, count in sorted(comparison.unscored.items())
        if count
    ]
    # An arm that scored only part of the population is still ranked over the
    # whole of it, with its unscored candidates last, so say so rather than
    # letting the arm look worse for a reason that is not about the scorer.
    lines.extend(f"  {line}" for line in missing)
    return lines


@dataclass(frozen=True)
class ArmInterval:
    label: str
    point: float
    low: float
    high: float

    @property
    def separable_from_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


def _recovered(candidates: Sequence[Candidate], order: Sequence[float], fraction: float) -> float:
    """Claims recovered at a budget, for one resample. `order` is the sort key
    per candidate, lower first."""
    total_tokens = sum(candidate.tokens for candidate in candidates)
    total_claims = sum(candidate.claims for candidate in candidates)
    if total_claims == 0:
        return 0.0
    ceiling = total_tokens * fraction
    spent = recovered = 0
    for _, candidate in sorted(zip(order, candidates, strict=True), key=lambda pair: pair[0]):
        if spent + candidate.tokens > ceiling:
            break
        spent += candidate.tokens
        recovered += candidate.claims
    return 100.0 * recovered / total_claims


def bootstrap_difference(
    candidates: Sequence[Candidate],
    arms: Mapping[str, Mapping[int, float]],
    baseline: str,
    fraction: float = 0.10,
    resamples: int = 2000,
    seed: int = 0,
    include_random: bool = True,
) -> list[ArmInterval]:
    """Paired bootstrap of each arm MINUS a baseline arm, over the same
    resampled population each time.

    An unpaired comparison of two rankings over 321 exchanges cannot separate
    anything: the gate-versus-random sign already flipped between slices at
    this sample size (RULE #315, and RULE #206 on quoting a magnitude from a
    sample too small to carry one). Pairing removes the population variance
    that both arms share, which is the only reason a difference this small
    could ever be resolvable."""
    if not candidates or baseline not in arms:
        return []
    rng = random.Random(seed)
    labels = [label for label in arms if label != baseline]
    if include_random:
        labels.append(RANDOM)
    keys = {
        label: [-arms[label].get(candidate.exchange_id, -math.inf) for candidate in candidates]
        for label in arms
    }
    draws: dict[str, list[float]] = {label: [] for label in labels}
    size = len(candidates)
    for _ in range(resamples):
        picks = [rng.randrange(size) for _ in range(size)]
        sample = [candidates[i] for i in picks]
        base = _recovered(sample, [keys[baseline][i] for i in picks], fraction)
        for label in labels:
            # The random control is re-drawn inside every resample. One fixed
            # shuffle is a single draw from the distribution of random
            # orderings, so treating it as "random" makes the verdict depend
            # on which shuffle happened to be seeded (RULE #217).
            order = (
                [rng.random() for _ in picks]
                if label == RANDOM
                else [keys[label][i] for i in picks]
            )
            draws[label].append(_recovered(sample, order, fraction) - base)
    intervals = []
    for label in labels:
        values = sorted(draws[label])
        intervals.append(
            ArmInterval(
                label=label,
                point=sum(values) / len(values),
                low=values[int(0.025 * len(values))],
                high=values[min(int(0.975 * len(values)), len(values) - 1)],
            )
        )
    return intervals


def candidates_for(
    conn: sqlite3.Connection, mode: str, min_input_tokens: int = MIN_PLAUSIBLE_INPUT_TOKENS
) -> list[Candidate]:
    rows = conn.execute(
        "SELECT e.id AS exchange_id, e.p_lore AS p_lore,"
        " r.input_tokens + r.output_tokens AS tokens,"
        " (SELECT COUNT(*) FROM claims c WHERE c.extraction_run_id = r.id) AS claims,"
        " r.input_tokens AS input_tokens"
        " FROM extraction_runs r JOIN exchanges e ON e.id = r.exchange_id"
        " WHERE r.outcome = 'ok' AND r.mode = ?"
        "   AND r.input_tokens IS NOT NULL AND r.output_tokens IS NOT NULL",
        (mode,),
    ).fetchall()
    return [
        Candidate(
            exchange_id=row["exchange_id"],
            tokens=row["tokens"],
            claims=row["claims"],
            p_lore=row["p_lore"] if row["p_lore"] is not None else 0.0,
        )
        for row in rows
        if row["input_tokens"] >= min_input_tokens
    ]
