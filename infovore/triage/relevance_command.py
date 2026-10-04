import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.config import ConfigError
from infovore.db.annotations import Annotation, record_annotation
from infovore.db.batch import exchange_inputs_for_ids
from infovore.eval.slices import BUILD, slice_ids, slice_names
from infovore.rows import Label
from infovore.triage.cascade import (
    SCORERS,
    EmbedStage,
    Outcome,
    StageReport,
    current_exchange_ids,
    residue_channels,
    run_cascade,
    run_cascade_batched,
    stage_reports,
    tune_high,
    tuning_labels,
    tuning_samples,
    write_outcomes,
)
from infovore.triage.embed import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    MAX_CHARS,
    POOLS,
    EmbeddingCache,
    Summary,
    cross_validate,
    embed_scores,
)
from infovore.triage.embed import (
    SCORER as EMBED_SCORER,
)
from infovore.triage.embed_backend import load_embedder
from infovore.triage.embed_stage import build_embed_stage, default_cache_path
from infovore.triage.human import held_out_ids, training_labels
from infovore.triage.lexicon import (
    MAX_OFF,
    MIN_COUNT,
    MIN_RATIO,
    MINE_LIMIT,
    OFF_CHANNELS,
    TECH_CHANNELS,
    Lexicon,
    collisions,
    load_lexicon,
    message_hits,
    mine_terms,
    score_lexicon,
)
from infovore.triage.llm_score import configure as configure_llm_score
from infovore.triage.llm_score import run_llm_score

ALL_BATCH = 2000
RESIDUE_CHANNELS = 15

if TYPE_CHECKING:
    from infovore.cli import AppContext


class RelevanceCommand:
    name = "relevance"
    help = "relevance cascade: lexicon, then embedding scorer, then residue (#209)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(
            dest="relevance_action", required=True, parser_class=argparse.ArgumentParser
        )
        cascade = sub.add_parser("cascade", help="run the cascade over eval slices and report")
        cascade.add_argument("--slices", default=None, help="comma-separated slice names")
        cascade.add_argument("--write", action="store_true", help="record derived annotations")
        cascade.add_argument("--all", action="store_true", help="every current exchange")
        cascade.add_argument("--explain", type=int, default=None, metavar="EXCHANGE_ID")
        for name, text in (
            ("compare", "stratified CV: human Bayes vs embedding + logistic head"),
            ("embed-score", "write p_relevant_embed for labelled exchanges"),
        ):
            embed = sub.add_parser(name, help=text)
            embed.add_argument("--cv", type=int, default=5)
            embed.add_argument("--model", default=DEFAULT_MODEL)
            embed.add_argument("--revision", default=DEFAULT_REVISION)
            embed.add_argument("--max-chars", type=int, default=MAX_CHARS, dest="max_chars")
            embed.add_argument("--pool", choices=POOLS, default="first")
            embed.add_argument("--cache", default=None, help="embedding cache file")
            if name == "compare":
                embed.add_argument("--residue", action="store_true")
        configure_llm_score(sub)
        mine = sub.add_parser("mine", help="candidate lexicon terms by channel log-odds")
        mine.add_argument("--tech", default=",".join(TECH_CHANNELS))
        mine.add_argument("--off", default=",".join(OFF_CHANNELS))
        mine.add_argument("--min-count", type=int, default=MIN_COUNT, dest="min_count")
        mine.add_argument("--limit", type=int, default=MINE_LIMIT)
        clash = sub.add_parser("collisions", help="lexicon terms common in off-topic channels")
        clash.add_argument("--tech", default=",".join(TECH_CHANNELS))
        clash.add_argument("--off", default=",".join(OFF_CHANNELS))
        clash.add_argument("--max-off", type=int, default=MAX_OFF, dest="max_off")
        clash.add_argument("--min-ratio", type=float, default=MIN_RATIO, dest="min_ratio")
        clash.add_argument("--terms", default=None, help="comma-separated; default: the lexicon")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        if args.relevance_action == "llm-score":
            return run_llm_score(context, args)
        if args.relevance_action == "mine":
            return self._mine(context, args)
        if args.relevance_action == "compare":
            return self._compare(context, args)
        if args.relevance_action == "embed-score":
            return self._embed_score(context, args)
        if args.relevance_action == "collisions":
            return self._collisions(context, args)
        if args.explain is not None:
            return self._explain(context, args.explain)
        return self._cascade(context, args)

    @staticmethod
    def _cache(context: "AppContext", args: argparse.Namespace) -> EmbeddingCache:
        path = args.cache or context.settings.db_path.parent / "embed-cache.db"
        return EmbeddingCache(Path(path))

    def _compare(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        embedder = load_embedder(args.model, args.revision)
        result = cross_validate(
            context.conn,
            embedder,
            self._cache(context, args),
            args.cv,
            args.max_chars,
            args.residue,
            context.settings.exclude_channels,
            args.pool,
        )
        context.stdout.write(
            f"model={embedder.model_id} revision={embedder.revision} cv={result.folds}"
            f" max_chars={args.max_chars} pool={args.pool}\n"
        )
        header = (
            "scorer\trelevant\tirrelevant\tauc\tthreshold\tprecision\trecall\tf1"
            "\tirr_recall@rel>=0.95\n"
        )
        for name, comparison in (("all labels", result.all_labels), ("residue", result.residue)):
            if comparison is None:
                continue
            context.stdout.write(f"{name}\n{header}")
            context.stdout.write(self._row("naive_bayes", comparison.bayes))
            context.stdout.write(self._row("embed_lr", comparison.embed))
        return int(ExitCode.OK)

    @staticmethod
    def _row(name: str, s: Summary) -> str:
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        return (
            f"{name}\t{s.relevant}\t{s.irrelevant}\t{fmt(s.auc)}\t{fmt(s.threshold)}"
            f"\t{s.precision:.3f}\t{s.recall:.3f}\t{s.f1:.3f}\t{fmt(s.irrelevant_recall)}\n"
        )

    def _embed_score(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        conn = context.conn
        embedder = load_embedder(args.model, args.revision)
        scores, recipe = embed_scores(
            conn,
            embedder,
            self._cache(context, args),
            args.cv,
            args.max_chars,
            context.settings.exclude_channels,
            args.pool,
        )
        row = conn.execute(
            "SELECT MAX(scorer_version) AS v FROM annotations WHERE scorer = ?", (EMBED_SCORER,)
        ).fetchone()
        version = int(row["v"] or 0) + 1
        for eid, score in sorted(scores.items()):
            note = Annotation(
                "exchange",
                eid,
                EMBED_SCORER,
                version,
                "derived",
                score=score,
                recipe=recipe,
                source_ref="relevance-embed-score",
            )
            record_annotation(conn, note, context.clock.now())
        conn.commit()
        context.stdout.write(f"wrote {len(scores)} labelled exchanges: {EMBED_SCORER} v{version}\n")
        return int(ExitCode.OK)

    def _mine(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        mined = mine_terms(
            context.conn,
            args.tech.split(","),
            args.off.split(","),
            args.min_count,
            args.limit,
            load_lexicon(),
        )
        if not mined:
            context.stdout.write("no candidates\n")
        for m in mined:
            context.stdout.write(f"{m.term}\t{m.tech}\t{m.off}\t{m.log_odds:.2f}\n")
        return int(ExitCode.OK)

    def _collisions(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        terms = args.terms.split(",") if args.terms else sorted(load_lexicon().terms)
        found = collisions(
            context.conn,
            terms,
            args.tech.split(","),
            args.off.split(","),
            args.max_off,
            args.min_ratio,
        )
        if not found:
            context.stdout.write("no collisions\n")
        for c in found:
            context.stdout.write(f"{c.term}\t{c.tech}\t{c.off}\n")
        return int(ExitCode.OK)

    def _explain(self, context: "AppContext", exchange_id: int) -> int:
        from infovore.cli import ExitCode

        lexicon = load_lexicon()
        messages = exchange_inputs_for_ids(context.conn, [exchange_id])[exchange_id].messages
        if not messages:
            raise ConfigError(f"exchange {exchange_id} has no messages")
        score = score_lexicon(lexicon, messages)
        context.stdout.write(
            f"exchange {exchange_id} lexicon v={lexicon.version}"
            f" share={score.share:.2f} hits={score.hits} messages={score.messages}\n"
        )
        for message in messages:
            hits = ",".join(message_hits(lexicon, message.content)) or "-"
            text = message.content.replace("\n", " ")[:80]
            context.stdout.write(f"{message.id}\t{hits}\t{text}\n")
        return int(ExitCode.OK)

    def _cascade(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        conn = context.conn
        if not args.slices and not args.all:
            raise ConfigError("--slices or --all is required")
        names = args.slices.split(",") if args.slices else []
        unknown = [n for n in names if n not in slice_names(conn)]
        if unknown:
            raise ConfigError(f"unknown slice(s): {', '.join(unknown)}")
        lexicon = load_lexicon()
        exclude = context.settings.exclude_channels
        tuned = tuning_labels(conn, exclude)
        lore = sum(label is Label.LORE for label in tuned.values())
        t_high = tune_high(tuning_samples(conn, lexicon, exclude))
        stage = build_embed_stage(conn, exclude, default_cache_path(context.settings.db_path))
        labels, _ = training_labels(conn)
        held = held_out_ids(conn)
        context.stdout.write(self._thresholds(stage))
        context.stdout.write(
            f"lexicon v={lexicon.version} size={lexicon.size} {lexicon.sources}"
            f" t_high={t_high:.3f} (tuned on {BUILD} random-slice labels, held-out and"
            f" queue-sourced excluded: n={len(tuned)}"
            f" relevant={lore} irrelevant={len(tuned) - lore})\n"
        )
        everything: set[int] = set()
        for name in names:
            ids = slice_ids(conn, name)
            if name == BUILD:
                ids = [i for i in ids if i not in held]
            everything.update(ids)
            outcomes = run_cascade(conn, ids, lexicon, t_high, stage, exclude)
            kind = "tuning, held-out excluded" if name == BUILD else "held-out"
            context.stdout.write(f"slice {name} ({kind}): n={len(ids)}\n")
            for report in stage_reports(outcomes, labels):
                context.stdout.write(self._line(report, stage))
        everything_outcomes = None
        if args.all:
            everything = set(current_exchange_ids(conn))
            everything_outcomes = self._all(
                context, sorted(everything), lexicon, t_high, stage, labels
            )
        if args.write:
            ids = sorted(everything)
            outcomes = everything_outcomes or run_cascade(
                conn, ids, lexicon, t_high, stage, exclude
            )
            versions = write_outcomes(conn, outcomes, lexicon, t_high, stage, context.clock.now())
            written = ", ".join(f"{SCORERS[s]} v{v}" for s, v in versions.items())
            context.stdout.write(f"wrote {len(ids)} exchanges: {written}\n")
        return int(ExitCode.OK)

    def _all(
        self,
        context: "AppContext",
        ids: list[int],
        lexicon: Lexicon,
        t_high: float,
        stage: EmbedStage,
        labels: dict[int, Label],
    ) -> list[Outcome]:
        def progress(done: int, total: int) -> None:
            sys.stderr.write(f"progress {done}/{total}\n")

        outcomes = run_cascade_batched(
            context.conn,
            ids,
            lexicon,
            t_high,
            stage,
            context.settings.exclude_channels,
            ALL_BATCH,
            progress,
        )
        context.stdout.write(f"corpus: n={len(ids)}\n")
        for report in stage_reports(outcomes, labels):
            context.stdout.write(self._line(report, stage, corpus=True))
        for name, count in residue_channels(context.conn, outcomes, RESIDUE_CHANNELS):
            context.stdout.write(f"residue channel {name}: {count}\n")
        return outcomes

    @staticmethod
    def _thresholds(stage: EmbedStage) -> str:
        if stage.abstain_reason:
            return ""
        labels = stage.recipe["labels"]
        return (
            f"embed {stage.recipe['model']}@{stage.recipe['revision']} pool={stage.recipe['pool']}"
            f" irrelevant_below={stage.t_irrelevant:.4f}"
            f" relevant_at_or_above={stage.t_relevant:.4f}"
            f" labels={labels}\n"
        )

    @staticmethod
    def _line(report: StageReport, stage: EmbedStage, corpus: bool = False) -> str:
        if report.stage == "residue":
            return f"stage residue: n={report.decided} share={report.share:.3f}\n"
        if report.stage == "no_text":
            return (
                f"stage no_text: n={report.decided} share={report.share:.3f}"
                " (set aside, not scored)\n"
            )
        if report.stage == "embed" and stage.abstain_reason:
            return f"stage embed: abstains on everything ({stage.abstain_reason})\n"
        accuracy = "n/a" if report.accuracy is None else f"{report.accuracy:.3f}"
        share = f" share={report.share:.3f}" if corpus else ""
        return (
            f"stage {report.stage}: decided={report.decided}"
            f" relevant={report.relevant} irrelevant={report.irrelevant}{share}"
            f" labelled={report.labelled} accuracy={accuracy}"
            f" tp={report.tp} fp={report.fp} fn={report.fn} tn={report.tn}\n"
        )
