import argparse
from typing import TYPE_CHECKING

from infovore.config import ConfigError
from infovore.db.batch import exchange_inputs_for_ids
from infovore.eval.slices import BUILD, slice_ids, slice_names
from infovore.triage.cascade import (
    SCORERS,
    StageReport,
    run_cascade,
    stage_reports,
    try_fit,
    tune_high,
    tuning_samples,
    write_outcomes,
)
from infovore.triage.human import held_out_ids, training_labels
from infovore.triage.lexicon import (
    MAX_OFF,
    MIN_COUNT,
    MIN_RATIO,
    MINE_LIMIT,
    OFF_CHANNELS,
    TECH_CHANNELS,
    collisions,
    load_lexicon,
    message_hits,
    mine_terms,
    score_lexicon,
)

if TYPE_CHECKING:
    from infovore.cli import AppContext


class RelevanceCommand:
    name = "relevance"
    help = "relevance cascade: lexicon, then Bayes, then residue (#209)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(
            dest="relevance_action", required=True, parser_class=argparse.ArgumentParser
        )
        cascade = sub.add_parser("cascade", help="run the cascade over eval slices and report")
        cascade.add_argument("--slices", default=None, help="comma-separated slice names")
        cascade.add_argument("--write", action="store_true", help="record derived annotations")
        cascade.add_argument("--explain", type=int, default=None, metavar="EXCHANGE_ID")
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
        if args.relevance_action == "mine":
            return self._mine(context, args)
        if args.relevance_action == "collisions":
            return self._collisions(context, args)
        if args.explain is not None:
            return self._explain(context, args.explain)
        return self._cascade(context, args)

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
        if not args.slices:
            raise ConfigError("--slices is required (comma-separated slice names)")
        names = args.slices.split(",")
        unknown = [n for n in names if n not in slice_names(conn)]
        if unknown:
            raise ConfigError(f"unknown slice(s): {', '.join(unknown)}")
        lexicon = load_lexicon()
        t_high = tune_high(tuning_samples(conn, lexicon))
        fit, why = try_fit(conn)
        labels, _ = training_labels(conn)
        held = held_out_ids(conn)
        context.stdout.write(
            f"lexicon v={lexicon.version} size={lexicon.size} {lexicon.sources}"
            f" t_high={t_high:.3f} (tuned on {BUILD} labels outside gold and s2)\n"
        )
        everything: set[int] = set()
        for name in names:
            ids = slice_ids(conn, name)
            if name == BUILD:
                ids = [i for i in ids if i not in held]
            everything.update(ids)
            outcomes = run_cascade(conn, ids, lexicon, t_high, fit)
            kind = "tuning, held-out excluded" if name == BUILD else "held-out"
            context.stdout.write(f"slice {name} ({kind}): n={len(ids)}\n")
            for report in stage_reports(outcomes, labels):
                context.stdout.write(self._line(report, fit is None, why))
        if args.write:
            ids = sorted(everything)
            outcomes = run_cascade(conn, ids, lexicon, t_high, fit)
            versions = write_outcomes(conn, outcomes, lexicon, t_high, fit, context.clock.now())
            written = ", ".join(f"{SCORERS[s]} v{v}" for s, v in versions.items())
            context.stdout.write(f"wrote {len(ids)} exchanges: {written}\n")
        return int(ExitCode.OK)

    @staticmethod
    def _line(report: StageReport, no_bayes: bool, why: str) -> str:
        if report.stage == "residue":
            return f"stage residue: n={report.decided} share={report.share:.3f}\n"
        if report.stage == "bayes" and no_bayes:
            return f"stage bayes: abstains on everything ({why})\n"
        accuracy = "n/a" if report.accuracy is None else f"{report.accuracy:.3f}"
        return (
            f"stage {report.stage}: decided={report.decided}"
            f" relevant={report.relevant} irrelevant={report.irrelevant}"
            f" labelled={report.labelled} accuracy={accuracy}"
            f" tp={report.tp} fp={report.fp} fn={report.fn} tn={report.tn}\n"
        )
