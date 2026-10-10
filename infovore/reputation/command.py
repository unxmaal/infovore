import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from infovore.claims.redact import pseudonyms, require_salt
from infovore.config import ConfigError
from infovore.reputation.compare import compare, embed_scores
from infovore.reputation.evaluate import PopulationReport, evaluate_population, scored_pairs
from infovore.reputation.evidence import Evidence, build_evidence
from infovore.reputation.exchange import score_exchanges
from infovore.reputation.people import People, load_people, people_path
from infovore.reputation.score import (
    K_LABEL,
    K_RESPONSE,
    Reputation,
    Smoothing,
    breakdown,
    build_reputation,
    scores,
)
from infovore.reputation.short import short_messages, short_test
from infovore.reputation.stats import Interval, accuracy_at, spearman, youden_threshold
from infovore.rows import Label
from infovore.triage.embed_stage import default_cache_path
from infovore.triage.human import held_out_ids, trainable_labels, training_labels
from infovore.triage.lexicon import load_lexicon

if TYPE_CHECKING:
    from infovore.cli import AppContext

TOP = 30


@dataclass(frozen=True)
class Built:
    people: People
    evidence: Evidence
    reputation: Reputation
    held_labels: dict[int, Label]
    labels: dict[int, Label]
    salt: str


def _smoothing(args: argparse.Namespace) -> Smoothing:
    if args.k_response < 0 or args.k_label < 0:
        raise ConfigError("--k-response and --k-label must not be negative")
    return Smoothing(args.k_response, args.k_label)


def _build(context: "AppContext", args: argparse.Namespace, keep: set[int]) -> Built:
    salt = require_salt(context.settings.pseudonym_salt)
    smoothing = _smoothing(args)
    people = load_people(people_path(os.environ, args.people))
    conn, exclude = context.conn, context.settings.exclude_channels
    held = held_out_ids(conn)
    everything, _ = training_labels(conn, exclude_channels=exclude)
    labels, _ = trainable_labels(conn, exclude)
    held_labels = {eid: label for eid, label in everything.items() if eid in held}
    evidence = build_evidence(conn, people, salt, labels, exclude, held, keep | set(labels))
    return Built(
        people, evidence, build_reputation(evidence, people, smoothing), held_labels, labels, salt
    )


def _interval(value: Interval) -> str:
    def part(number: float | None) -> str:
        return "n/a" if number is None else f"{number:.3f}"

    return (
        f"n={value.n} relevant={value.relevant} irrelevant={value.irrelevant}"
        f" auc={part(value.auc)} [{part(value.low)}, {part(value.high)}]"
    )


def _population(report: PopulationReport) -> dict[str, Any]:
    return {
        "overall": asdict(report.overall),
        "subsets": {name: asdict(value) for name, value in report.subsets.items()},
    }


def analyse(context: "AppContext", args: argparse.Namespace) -> dict[str, Any]:
    conn, exclude = context.conn, context.settings.exclude_channels
    lexicon = load_lexicon(conn)
    short = short_messages(conn, lexicon, exclude)
    built = _build(context, args, {m.exchange_id for m in short})
    reputation, held_labels, labels = built.reputation, built.held_labels, built.labels
    held_scores = score_exchanges(conn, reputation, sorted(held_labels))
    scores_loo = score_exchanges(conn, reputation, sorted(labels))
    held = evaluate_population(conn, held_scores, held_labels, args.seed)
    secondary = evaluate_population(conn, scores_loo, labels, args.seed)
    threshold = youden_threshold(scored_pairs(scores_loo, labels, sorted(labels)))
    accuracy = accuracy_at(threshold, scored_pairs(held_scores, held_labels, sorted(held_labels)))
    everyone = scores(reputation)
    persons = sorted(everyone)
    report: dict[str, Any] = {
        "seed": args.seed,
        "k_response": reputation.smoothing.k_response,
        "k_label": reputation.smoothing.k_label,
        "persons": len(persons),
        "people_file": {"listed": len(built.people.members), "banned": len(built.people.banned)},
        "held_out": {
            **_population(held),
            "threshold": threshold,
            "accuracy": None if accuracy is None else asdict(accuracy),
        },
        "secondary": _population(secondary),
        "volume_correlation": spearman(
            [everyone[p] for p in persons], [built.evidence.volume(p) for p in persons]
        ),
        "embed": None,
        "short": None,
    }
    if not args.no_embed:
        found = embed_scores(
            conn, exclude, default_cache_path(context.settings.db_path), sorted(held_labels)
        )
        if found is not None:
            report["embed"] = {
                "held_out": asdict(compare(found.held_out, held_scores, held_labels, args.seed)),
                "secondary": asdict(compare(found.out_of_fold, scores_loo, labels, args.seed)),
            }
    outcome = short_test(reputation, short, args.seed)
    report["short"] = None if outcome is None else asdict(outcome)
    return report


def render(report: dict[str, Any]) -> str:
    def interval(data: dict[str, Any]) -> str:
        return _interval(Interval(**data))

    def number(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        f"reputation eval: seed={report['seed']} k_response={report['k_response']:g}"
        f" k_label={report['k_label']:g} persons={report['persons']}"
        f" people-file listed={report['people_file']['listed']}"
        f" banned={report['people_file']['banned']}"
    ]
    held = report["held_out"]
    lines.append(f"held-out: {interval(held['overall'])}")
    accuracy = held["accuracy"]
    if accuracy is None:
        lines.append("held-out accuracy: n/a (no threshold)")
    else:
        lines.append(
            f"held-out accuracy at youden threshold {accuracy['threshold']:.4f}:"
            f" {accuracy['accuracy']:.3f} vs majority baseline {accuracy['baseline']:.3f}"
            f" (n={accuracy['n']})"
        )
    lines += [f"  held-out {name}: {interval(v)}" for name, v in held["subsets"].items()]
    lines.append(f"secondary (leave-one-out): {interval(report['secondary']['overall'])}")
    lines += [
        f"  secondary {name}: {interval(v)}" for name, v in report["secondary"]["subsets"].items()
    ]
    if report["embed"] is None:
        lines.append("embed comparison: skipped")
    else:
        for population in ("held_out", "secondary"):
            for name, value in report["embed"][population].items():
                lines.append(f"embed comparison {population} {name}: {interval(value)}")
    short = report["short"]
    if short is None:
        lines.append("short low-hit messages: none labelled")
    else:
        lines.append(
            f"short low-hit messages: n={short['n']} top={short['top_n']}"
            f" base keep rate {short['base_rate']:.3f} top keep rate {short['top_rate']:.3f}"
            f" wilson [{short['low']:.3f}, {short['high']:.3f}] p={short['p']:.4f}"
        )
    lines.append(f"reputation vs message volume (spearman): {number(report['volume_correlation'])}")
    return "\n".join(lines) + "\n"


def top_lines(built: Built, count: int) -> list[str]:
    everyone = scores(built.reputation)
    ranked = sorted(
        (p for p in everyone if not built.people.is_banned(p)), key=lambda p: (-everyone[p], p)
    )[:count]
    names = pseudonyms((built.people.representative(p) for p in ranked), built.salt)
    lines = []
    for rank, person in enumerate(ranked, start=1):
        cells = built.evidence.totals[person]
        parts = breakdown(cells, built.evidence.priors, built.reputation.smoothing)
        detail = " ".join(
            f"{signal}={lift:+.2f}({cells[signal][0]:g}/{cells[signal][1]:g})"
            for signal, lift in parts.items()
            if signal in cells
        )
        lines.append(
            f"{rank}\t{names[built.people.representative(person)]}\t{everyone[person]:+.3f}"
            f"\tmessages={built.evidence.volume(person):g}\t{detail}"
        )
    return lines


class ReputationCommand:
    name = "reputation"
    help = "author reputation from independent evidence, held-out evaluation (#320)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(
            dest="reputation_action", required=True, parser_class=argparse.ArgumentParser
        )
        evaluation = sub.add_parser("eval", help="held-out and leave-one-out evaluation")
        top = sub.add_parser("top", help="top reputations by pseudonym with an evidence breakdown")
        for each in (evaluation, top):
            each.add_argument("--people", default=None, help="TOML people file")
            each.add_argument("--k-response", type=float, default=K_RESPONSE, dest="k_response")
            each.add_argument("--k-label", type=float, default=K_LABEL, dest="k_label")
        evaluation.add_argument("--seed", type=int, default=0)
        evaluation.add_argument("--no-embed", action="store_true", dest="no_embed")
        evaluation.add_argument("--json", action="store_true", dest="as_json")
        top.add_argument("--n", type=int, default=TOP)

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        if args.reputation_action == "top":
            built = _build(context, args, set())
            context.stdout.write("\n".join(top_lines(built, args.n)) + "\n")
            return int(ExitCode.OK)
        report = analyse(context, args)
        if args.as_json:
            context.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        else:
            context.stdout.write(render(report))
        return int(ExitCode.OK)
