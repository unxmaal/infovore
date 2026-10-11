import argparse
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.config import ConfigError
from infovore.db.claims_v2 import run_ids
from infovore.db.wiki_articles import sections_for
from infovore.wiki.build import (
    DEFAULT_MIN_SECTION,
    Article,
    build_site,
    compute_stats,
    load_claims,
    page_groups,
)
from infovore.wiki.eligibility import DEFAULT_VERDICTS, VERDICTS_HELP, parse_verdicts
from infovore.wiki.groups import GROUPING_HELP, GROUPINGS, make_grouper_for, summarise
from infovore.wiki.tag_command import run_tag
from infovore.wiki.write_command import run_write

if TYPE_CHECKING:
    from infovore.cli import AppContext

DEFAULT_MIN_CLAIMS = 3


def _runs(conn: sqlite3.Connection, text: str | None) -> list[int] | None:
    if text is None:
        return None
    try:
        runs = sorted({int(p) for p in text.split(",")})
    except ValueError as error:
        raise ConfigError("--runs must be comma-separated run ids") from error
    if unknown := sorted(set(runs) - set(run_ids(conn))):
        raise ConfigError(f"unknown run {unknown[0]}")
    return runs


def _articles(conn: sqlite3.Connection, args: argparse.Namespace) -> Mapping[str, Article] | None:
    if args.article_run is None:
        return None
    if args.tag_run is None:
        raise ConfigError("--article-run needs --tag-run")
    if (
        conn.execute("SELECT 1 FROM article_runs WHERE id = ?", (args.article_run,)).fetchone()
        is None
    ):
        raise ConfigError(f"unknown article run {args.article_run}")
    return sections_for(conn, [args.article_run])


class WikiCommand:
    name = "wiki"
    help = "static Markdown wiki from publishable claims: build pages, print stats"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(
            dest="wiki_action", required=True, parser_class=argparse.ArgumentParser
        )
        build = sub.add_parser("build", help="write one page per topic plus an index")
        build.add_argument("--out", type=Path, required=True)
        build.add_argument("--min-claims", type=int, default=DEFAULT_MIN_CLAIMS)
        build.add_argument("--claim-gate", action="store_true", dest="claim_gate")
        build.add_argument("--verdicts", default=DEFAULT_VERDICTS, help=VERDICTS_HELP)
        build.add_argument("--grouping", choices=GROUPINGS, default="jaccard", help=GROUPING_HELP)
        build.add_argument(
            "--min-section", type=int, default=DEFAULT_MIN_SECTION, dest="min_section"
        )
        build.add_argument("--runs", default=None, help="comma-separated claim run ids")
        build.add_argument("--tag-run", type=int, default=None, dest="tag_run")
        build.add_argument("--article-run", type=int, default=None, dest="article_run")
        stats = sub.add_parser("stats", help="topics, pages and claims per page")
        stats.add_argument("--min-claims", type=int, default=DEFAULT_MIN_CLAIMS)
        stats.add_argument("--claim-gate", action="store_true", dest="claim_gate")
        stats.add_argument("--verdicts", default=DEFAULT_VERDICTS, help=VERDICTS_HELP)
        stats.add_argument("--grouping", choices=GROUPINGS, default="jaccard", help=GROUPING_HELP)
        stats.add_argument("--runs", default=None, help="comma-separated claim run ids")
        stats.add_argument("--tag-run", type=int, default=None, dest="tag_run")
        tag = sub.add_parser("tag", help="tag claims with the things they are about")
        tag.add_argument("--runs", required=True, help="comma-separated claim run ids")
        tag.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
        tag.add_argument("--model", required=True, help="model alias on the server")
        tag.add_argument("--concurrency", type=int, default=8)
        tag.add_argument("--limit", type=int, default=None, help="number of claims")
        tag.add_argument("--write", action="store_true", help="without it: print one request")
        tag.add_argument("--resume", action="store_true")
        tag.add_argument("--progress-every", type=int, default=500, dest="progress_every")
        write = sub.add_parser("write", help="write cited article sections for each topic")
        write.add_argument("--tag-run", type=int, required=True, dest="tag_run")
        write.add_argument("--runs", default=None, help="comma-separated claim run ids")
        write.add_argument("--claim-gate", action="store_true", dest="claim_gate")
        write.add_argument("--verdicts", default=DEFAULT_VERDICTS, help=VERDICTS_HELP)
        write.add_argument("--grouping", choices=GROUPINGS, default="jaccard", help=GROUPING_HELP)
        write.add_argument(
            "--min-section", type=int, default=DEFAULT_MIN_SECTION, dest="min_section"
        )
        write.add_argument("--min-claims", type=int, default=DEFAULT_MIN_CLAIMS, dest="min_claims")
        write.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
        write.add_argument("--model", required=True, help="model alias on the server")
        write.add_argument("--concurrency", type=int, default=8)
        write.add_argument("--limit", type=int, default=None, help="number of topics")
        write.add_argument("--write", action="store_true", help="without it: print one request")
        write.add_argument("--resume", action="store_true")
        write.add_argument("--log", type=Path, default=None, help="JSONL of drops and failures")
        write.add_argument("--progress-every", type=int, default=10, dest="progress_every")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        if args.wiki_action == "tag":
            return run_tag(context, args)
        if args.wiki_action == "write":
            return run_write(context, args, _runs(context.conn, args.runs))
        articles = _articles(context.conn, args) if args.wiki_action == "build" else None
        claims, excluded = load_claims(
            context.conn,
            tech_only=args.claim_gate,
            salt=context.settings.pseudonym_salt,
            runs=_runs(context.conn, args.runs),
            tag_runs=None if args.tag_run is None else [args.tag_run],
            verdicts=parse_verdicts(args.verdicts),
        )
        stats = compute_stats(claims, args.min_claims)
        grouper = make_grouper_for(args.grouping, claims)
        lines = [
            f"topics: {stats.topics}",
            f"pages: {stats.pages}",
            f"claims: {len(claims)}",
            f"unassigned: {stats.unassigned}",
            f"on pages: {stats.on_pages}/{len(claims)}",
            f"excluded: {excluded}",
            f"corroboration: {summarise(page_groups(claims, args.min_claims, grouper))}",
        ]
        if args.wiki_action == "build":
            build_site(claims, args.min_claims, args.out, articles, args.min_section, grouper)
            lines.append(f"wrote: {args.out}")
        else:
            lines += [f"  {label}: {n}" for label, n in stats.distribution().items()]
        context.stdout.write("\n".join(lines) + "\n")
        return 0
