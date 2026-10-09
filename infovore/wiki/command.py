import argparse
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.config import ConfigError
from infovore.db.claims_v2 import run_ids
from infovore.wiki.build import build_site, compute_stats, load_claims
from infovore.wiki.tag_command import run_tag

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
        build.add_argument("--runs", default=None, help="comma-separated claim run ids")
        stats = sub.add_parser("stats", help="topics, pages and claims per page")
        stats.add_argument("--min-claims", type=int, default=DEFAULT_MIN_CLAIMS)
        stats.add_argument("--claim-gate", action="store_true", dest="claim_gate")
        stats.add_argument("--runs", default=None, help="comma-separated claim run ids")
        tag = sub.add_parser("tag", help="tag claims with the things they are about")
        tag.add_argument("--runs", required=True, help="comma-separated claim run ids")
        tag.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
        tag.add_argument("--model", required=True, help="model alias on the server")
        tag.add_argument("--concurrency", type=int, default=8)
        tag.add_argument("--limit", type=int, default=None, help="number of claims")
        tag.add_argument("--write", action="store_true", help="without it: print one request")
        tag.add_argument("--resume", action="store_true")
        tag.add_argument("--progress-every", type=int, default=500, dest="progress_every")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        if args.wiki_action == "tag":
            return run_tag(context, args)
        claims, excluded = load_claims(
            context.conn,
            tech_only=args.claim_gate,
            salt=context.settings.pseudonym_salt,
            runs=_runs(context.conn, args.runs),
        )
        stats = compute_stats(claims, args.min_claims)
        lines = [
            f"topics: {stats.topics}",
            f"pages: {stats.pages}",
            f"claims: {len(claims)}",
            f"unassigned: {stats.unassigned}",
            f"excluded: {excluded}",
        ]
        if args.wiki_action == "build":
            build_site(claims, args.min_claims, args.out)
            lines.append(f"wrote: {args.out}")
        else:
            lines += [f"  {label}: {n}" for label, n in stats.distribution().items()]
        context.stdout.write("\n".join(lines) + "\n")
        return 0
