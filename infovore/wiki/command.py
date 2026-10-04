import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.wiki.build import build_site, compute_stats, load_claims

if TYPE_CHECKING:
    from infovore.cli import AppContext

DEFAULT_MIN_CLAIMS = 3


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
        stats = sub.add_parser("stats", help="topics, pages and claims per page")
        stats.add_argument("--min-claims", type=int, default=DEFAULT_MIN_CLAIMS)

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        claims, excluded = load_claims(context.conn)
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
