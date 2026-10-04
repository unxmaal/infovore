import argparse
import threading
from typing import TYPE_CHECKING

from infovore.db.reviewed_words import decision_counts
from infovore.triage.lexicon import load_lexicon
from infovore.words.httpd import DEFAULT_WORDS_PORT, listening_url, shutdown_all, start_all
from infovore.words.review import DEFAULT_TOP_N, ReviewQueue, candidates, reviewed_gain

if TYPE_CHECKING:
    from infovore.cli import AppContext


class WordsCommand:
    name = "words"
    help = "review frequent words in undecided conversations for the tech lexicon"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(
            dest="words_action", required=True, parser_class=argparse.ArgumentParser
        )
        serve = sub.add_parser("serve", help="serve the word judging page until Ctrl-C")
        serve.add_argument("--host", action="append", default=None, dest="hosts")
        serve.add_argument("--port", type=int, default=DEFAULT_WORDS_PORT)
        serve.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
        report = sub.add_parser("report", help="reviewed counts and projected gain")
        report.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
        report.add_argument("--show", type=int, default=0, help="print the top N candidates")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        found = candidates(context.conn, args.top_n)
        if args.words_action == "report":
            approved, rejected = decision_counts(context.conn)
            undecided, with_word = reviewed_gain(context.conn)
            context.stdout.write(
                f"lexicon: {load_lexicon(context.conn).version}\n"
                f"reviewed: {approved + rejected} (approved {approved}, not tech {rejected})\n"
                f"undecided conversations: {undecided}\n"
                f"with an approved word: {with_word}\n"
                f"candidates: {len(found)}\n"
            )
            for word, count in found[: args.show]:
                context.stdout.write(f"{word}\t{count}\n")
            return int(ExitCode.OK)

        from infovore.sift.httpd import block_until_interrupted

        context.stdout.write(f"{len(found)} candidate words\n")
        servers = start_all(
            args.hosts or ["127.0.0.1"],
            args.port,
            context.conn,
            context.clock,
            ReviewQueue(found),
        )
        try:
            for server in servers:
                context.stdout.write(f"listening on {listening_url(server)}\n")
            context.stdout.flush()
            block_until_interrupted(threading.Event())
        finally:
            shutdown_all(servers)
        return int(ExitCode.OK)
