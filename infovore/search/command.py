import argparse
from typing import TYPE_CHECKING

from infovore.db.messages_fts import (
    DEFAULT_CONTEXT,
    DEFAULT_SEARCH_LIMIT,
    ContextMessage,
    MessageHit,
    message_context,
    search_messages,
)

if TYPE_CHECKING:
    from infovore.cli import AppContext


def format_hit(hit: MessageHit) -> str:
    exchange = f"ex:{hit.exchange_id}" if hit.exchange_id is not None else "ex:none"
    return f"{hit.created_at} #{hit.channel_name} [{exchange}] {hit.author_name}: {hit.content}"


def format_context_line(message: ContextMessage) -> str:
    marker = ">" if message.is_hit else " "
    return f"{marker} {message.created_at} {message.author_name}: {message.content}"


class SearchCommand:
    name = "search"
    help = "full-text search the raw message corpus"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("terms", nargs="+")
        parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT)
        parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        query = " ".join(args.terms)
        hits = search_messages(context.conn, query, limit=args.limit)
        if not hits:
            context.stdout.write(f"search: no matches for {query!r}\n")
            return ExitCode.OK
        for hit in hits:
            if args.context <= 0:
                context.stdout.write(f"{format_hit(hit)}\n")
                continue
            exchange = f"ex:{hit.exchange_id}" if hit.exchange_id is not None else "ex:none"
            context.stdout.write(f"#{hit.channel_name} [{exchange}]\n")
            for message in message_context(context.conn, hit, args.context):
                context.stdout.write(f"{format_context_line(message)}\n")
            context.stdout.write("\n")
        context.stdout.write(f"search: {len(hits)} match(es) for {query!r}\n")
        return ExitCode.OK
