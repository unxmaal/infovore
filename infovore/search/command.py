import argparse
from typing import TYPE_CHECKING

from infovore.db.archive import export_archive
from infovore.db.exchange_search import ExchangeHit, search_exchanges
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


def format_exchange_hit(hit: ExchangeHit) -> list[str]:
    gate = "" if hit.in_archive else " REJECTED"
    lines = [
        f"#{hit.channel_name} [ex:{hit.exchange_id}]{gate} {hit.started_at} .. {hit.ended_at}"
        f" {hit.participants} participant(s), {hit.hit_count} matching message(s)"
    ]
    lines.extend(f"  > {text}" for text in hit.snippets)
    lines.append(f"  {hit.jump_url}")
    return lines


class SearchCommand:
    name = "search"
    help = "full-text search the raw message corpus"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("terms", nargs="+")
        parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT)
        parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
        parser.add_argument("--exchanges", action="store_true")
        parser.add_argument("--all", action="store_true", dest="include_rejected")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        query = " ".join(args.terms)
        if args.exchanges:
            return self._run_exchanges(context, query, args)
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

    def _run_exchanges(self, context: "AppContext", query: str, args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        settings = context.settings
        found = search_exchanges(
            context.conn,
            query,
            include_rejected=args.include_rejected,
            exclude_channels=settings.exclude_channels,
            limit=args.limit,
        )
        if not found:
            context.stdout.write(f"search: no exchanges for {query!r}\n")
            return ExitCode.OK
        for hit in found:
            for line in format_exchange_hit(hit):
                context.stdout.write(f"{line}\n")
            context.stdout.write("\n")
        context.stdout.write(f"search: {len(found)} exchange(s) for {query!r}\n")
        return ExitCode.OK


class ExportArchiveCommand:
    name = "export-archive"
    help = "write a shareable SQLite archive of archived exchanges"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("dest")
        parser.add_argument("--force", action="store_true")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        from infovore.cli import ExitCode

        settings = context.settings
        try:
            report = export_archive(
                context.conn,
                args.dest,
                exclude_channels=settings.exclude_channels,
                force=args.force,
            )
        except FileExistsError as error:
            context.stdout.write(f"export-archive: {error}\n")
            return ExitCode.FAILURE
        context.stdout.write(
            f"export-archive: {report.dest} ({report.exchanges} exchanges,"
            f" {report.messages} messages, {report.size_bytes} bytes)\n"
        )
        return ExitCode.OK
