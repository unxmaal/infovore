import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.claims.extract import ClaimsReplyError, Transport, fetch_model_id, http_get, http_post
from infovore.config import ConfigError
from infovore.db.wiki_articles import create_article_run, record_section, written_sections
from infovore.triage.lexicon import Lexicon, load_lexicon
from infovore.wiki import writer
from infovore.wiki.build import WikiClaim, _by_topic, load_claims, sections_of
from infovore.wiki.groups import group_claims
from infovore.wiki.support import Dropped, keep_supported

if TYPE_CHECKING:
    from infovore.cli import AppContext

MAX_CONCURRENCY = 32
post_for: Callable[[str], Transport] = http_post
get = http_get


@dataclass(frozen=True)
class Unit:
    topic: str
    section: str
    leads: tuple[WikiClaim, ...]


Written = tuple[list[tuple[str, list[int]]], list[Dropped]]


def _write_unit(post: Transport, model: str, lexicon: Lexicon, unit: Unit) -> Written:
    statements = [lead.statement for lead in unit.leads]
    body, _ = post(writer.build_request(model, unit.topic, unit.section, statements))
    sentences = writer.parse_reply(body, len(statements))
    kept, dropped = keep_supported(sentences, statements, lexicon)
    return [(text, [unit.leads[i].claim_id for i in cited]) for text, cited in kept], dropped


def _log(path: Path | None, unit: Unit, rows: list[dict[str, object]]) -> None:
    if path is not None:
        with path.open("a") as handle:
            for row in rows:
                handle.write(
                    json.dumps({"topic": unit.topic, "section": unit.section, **row}) + "\n"
                )


def _units(
    claims: Sequence[WikiClaim], min_claims: int, min_section: int, limit: int | None
) -> list[Unit]:
    by_topic = {n: v for n, v in _by_topic(claims).items() if len(v) >= min_claims}
    names = sorted(by_topic, key=lambda n: (-len(by_topic[n]), n))[:limit]
    return [
        Unit(
            name,
            section,
            tuple(g.lead for g in group_claims(members)[: writer.MAX_GROUPS]),
        )
        for name in names
        for section, members in sections_of(name, by_topic[name], min_section).items()
    ]


def run_write(context: "AppContext", args: argparse.Namespace, runs: list[int] | None) -> int:
    from infovore.claims.command import StopFlag, _stop_on_signals

    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        raise ConfigError("--concurrency must be between 1 and 32")
    conn, out = context.conn, context.stdout
    if conn.execute("SELECT 1 FROM tag_runs WHERE id = ?", (args.tag_run,)).fetchone() is None:
        raise ConfigError(f"unknown tag run {args.tag_run}")
    claims, _ = load_claims(
        conn,
        tech_only=args.claim_gate,
        salt=context.settings.pseudonym_salt,
        runs=runs,
        tag_runs=[args.tag_run],
    )
    units = _units(claims, args.min_claims, args.min_section, args.limit)
    if not args.write:
        if units:
            first = units[0]
            request = writer.build_request(
                args.model, first.topic, first.section, [lead.statement for lead in first.leads]
            )
            out.write(request["messages"][1]["content"] + "\n")
        else:
            out.write("nothing to do\n")
        return 0
    model_id, _ = fetch_model_id(args.endpoint, args.model, get)
    if args.resume:
        done = written_sections(conn, model_id, writer.prompt_hash(), args.tag_run)
        units = [u for u in units if (u.topic, u.section) not in done]
    if not units:
        out.write("nothing to do\n")
        return 0
    article_run_id = create_article_run(
        conn,
        endpoint=args.endpoint,
        model_alias=args.model,
        model_id=model_id,
        prompt_hash=writer.prompt_hash(),
        tag_run_id=args.tag_run,
        now=context.clock.now(),
    )
    conn.commit()
    post = post_for(args.endpoint)
    lexicon = load_lexicon(conn)
    attempted = failed = sentence_count = dropped_count = 0
    started = time.monotonic()
    stop = StopFlag()
    with _stop_on_signals(stop), ThreadPoolExecutor(args.concurrency) as pool:
        pending: dict[Future[Written], int] = {}
        queue = iter(enumerate(units))
        exhausted = False
        while True:
            while not exhausted and not stop.requested and len(pending) < args.concurrency:
                item = next(queue, None)
                if item is None:
                    exhausted = True
                    break
                pending[pool.submit(_write_unit, post, args.model, lexicon, item[1])] = item[0]
            if not pending:
                break
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(ready, key=pending.__getitem__):
                unit = units[pending.pop(future)]
                attempted += 1
                try:
                    sentences, dropped = future.result()
                except ClaimsReplyError as error:
                    failed += 1
                    sys.stderr.write(f"failed: {unit.topic} / {unit.section}: {error}\n")
                    _log(args.log, unit, [{"error": str(error)}])
                    continue
                record_section(
                    conn, article_run_id, unit.topic, unit.section, sentences, len(dropped)
                )
                conn.commit()
                _log(
                    args.log,
                    unit,
                    [{"text": d.text, "cited": d.cited, "reason": d.reason} for d in dropped],
                )
                sentence_count += len(sentences)
                dropped_count += len(dropped)
                if attempted % args.progress_every == 0:
                    sys.stderr.write(f"progress: {attempted}/{len(units)} sections\n")
    seconds = time.monotonic() - started
    out.write(
        f"sections={attempted} failed={failed} sentences={sentence_count}"
        f" dropped={dropped_count} seconds={seconds:.2f}\n"
    )
    out.write(f"article run {article_run_id} written\n")
    if stop.requested:
        out.write(f"interrupted after {attempted} sections; rerun with --resume\n")
        return 130
    return 0
