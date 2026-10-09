import argparse
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

from infovore.claims.extract import ClaimsReplyError, Transport, fetch_model_id, http_get, http_post
from infovore.config import ConfigError
from infovore.db.claims_v2 import run_ids
from infovore.db.wiki_tags import create_tag_run, record_tags, tagged_claim_ids
from infovore.wiki import tagger

if TYPE_CHECKING:
    from infovore.cli import AppContext

MAX_CONCURRENCY = 32
post_for: Callable[[str], Transport] = http_post
get = http_get


def _run_ids(args: argparse.Namespace, known: Sequence[int]) -> list[int]:
    try:
        runs = sorted({int(p) for p in args.runs.split(",")})
    except ValueError as error:
        raise ConfigError("--runs must be comma-separated run ids") from error
    if unknown := sorted(set(runs) - set(known)):
        raise ConfigError(f"unknown run {unknown[0]}")
    return runs


def _tag_batch(
    post: Transport, model: str, batch: Sequence[tuple[int, str]]
) -> dict[int, list[str]]:
    body, _ = post(tagger.build_request(model, [statement for _, statement in batch]))
    return {
        batch[position][0]: tags for position, tags in tagger.parse_reply(body, len(batch)).items()
    }


def run_tag(context: "AppContext", args: argparse.Namespace) -> int:
    from infovore.claims.command import StopFlag, _stop_on_signals

    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        raise ConfigError("--concurrency must be between 1 and 32")
    conn, out = context.conn, context.stdout
    runs = _run_ids(args, run_ids(conn))
    marks = ",".join("?" * len(runs))
    claims = [
        (row[0], row[1])
        for row in conn.execute(
            f"SELECT id, statement FROM claims_v2 WHERE run_id IN ({marks}) ORDER BY id", runs
        )
    ]
    if not args.write:
        if claims:
            first = [statement for _, statement in claims[: tagger.BATCH]]
            out.write(tagger.build_request(args.model, first)["messages"][1]["content"] + "\n")
        else:
            out.write("nothing to do\n")
        return 0
    model_id, _ = fetch_model_id(args.endpoint, args.model, get)
    if args.resume:
        done = tagged_claim_ids(conn, model_id, tagger.prompt_hash())
        claims = [claim for claim in claims if claim[0] not in done]
    if args.limit is not None:
        claims = claims[: args.limit]
    if not claims:
        out.write("nothing to do\n")
        return 0
    tag_run_id = create_tag_run(
        conn,
        endpoint=args.endpoint,
        model_alias=args.model,
        model_id=model_id,
        prompt_hash=tagger.prompt_hash(),
        now=context.clock.now(),
    )
    conn.commit()
    size = tagger.BATCH
    batches = [claims[i : i + size] for i in range(0, len(claims), size)]
    post = post_for(args.endpoint)
    batch_count = failed = tag_count = finished = 0
    started = time.monotonic()
    stop = StopFlag()
    with _stop_on_signals(stop), ThreadPoolExecutor(args.concurrency) as pool:
        pending: dict[Future[dict[int, list[str]]], int] = {}
        queue = iter(enumerate(batches))
        exhausted = False
        while True:
            while not exhausted and not stop.requested and len(pending) < args.concurrency:
                item = next(queue, None)
                if item is None:
                    exhausted = True
                    break
                pending[pool.submit(_tag_batch, post, args.model, item[1])] = item[0]
            if not pending:
                break
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(ready, key=pending.__getitem__):
                size_done = len(batches[pending.pop(future)])
                batch_count += 1
                try:
                    tags = future.result()
                except ClaimsReplyError:
                    failed += 1
                    continue
                record_tags(conn, tag_run_id, tags)
                conn.commit()
                tag_count += sum(len(names) for names in tags.values())
                if (finished + size_done) // args.progress_every > finished // args.progress_every:
                    sys.stderr.write(f"progress: {finished + size_done}/{len(claims)} claims\n")
                finished += size_done
    seconds = time.monotonic() - started
    out.write(
        f"claims={len(claims)} batches={batch_count} failed={failed} tags={tag_count}"
        f" seconds={seconds:.2f}\n"
    )
    out.write(f"tag run {tag_run_id} written\n")
    if stop.requested:
        out.write(f"interrupted after {batch_count} batches; rerun with --resume\n")
        return 130
    return 0
