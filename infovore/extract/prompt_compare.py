import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from infovore.db.exchanges import get_exchange
from infovore.extract.claim_shape import shape_counts
from infovore.extract.protocol import ClaimExtractor
from infovore.extract.request import build_request

DEFAULT_COMPARE_LIMIT = 40


@dataclass(frozen=True)
class ArmResult:
    version: str
    exchanges: int
    failures: int
    claims: int
    input_tokens: int
    output_tokens: int
    claim_chars: int
    source_chars: int
    person_subject: int

    @property
    def claims_per_exchange(self) -> float:
        return self.claims / self.exchanges if self.exchanges else 0.0

    @property
    def compression(self) -> float:
        """Claim payload over the source text it came from. v5 measures 1.075
        across the whole corpus: an extraction stage emitting MORE than it
        reads has distilled nothing (issue #165)."""
        return self.claim_chars / self.source_chars if self.source_chars else 0.0

    @property
    def person_subject_share(self) -> float:
        return self.person_subject / self.claims if self.claims else 0.0

    @property
    def tokens_per_claim(self) -> float:
        total = self.input_tokens + self.output_tokens
        return total / self.claims if self.claims else 0.0


@dataclass(frozen=True)
class PromptComparison:
    exchange_ids: tuple[int, ...]
    arms: tuple[ArmResult, ...]


SIZE_BUCKETS = ((1, 2), (3, 5), (6, 15), (16, 49), (50, 10**9))


def _bucket(message_count: int) -> tuple[int, int]:
    for low, high in SIZE_BUCKETS:
        if low <= message_count <= high:
            return (low, high)
    return SIZE_BUCKETS[-1]  # pragma: no cover - the last bucket is unbounded


def queue_size_mix(conn: sqlite3.Connection) -> dict[tuple[int, int], int]:
    """How the exchanges still waiting to be extracted are distributed by size.
    A comparison drawn from the ALREADY-extracted population measures the wrong
    thing: 90% of those are 16-49 message exchanges, while 76% of the queue is
    15 messages or fewer."""
    rows = conn.execute(
        "SELECT message_count FROM exchanges WHERE extraction_status = 'pending'"
    ).fetchall()
    mix: dict[tuple[int, int], int] = {bucket: 0 for bucket in SIZE_BUCKETS}
    for row in rows:
        mix[_bucket(row["message_count"])] += 1
    return mix


def sample_extracted_exchanges(conn: sqlite3.Connection, limit: int, seed: int = 0) -> list[int]:
    """Already-extracted live exchanges, stratified to match the size mix of
    the queue they stand in for, and deterministic for a seed so the set is
    stable: a comparison whose sample moves cannot be re-run."""
    available: dict[tuple[int, int], list[int]] = {bucket: [] for bucket in SIZE_BUCKETS}
    for row in conn.execute(
        "SELECT DISTINCT e.id AS id, e.message_count AS message_count FROM exchanges e"
        " JOIN extraction_runs r ON r.exchange_id = e.id"
        " WHERE r.mode = 'live' AND r.outcome = 'ok' AND e.extraction_status = 'done'"
        " ORDER BY e.id"
    ).fetchall():
        available[_bucket(row["message_count"])].append(row["id"])

    mix = queue_size_mix(conn)
    total = sum(mix.values())
    wanted = {bucket: (limit * count // total if total else 0) for bucket, count in mix.items()}
    # Largest remainder, so the parts sum to `limit` instead of losing slots.
    while sum(min(wanted[b], len(available[b])) for b in SIZE_BUCKETS) < limit:
        room = [b for b in SIZE_BUCKETS if wanted[b] < len(available[b])]
        if not room:
            break
        best = max(room, key=lambda b: (limit * mix[b] / total if total else 0) - wanted[b])
        wanted[best] += 1

    rng = random.Random(seed)
    picked: list[int] = []
    for bucket in SIZE_BUCKETS:
        pool = available[bucket]
        take = min(wanted[bucket], len(pool))
        picked.extend(pool if take >= len(pool) else rng.sample(pool, take))
    return sorted(picked)


class ArmLabelMismatchError(ValueError):
    pass


async def run_arm(
    conn: sqlite3.Connection,
    extractor: ClaimExtractor,
    version: str,
    exchange_ids: list[int],
    dump: Path | None = None,
) -> ArmResult:
    """Extract the same exchanges under one prompt version. Writes nothing: a
    comparison that mutates `claims` destroys the baseline it is measured
    against and cannot be run twice. `dump` writes one JSONL record per
    exchange, including the ones that yielded nothing, because the aggregate
    is not the evidence: without the claims themselves every follow-up
    question costs another full run, which is how the coverage check on v6
    was lost after being identified as the decisive one.

    The label is checked against the extractor's own prompt version. An
    experiment whose arms are labelled by hand can report two differently
    named arms that ran the same configuration, which is how RULE #215's
    three-hop comparison described a run that never happened."""
    actual = getattr(extractor, "prompt_version", None)
    if actual is not None and actual != version:
        raise ArmLabelMismatchError(f"arm labelled {version} but the extractor renders {actual}")
    failures = claims = input_tokens = output_tokens = 0
    claim_chars = source_chars = person_subject = 0
    records: list[dict[str, object]] = []
    for exchange_id in exchange_ids:
        exchange = get_exchange(conn, exchange_id)
        if exchange is None:  # pragma: no cover - ids come from the same table
            continue
        request = build_request(conn, exchange)
        source_chars += sum(len(message.content) for message in request.messages)
        outcome = await extractor.extract(request)
        input_tokens += outcome.input_tokens or 0
        output_tokens += outcome.output_tokens or 0
        if outcome.failure is not None:
            failures += 1
            continue
        claims += len(outcome.claims)
        statements = [claim.statement for claim in outcome.claims]
        claim_chars += sum(
            len(claim.statement) + len(claim.subject) + len(claim.probe_question)
            for claim in outcome.claims
        )
        person_subject += shape_counts(statements).person_subject
        records.append(
            {
                "exchange_id": exchange_id,
                "version": version,
                "claims": [
                    {
                        "statement": claim.statement,
                        "subject": claim.subject,
                        "kind": claim.kind.value,
                        "confidence": claim.confidence,
                        "sources": list(claim.source_message_ids),
                    }
                    for claim in outcome.claims
                ],
            }
        )
    if dump is not None:
        dump.write_text("".join(f"{json.dumps(record)}\n" for record in records))
    return ArmResult(
        version=version,
        exchanges=len(exchange_ids),
        failures=failures,
        claims=claims,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        claim_chars=claim_chars,
        source_chars=source_chars,
        person_subject=person_subject,
    )


def format_comparison(comparison: PromptComparison) -> list[str]:
    lines = [
        f"prompt comparison over {len(comparison.exchange_ids)} already-extracted exchanges",
        f"  {'version':<9} {'claims':>7} {'per ex':>7} {'compress':>9}"
        f" {'person%':>8} {'tok/claim':>10} {'failed':>7}",
    ]
    for arm in comparison.arms:
        lines.append(
            f"  {arm.version:<9} {arm.claims:>7} {arm.claims_per_exchange:>7.2f}"
            f" {arm.compression:>9.3f} {100 * arm.person_subject_share:>7.1f}%"
            f" {arm.tokens_per_claim:>10.0f} {arm.failures:>7}"
        )
    return lines
