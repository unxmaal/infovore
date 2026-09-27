import argparse
import html
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from infovore.cli import ExitCode
from infovore.config import ConfigError
from infovore.db.claims import (
    UnknownPromptVersionError,
    claim_source_ids,
    claims_for_run,
    live_prompt_version,
    promote_prompt_version,
    register_prompt_version,
)
from infovore.db.exchanges import exchange_message_ids, get_exchange
from infovore.db.labels import effective_labels
from infovore.db.raw import get_channel, messages_by_ids
from infovore.db.run_selection import (
    InvalidRunSelectorError,
    NoTrialBatchError,
    resolve_run_selector,
)
from infovore.extract.prompt import PROMPT_SHA256, PROMPT_VERSION, permalink
from infovore.rows import ClaimKind, ClaimRow, Label, Novelty, RunOutcome

if TYPE_CHECKING:
    from datetime import datetime

    from infovore.cli import AppContext


@dataclass(frozen=True)
class VersionKey:
    prompt_version: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.prompt_version} / {self.model}"


class UnknownRunIdsError(Exception):
    def __init__(self, run_ids: Sequence[int]) -> None:
        self.run_ids = tuple(run_ids)
        joined = ", ".join(str(run_id) for run_id in self.run_ids)
        super().__init__(f"unknown run ids: {joined}")


@dataclass(frozen=True)
class ReviewMessage:
    id: int
    author: str
    created_at: "datetime"
    content: str


@dataclass(frozen=True)
class ReviewClaim:
    id: int
    kind: ClaimKind
    subject: str
    statement: str
    confidence: float
    source_message_ids: tuple[int, ...]
    novelty: Novelty
    probe_answer: str | None


@dataclass(frozen=True)
class ReviewRun:
    run_id: int
    model: str
    outcome: RunOutcome
    error: str | None
    input_tokens: int | None
    output_tokens: int | None
    claims: tuple[ReviewClaim, ...]


@dataclass(frozen=True)
class VerdictShift:
    subject: str
    statement: str
    before: Novelty
    after: Novelty


@dataclass(frozen=True)
class ExchangeDiff:
    added: tuple[ReviewClaim, ...]
    dropped: tuple[ReviewClaim, ...]
    changed: tuple[tuple[ReviewClaim, ReviewClaim], ...]
    verdict_shifts: tuple[VerdictShift, ...]


@dataclass(frozen=True)
class ExchangeReview:
    exchange_id: int
    channel_name: str
    permalink: str
    messages: tuple[ReviewMessage, ...]
    runs: dict[VersionKey, ReviewRun]
    diff: ExchangeDiff | None
    triage_score: float | None
    p_lore: float | None
    triage_reasons: tuple[tuple[str, float], ...]
    effective_label: Label | None


@dataclass(frozen=True)
class VersionSummary:
    version: VersionKey
    exchanges: int
    claims: int
    claims_per_exchange: float
    verdict_distribution: dict[Novelty, int]
    known_share: float
    failed_runs: int
    input_tokens_per_100_exchanges: float
    output_tokens_per_100_exchanges: float


@dataclass(frozen=True)
class Review:
    versions: tuple[VersionKey, ...]
    exchanges: tuple[ExchangeReview, ...]
    summaries: dict[VersionKey, VersionSummary]


@dataclass(frozen=True)
class _RunRow:
    id: int
    exchange_id: int
    model: str
    prompt_version: str
    outcome: RunOutcome
    error: str | None
    input_tokens: int | None
    output_tokens: int | None


def _fetch_runs(conn: sqlite3.Connection, run_ids: Sequence[int]) -> dict[int, _RunRow]:
    placeholders = ",".join("?" for _ in run_ids)
    rows = conn.execute(
        "SELECT id, exchange_id, model, prompt_version, outcome, error, input_tokens,"
        f" output_tokens FROM extraction_runs WHERE id IN ({placeholders})",
        tuple(run_ids),
    ).fetchall()
    return {
        row["id"]: _RunRow(
            id=row["id"],
            exchange_id=row["exchange_id"],
            model=row["model"],
            prompt_version=row["prompt_version"],
            outcome=RunOutcome(row["outcome"]),
            error=row["error"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
        )
        for row in rows
    }


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _review_claim(conn: sqlite3.Connection, claim: ClaimRow) -> ReviewClaim:
    assert claim.id is not None
    return ReviewClaim(
        id=claim.id,
        kind=claim.kind,
        subject=claim.subject,
        statement=claim.statement,
        confidence=claim.confidence,
        source_message_ids=tuple(claim_source_ids(conn, claim.id)),
        novelty=claim.novelty,
        probe_answer=claim.probe_answer,
    )


def _diff_claims(before: Sequence[ReviewClaim], after: Sequence[ReviewClaim]) -> ExchangeDiff:
    matched_after_ids: set[int] = set()
    exact_matches: list[tuple[ReviewClaim, ReviewClaim]] = []
    still_before: list[ReviewClaim] = []

    for candidate in before:
        match = next(
            (
                other
                for other in after
                if other.id not in matched_after_ids
                and _normalize(other.subject) == _normalize(candidate.subject)
                and _normalize(other.statement) == _normalize(candidate.statement)
            ),
            None,
        )
        if match is None:
            still_before.append(candidate)
        else:
            matched_after_ids.add(match.id)
            exact_matches.append((candidate, match))

    changed: list[tuple[ReviewClaim, ReviewClaim]] = []
    dropped: list[ReviewClaim] = []

    for candidate in still_before:
        match = next(
            (
                other
                for other in after
                if other.id not in matched_after_ids
                and _normalize(other.subject) == _normalize(candidate.subject)
            ),
            None,
        )
        if match is None:
            dropped.append(candidate)
        else:
            matched_after_ids.add(match.id)
            changed.append((candidate, match))

    added = [candidate for candidate in after if candidate.id not in matched_after_ids]

    verdict_shifts = [
        VerdictShift(
            subject=candidate.subject,
            statement=candidate.statement,
            before=candidate.novelty,
            after=match.novelty,
        )
        for candidate, match in (*exact_matches, *changed)
        if candidate.novelty is not match.novelty
    ]

    return ExchangeDiff(
        added=tuple(added),
        dropped=tuple(dropped),
        changed=tuple(changed),
        verdict_shifts=tuple(verdict_shifts),
    )


def _build_exchange_review(
    conn: sqlite3.Connection,
    exchange_id: int,
    versions: Sequence[VersionKey],
    runs_by_version_and_exchange: dict[VersionKey, dict[int, _RunRow]],
    labels: dict[int, Label],
) -> ExchangeReview:
    exchange = get_exchange(conn, exchange_id)
    assert exchange is not None
    channel = get_channel(conn, exchange.channel_id)
    channel_name = channel.name if channel is not None else str(exchange.channel_id)

    message_ids = exchange_message_ids(conn, exchange_id)
    messages = messages_by_ids(conn, message_ids)
    review_messages = tuple(
        ReviewMessage(
            id=message.id,
            author=message.author_name_at_time,
            created_at=message.created_at,
            content=message.content,
        )
        for message in messages
    )
    guild_id = messages[0].guild_id
    link = permalink(guild_id, exchange.channel_id, exchange.first_message_id)

    runs: dict[VersionKey, ReviewRun] = {}
    for version in versions:
        run_row = runs_by_version_and_exchange.get(version, {}).get(exchange_id)
        if run_row is None:
            continue
        claims = tuple(_review_claim(conn, claim) for claim in claims_for_run(conn, run_row.id))
        runs[version] = ReviewRun(
            run_id=run_row.id,
            model=run_row.model,
            outcome=run_row.outcome,
            error=run_row.error,
            input_tokens=run_row.input_tokens,
            output_tokens=run_row.output_tokens,
            claims=claims,
        )

    present_versions = [version for version in versions if version in runs]
    diff = None
    if len(present_versions) == 2:
        diff = _diff_claims(runs[present_versions[0]].claims, runs[present_versions[1]].claims)

    reasons = tuple((name, weight) for name, weight in json.loads(exchange.triage_reasons or "[]"))

    return ExchangeReview(
        exchange_id=exchange_id,
        channel_name=channel_name,
        permalink=link,
        messages=review_messages,
        runs=runs,
        diff=diff,
        triage_score=exchange.triage_score,
        p_lore=exchange.p_lore,
        triage_reasons=reasons,
        effective_label=labels.get(exchange_id),
    )


def _summarize_version(
    version: VersionKey, exchange_reviews: Sequence[ExchangeReview]
) -> VersionSummary:
    runs = [exchange.runs[version] for exchange in exchange_reviews if version in exchange.runs]
    exchanges = len(runs)
    claims = [claim for run in runs for claim in run.claims]
    claim_count = len(claims)
    claims_per_exchange = claim_count / exchanges if exchanges else 0.0

    verdict_distribution: dict[Novelty, int] = {}
    for claim in claims:
        verdict_distribution[claim.novelty] = verdict_distribution.get(claim.novelty, 0) + 1
    known_share = verdict_distribution.get(Novelty.KNOWN, 0) / claim_count if claim_count else 0.0

    failed_runs = sum(1 for run in runs if run.outcome is RunOutcome.FAILED)
    input_tokens_total = sum(run.input_tokens or 0 for run in runs)
    output_tokens_total = sum(run.output_tokens or 0 for run in runs)
    scale = 100 / exchanges if exchanges else 0.0

    return VersionSummary(
        version=version,
        exchanges=exchanges,
        claims=claim_count,
        claims_per_exchange=claims_per_exchange,
        verdict_distribution=verdict_distribution,
        known_share=known_share,
        failed_runs=failed_runs,
        input_tokens_per_100_exchanges=input_tokens_total * scale,
        output_tokens_per_100_exchanges=output_tokens_total * scale,
    )


def build_review(conn: sqlite3.Connection, run_ids: Sequence[int]) -> Review:
    unique_ids = list(dict.fromkeys(run_ids))
    if not unique_ids:
        raise ValueError("run_ids must not be empty")
    runs_by_id = _fetch_runs(conn, unique_ids)
    missing = [run_id for run_id in unique_ids if run_id not in runs_by_id]
    if missing:
        raise UnknownRunIdsError(missing)

    versions: list[VersionKey] = []
    runs_by_version_and_exchange: dict[VersionKey, dict[int, _RunRow]] = {}
    for run_id in unique_ids:
        run_row = runs_by_id[run_id]
        key = VersionKey(run_row.prompt_version, run_row.model)
        by_exchange = runs_by_version_and_exchange.setdefault(key, {})
        if key not in versions:
            versions.append(key)
        by_exchange[run_row.exchange_id] = run_row

    labels = effective_labels(conn)
    exchange_ids = sorted({run_row.exchange_id for run_row in runs_by_id.values()})
    exchange_reviews = [
        _build_exchange_review(conn, exchange_id, versions, runs_by_version_and_exchange, labels)
        for exchange_id in exchange_ids
    ]

    summaries = {version: _summarize_version(version, exchange_reviews) for version in versions}

    return Review(versions=tuple(versions), exchanges=tuple(exchange_reviews), summaries=summaries)


_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #f7f7f8;
  --fg: #1b1b1f;
  --panel: #ffffff;
  --border: #d8d8dd;
  --muted: #6b6b76;
  --known: #6b7280;
  --unknown: #2563eb;
  --partial: #ca8a04;
  --contradicts: #dc2626;
  --unprobed: #9ca3af;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #121214;
    --fg: #e8e8ec;
    --panel: #1c1c20;
    --border: #33333a;
    --muted: #a0a0ab;
  }
}
body {
  background: var(--bg);
  color: var(--fg);
  font-family: system-ui, sans-serif;
  margin: 0;
  padding: 1.5rem;
}
table.summary {
  border-collapse: collapse;
  margin-bottom: 1.5rem;
}
table.summary th, table.summary td {
  border: 1px solid var(--border);
  padding: 0.35rem 0.6rem;
  text-align: left;
}
details.exchange {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 0.4rem;
  margin-bottom: 0.75rem;
  padding: 0.5rem 0.75rem;
}
.exchange-body {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-top: 0.5rem;
}
.messages .message {
  border-left: 3px solid var(--border);
  margin-bottom: 0.4rem;
  padding-left: 0.5rem;
}
.message-meta, .run-meta {
  color: var(--muted);
  font-size: 0.85em;
}
.versions {
  display: grid;
  gap: 0.75rem;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}
.version-column {
  border: 1px solid var(--border);
  border-radius: 0.3rem;
  padding: 0.5rem;
}
.claim {
  border-top: 1px solid var(--border);
  padding: 0.35rem 0;
}
.badge {
  border-radius: 0.75rem;
  color: white;
  font-size: 0.75em;
  padding: 0.1rem 0.5rem;
}
.badge-known { background: var(--known); }
.badge-unknown { background: var(--unknown); }
.badge-partial { background: var(--partial); }
.badge-contradicts { background: var(--contradicts); }
.badge-unprobed { background: var(--unprobed); }
.diff {
  border-top: 1px dashed var(--border);
  padding-top: 0.5rem;
}
.changed-pair {
  align-items: center;
  display: flex;
  gap: 0.5rem;
}
.missing {
  color: var(--muted);
  font-style: italic;
}
"""


def _badge(novelty: Novelty) -> str:
    return f'<span class="badge badge-{novelty.value}">{html.escape(novelty.value)}</span>'


def _render_messages(messages: Sequence[ReviewMessage]) -> str:
    return "\n".join(
        '<div class="message">'
        f'<div class="message-meta">#{message.id} · {html.escape(message.author)} · '
        f"{html.escape(message.created_at.isoformat())}</div>"
        f'<div class="message-content">{html.escape(message.content)}</div>'
        "</div>"
        for message in messages
    )


def _render_claim(claim: ReviewClaim) -> str:
    sources = ", ".join(f"#{message_id}" for message_id in claim.source_message_ids)
    probe = html.escape(claim.probe_answer) if claim.probe_answer is not None else "—"
    return (
        '<div class="claim">'
        f'<div class="claim-head">{_badge(claim.novelty)} '
        f'<span class="kind">{html.escape(claim.kind.value)}</span> '
        f"<strong>{html.escape(claim.subject)}</strong></div>"
        f'<div class="claim-statement">{html.escape(claim.statement)}</div>'
        f'<div class="claim-meta">confidence {claim.confidence:.2f} · sources {sources}</div>'
        f'<div class="claim-probe">probe answer: {probe}</div>'
        "</div>"
    )


def _render_run_column(version: VersionKey, run: ReviewRun | None) -> str:
    label = html.escape(version.label)
    if run is None:
        return f'<div class="version-column"><h4>{label}</h4><p class="missing">no run</p></div>'
    input_tokens = run.input_tokens if run.input_tokens is not None else "—"
    output_tokens = run.output_tokens if run.output_tokens is not None else "—"
    error_html = (
        f'<div class="run-error">error: {html.escape(run.error)}</div>' if run.error else ""
    )
    claims_html = (
        "\n".join(_render_claim(claim) for claim in run.claims)
        if run.claims
        else '<p class="missing">no claims</p>'
    )
    return (
        '<div class="version-column">'
        f"<h4>{label}</h4>"
        f'<div class="run-meta">run #{run.run_id} · {html.escape(run.outcome.value)} · '
        f"{html.escape(run.model)} · tokens {input_tokens} in / {output_tokens} out</div>"
        f"{error_html}{claims_html}"
        "</div>"
    )


def _render_diff(diff: ExchangeDiff | None) -> str:
    if diff is None:
        return ""
    sections: list[str] = []
    if diff.added:
        sections.append("<h4>Added</h4>" + "\n".join(_render_claim(claim) for claim in diff.added))
    if diff.dropped:
        sections.append(
            "<h4>Dropped</h4>" + "\n".join(_render_claim(claim) for claim in diff.dropped)
        )
    if diff.changed:
        rows = "\n".join(
            '<div class="changed-pair">'
            f"{_render_claim(before)}<span>→</span>{_render_claim(after)}"
            "</div>"
            for before, after in diff.changed
        )
        sections.append("<h4>Changed</h4>" + rows)
    if diff.verdict_shifts:
        items = "\n".join(
            f"<li>{html.escape(shift.subject)}: {html.escape(shift.statement)} "
            f"{_badge(shift.before)} → {_badge(shift.after)}</li>"
            for shift in diff.verdict_shifts
        )
        sections.append(f"<h4>Verdict shifts</h4><ul>{items}</ul>")
    if not sections:
        return '<div class="diff"><p class="missing">no differences</p></div>'
    return '<div class="diff">' + "\n".join(sections) + "</div>"


def _render_exchange_meta(exchange: ExchangeReview) -> str:
    score = f"{exchange.triage_score:.3f}" if exchange.triage_score is not None else "—"
    p_lore = f"{exchange.p_lore:.3f}" if exchange.p_lore is not None else "—"
    label = exchange.effective_label.value if exchange.effective_label is not None else "—"
    reasons = ", ".join(f"{name}={weight:+.2f}" for name, weight in exchange.triage_reasons) or "—"
    return (
        '<div class="exchange-meta">'
        f"rule score: {score} &middot; p_lore: {p_lore} &middot; label: {html.escape(label)}"
        f" &middot; reasons: {html.escape(reasons)}"
        "</div>"
    )


def _render_exchange(exchange: ExchangeReview, versions: Sequence[VersionKey]) -> str:
    columns = "\n".join(
        _render_run_column(version, exchange.runs.get(version)) for version in versions
    )
    return (
        '<details class="exchange">'
        f"<summary>Exchange #{exchange.exchange_id} · {html.escape(exchange.channel_name)} "
        f'· <a href="{html.escape(exchange.permalink)}">permalink</a></summary>'
        '<div class="exchange-body">'
        f"{_render_exchange_meta(exchange)}"
        f'<div class="messages">{_render_messages(exchange.messages)}</div>'
        f'<div class="versions">{columns}</div>'
        f"{_render_diff(exchange.diff)}"
        "</div>"
        "</details>"
    )


def _format_summary_cell(summary: VersionSummary, metric: str) -> str:
    if metric == "exchanges":
        return str(summary.exchanges)
    if metric == "claims":
        return str(summary.claims)
    if metric == "claims_per_exchange":
        return f"{summary.claims_per_exchange:.2f}"
    if metric == "known_share":
        return f"{summary.known_share:.1%}"
    if metric == "failed_runs":
        return str(summary.failed_runs)
    if metric == "input_tokens_per_100":
        return f"{summary.input_tokens_per_100_exchanges:.0f}"
    return f"{summary.output_tokens_per_100_exchanges:.0f}"


_SUMMARY_METRICS = (
    ("exchanges", "exchanges"),
    ("claims", "claims"),
    ("claims_per_exchange", "claims / exchange"),
    ("known_share", "share known"),
    ("failed_runs", "failed runs"),
    ("input_tokens_per_100", "input tokens / 100 exchanges"),
    ("output_tokens_per_100", "output tokens / 100 exchanges"),
)


def _render_summary_table(review: Review) -> str:
    header = "".join(f"<th>{html.escape(version.label)}</th>" for version in review.versions)
    rows = ["<tr><th>metric</th>" + header + "</tr>"]
    for metric, label in _SUMMARY_METRICS:
        cells = "".join(
            f"<td>{_format_summary_cell(review.summaries[version], metric)}</td>"
            for version in review.versions
        )
        rows.append(f"<tr><th>{html.escape(label)}</th>{cells}</tr>")

    all_verdicts = sorted(
        {
            verdict
            for summary in review.summaries.values()
            for verdict in summary.verdict_distribution
        },
        key=lambda verdict: verdict.value,
    )
    for verdict in all_verdicts:
        cells = "".join(
            f"<td>{review.summaries[version].verdict_distribution.get(verdict, 0)}</td>"
            for version in review.versions
        )
        rows.append(f"<tr><th>verdict: {html.escape(verdict.value)}</th>{cells}</tr>")

    return (
        '<table class="summary"><thead>'
        + rows[0]
        + "</thead><tbody>"
        + "".join(rows[1:])
        + "</tbody></table>"
    )


def render_review_html(review: Review) -> str:
    body = [_render_summary_table(review)]
    body.extend(_render_exchange(exchange, review.versions) for exchange in review.exchanges)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Extraction review</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{chr(10).join(body)}\n"
        "</body>\n"
        "</html>\n"
    )


class ReviewCommand:
    name = "review"
    help = "render a self-contained HTML review report for one or two prompt-version run sets"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--run-ids", nargs="*", default=None, dest="run_ids", metavar="RUN_ID_OR_RANGE"
        )
        parser.add_argument("--out", type=Path, default=None)

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        try:
            run_ids = resolve_run_selector(context.conn, args.run_ids)
        except NoTrialBatchError as error:
            raise ConfigError(
                "no trial batch found; pass --run-ids explicitly, or run"
                " `infovore extract --mode trial` first"
            ) from error
        except InvalidRunSelectorError as error:
            raise ConfigError(str(error)) from error

        try:
            review = build_review(context.conn, run_ids)
        except UnknownRunIdsError as error:
            raise ConfigError(str(error)) from error

        out_path = (
            args.out if args.out is not None else context.settings.scratch_dir / "review.html"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_review_html(review), encoding="utf-8")
        context.stdout.write(f"{out_path}\n")
        return ExitCode.OK


class PromoteCommand:
    name = "promote"
    help = "promote a prompt version to live"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--prompt-version", required=True, dest="prompt_version", metavar="V")

    async def run(self, context: "AppContext", args: argparse.Namespace) -> int:
        version = args.prompt_version
        if version == PROMPT_VERSION:
            register_prompt_version(
                context.conn, PROMPT_VERSION, PROMPT_SHA256, context.clock.now()
            )
        try:
            promote_prompt_version(context.conn, version, context.clock.now())
        except UnknownPromptVersionError as error:
            raise ConfigError(f"unknown prompt version: {version}") from error

        context.stdout.write(f"live prompt version: {live_prompt_version(context.conn)}\n")
        return ExitCode.OK
