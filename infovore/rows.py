from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class ChannelKind(StrEnum):
    TEXT = "text"
    THREAD = "thread"


class GroupingRule(StrEnum):
    THREAD = "thread"
    REPLY_CHAIN = "reply_chain"
    QUIET_GAP = "quiet_gap"


class ExtractionStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"
    STALE = "stale"


class RunMode(StrEnum):
    TRIAL = "trial"
    LIVE = "live"


class RunOutcome(StrEnum):
    OK = "ok"
    FAILED = "failed"


class ClaimKind(StrEnum):
    FACT = "fact"
    CORRECTION = "correction"
    PROCEDURE = "procedure"
    REFERENCE = "reference"


class Novelty(StrEnum):
    UNPROBED = "unprobed"
    UNKNOWN = "unknown"
    PARTIAL = "partial"
    CONTRADICTS = "contradicts"
    KNOWN = "known"


class Label(StrEnum):
    LORE = "lore"
    NOISE = "noise"


class LabelSource(StrEnum):
    LLM = "llm"
    HUMAN = "human"


class MessageLabel(StrEnum):
    TRASH = "trash"
    KEEP = "keep"


class MessageLabelSource(StrEnum):
    HUMAN = "human"
    CITATION = "citation"
    RULE = "rule"


@dataclass(frozen=True)
class ChannelRow:
    id: int
    guild_id: int
    parent_id: int | None
    name: str
    kind: ChannelKind
    archived: bool
    last_backfilled_message_id: int | None


@dataclass(frozen=True)
class MessageRow:
    id: int
    channel_id: int
    guild_id: int
    author_id: int
    author_name_at_time: str
    author_is_bot: bool
    created_at: datetime
    edited_at: datetime | None
    content: str
    reply_to_id: int | None
    thread_id: int | None
    deleted_at: datetime | None
    ingested_at: datetime
    raw_json: str


@dataclass(frozen=True)
class MessageRevisionRow:
    message_id: int
    revision: int
    content: str
    edited_at: datetime | None
    raw_json: str


@dataclass(frozen=True)
class AttachmentRow:
    id: int
    message_id: int
    filename: str
    content_type: str | None
    size: int
    url: str
    sha256: str | None
    local_path: str | None


@dataclass(frozen=True)
class ReactionRow:
    message_id: int
    emoji: str
    count: int


@dataclass(frozen=True)
class ExchangeRow:
    id: int | None
    channel_id: int
    thread_id: int | None
    first_message_id: int
    last_message_id: int
    started_at: datetime
    ended_at: datetime
    message_count: int
    grouping_rule: GroupingRule
    content_hash: str
    parent_exchange_id: int | None
    extraction_status: ExtractionStatus
    retry_count: int
    last_error: str | None
    triage_score: float | None = None
    triage_reasons: str | None = None
    triage_version: str | None = None
    p_lore: float | None = None
    p_lore_model: int | None = None


@dataclass(frozen=True)
class ExtractionRunRow:
    id: int | None
    exchange_id: int
    model: str
    prompt_version: str
    started_at: datetime
    finished_at: datetime | None
    input_tokens: int | None
    output_tokens: int | None
    mode: RunMode
    outcome: RunOutcome
    error: str | None
    batch_id: str | None = None
    sampled_by: str | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class ProbeRunRow:
    """One recall+judge call pair, covering `claim_count` claims."""

    id: int | None
    probe_model: str | None
    judge_model: str | None
    started_at: datetime
    finished_at: datetime
    claim_count: int
    batched: bool
    recall_input_tokens: int | None
    recall_output_tokens: int | None
    judge_input_tokens: int | None
    judge_output_tokens: int | None
    cost_usd: float | None
    outcome: RunOutcome
    error: str | None


@dataclass(frozen=True)
class PromptVersionRow:
    version: str
    text_sha256: str
    created_at: datetime
    promoted_at: datetime | None


@dataclass(frozen=True)
class ClaimRow:
    id: int | None
    exchange_id: int
    extraction_run_id: int
    statement: str
    subject: str
    kind: ClaimKind
    confidence: float
    probe_question: str
    permalink: str
    supersedes_claim_id: int | None
    novelty: Novelty
    probe_model: str | None
    probe_answer: str | None
    probed_at: datetime | None
    probe_error: str | None
    retracted_at: datetime | None
    retraction_reason: str | None
    probe_run_id: int | None = None


@dataclass(frozen=True)
class ClaimSourceRow:
    claim_id: int
    message_id: int
