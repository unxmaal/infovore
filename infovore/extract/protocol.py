from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from infovore.rows import (
    AttachmentRow,
    ClaimKind,
    ClaimRow,
    ExchangeRow,
    MessageRow,
    Novelty,
    ReactionRow,
)


class FailureKind(StrEnum):
    TRANSIENT = "transient"
    USAGE_LIMIT = "usage_limit"
    FATAL = "fatal"
    INVALID_OUTPUT = "invalid_output"


@dataclass(frozen=True)
class Failure:
    kind: FailureKind
    message: str
    retry_after: float | None


@dataclass(frozen=True)
class ExtractionRequest:
    exchange: ExchangeRow
    channel_name: str
    messages: tuple[MessageRow, ...]
    context_messages: tuple[MessageRow, ...]
    attachments: tuple[AttachmentRow, ...]
    reactions: tuple[ReactionRow, ...]
    related_claims: tuple[ClaimRow, ...]
    opted_out_user_ids: frozenset[int]


@dataclass(frozen=True)
class ExtractedClaim:
    statement: str
    subject: str
    kind: ClaimKind
    confidence: float
    probe_question: str
    source_message_ids: tuple[int, ...]
    supersedes_claim_id: int | None


@dataclass(frozen=True)
class ExtractionOutcome:
    claims: tuple[ExtractedClaim, ...]
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    failure: Failure | None

    @property
    def succeeded(self) -> bool:
        return self.failure is None


@dataclass(frozen=True)
class ProbeOutcome:
    verdict: Novelty | None
    model: str | None
    answer: str | None
    failure: Failure | None

    @property
    def succeeded(self) -> bool:
        return self.failure is None


class ClaimExtractor(Protocol):
    async def extract(self, request: ExtractionRequest) -> ExtractionOutcome: ...


class NoveltyProbe(Protocol):
    async def probe(self, claim: ClaimRow) -> ProbeOutcome: ...
