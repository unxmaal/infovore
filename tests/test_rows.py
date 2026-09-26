import dataclasses
from datetime import UTC, datetime

import pytest

from infovore.rows import (
    AttachmentRow,
    ChannelKind,
    ChannelRow,
    ClaimKind,
    ClaimRow,
    ClaimSourceRow,
    ExchangeRow,
    ExtractionRunRow,
    ExtractionStatus,
    GroupingRule,
    MessageRevisionRow,
    MessageRow,
    Novelty,
    PromptVersionRow,
    ReactionRow,
    RunMode,
    RunOutcome,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_message(**overrides: object) -> MessageRow:
    fields: dict[str, object] = {
        "id": 1,
        "channel_id": 10,
        "guild_id": 100,
        "author_id": 5,
        "author_name_at_time": "alice",
        "author_is_bot": False,
        "created_at": NOW,
        "edited_at": None,
        "content": "hello",
        "reply_to_id": None,
        "thread_id": None,
        "deleted_at": None,
        "ingested_at": NOW,
        "raw_json": "{}",
    }
    fields.update(overrides)
    return MessageRow(**fields)  # type: ignore[arg-type]


def test_enum_values_match_the_data_model() -> None:
    assert [k.value for k in ChannelKind] == ["text", "thread"]
    assert [r.value for r in GroupingRule] == ["thread", "reply_chain", "quiet_gap"]
    assert [s.value for s in ExtractionStatus] == ["pending", "done", "skipped", "failed", "stale"]
    assert [m.value for m in RunMode] == ["trial", "live"]
    assert [o.value for o in RunOutcome] == ["ok", "failed"]
    assert [k.value for k in ClaimKind] == ["fact", "correction", "procedure", "reference"]
    assert [n.value for n in Novelty] == ["unprobed", "unknown", "partial", "contradicts", "known"]


def test_rows_are_frozen() -> None:
    message = make_message()
    with pytest.raises(dataclasses.FrozenInstanceError):
        message.content = "changed"  # type: ignore[misc]


def test_every_row_type_constructs() -> None:
    rows: list[object] = [
        ChannelRow(1, 100, None, "general", ChannelKind.TEXT, False, None),
        make_message(),
        MessageRevisionRow(1, 0, "old", NOW, "{}"),
        AttachmentRow(1, 1, "a.pdf", "application/pdf", 10, "https://x", None, None),
        ReactionRow(1, "👍", 3),
        ExchangeRow(
            id=None,
            channel_id=10,
            thread_id=None,
            first_message_id=1,
            last_message_id=2,
            started_at=NOW,
            ended_at=NOW,
            message_count=2,
            grouping_rule=GroupingRule.QUIET_GAP,
            content_hash="h",
            parent_exchange_id=None,
            extraction_status=ExtractionStatus.PENDING,
            retry_count=0,
            last_error=None,
        ),
        ExtractionRunRow(
            id=None,
            exchange_id=1,
            model="m",
            prompt_version="v1",
            started_at=NOW,
            finished_at=None,
            input_tokens=None,
            output_tokens=None,
            mode=RunMode.TRIAL,
            outcome=RunOutcome.OK,
            error=None,
        ),
        PromptVersionRow("v1", "sha", NOW, None),
        ClaimRow(
            id=None,
            exchange_id=1,
            extraction_run_id=1,
            statement="s",
            subject="subj",
            kind=ClaimKind.FACT,
            confidence=0.5,
            probe_question="q?",
            permalink="https://discord.com/channels/1/2/3",
            supersedes_claim_id=None,
            novelty=Novelty.UNPROBED,
            probe_model=None,
            probe_answer=None,
            probed_at=None,
            probe_error=None,
            retracted_at=None,
            retraction_reason=None,
        ),
        ClaimSourceRow(1, 1),
    ]
    assert all(dataclasses.is_dataclass(row) for row in rows)
