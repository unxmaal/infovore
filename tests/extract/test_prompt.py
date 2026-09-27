import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path

from infovore.extract.prompt import (
    PROMPT_SHA256,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    permalink,
    render_prompt,
)
from infovore.extract.protocol import ExtractionRequest
from infovore.rows import (
    AttachmentRow,
    ClaimKind,
    ClaimRow,
    ExchangeRow,
    ExtractionStatus,
    GroupingRule,
    MessageRow,
    Novelty,
    ReactionRow,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def a_message(
    message_id: int,
    author_id: int = 1,
    author_name: str = "alice",
    content: str = "hello",
    created_at: datetime = NOW,
    guild_id: int = 100,
    channel_id: int = 10,
) -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=channel_id,
        guild_id=guild_id,
        author_id=author_id,
        author_name_at_time=author_name,
        author_is_bot=False,
        created_at=created_at,
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=None,
        deleted_at=None,
        ingested_at=NOW,
        raw_json="{}",
    )


def an_exchange(
    exchange_id: int = 1,
    channel_id: int = 10,
    first_message_id: int = 1,
    last_message_id: int = 2,
    parent_exchange_id: int | None = None,
) -> ExchangeRow:
    return ExchangeRow(
        id=exchange_id,
        channel_id=channel_id,
        thread_id=None,
        first_message_id=first_message_id,
        last_message_id=last_message_id,
        started_at=NOW,
        ended_at=NOW,
        message_count=2,
        grouping_rule=GroupingRule.QUIET_GAP,
        content_hash="hash",
        parent_exchange_id=parent_exchange_id,
        extraction_status=ExtractionStatus.PENDING,
        retry_count=0,
        last_error=None,
    )


def a_claim(claim_id: int = 5) -> ClaimRow:
    return ClaimRow(
        id=claim_id,
        exchange_id=99,
        extraction_run_id=1,
        statement="Octane2 needs PROM 6.5",
        subject="IP30",
        kind=ClaimKind.FACT,
        confidence=0.9,
        probe_question="what prom does the octane2 need?",
        permalink="https://discord.com/channels/1/1/1",
        supersedes_claim_id=None,
        novelty=Novelty.UNPROBED,
        probe_model=None,
        probe_answer=None,
        probed_at=None,
        probe_error=None,
        retracted_at=None,
        retraction_reason=None,
    )


def a_request(
    messages: tuple[MessageRow, ...] | None = None,
    context_messages: tuple[MessageRow, ...] = (),
    attachments: tuple[AttachmentRow, ...] = (),
    reactions: tuple[ReactionRow, ...] = (),
    related_claims: tuple[ClaimRow, ...] = (),
    opted_out_user_ids: frozenset[int] = frozenset(),
    channel_name: str = "hardware",
) -> ExtractionRequest:
    return ExtractionRequest(
        exchange=an_exchange(),
        channel_name=channel_name,
        messages=messages if messages is not None else (a_message(1), a_message(2)),
        context_messages=context_messages,
        attachments=attachments,
        reactions=reactions,
        related_claims=related_claims,
        opted_out_user_ids=opted_out_user_ids,
    )


def test_permalink_formats_discord_url() -> None:
    assert permalink(1, 2, 3) == "https://discord.com/channels/1/2/3"


def test_prompt_version_is_v3() -> None:
    assert PROMPT_VERSION == "v5"


def test_prompt_sha256_matches_system_prompt() -> None:
    assert hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest() == PROMPT_SHA256


def test_render_prompt_uses_system_prompt_and_version() -> None:
    rendered = render_prompt(a_request())
    assert rendered.system == SYSTEM_PROMPT
    assert rendered.version == PROMPT_VERSION


def test_render_prompt_contains_channel_name_and_permalink() -> None:
    rendered = render_prompt(a_request(channel_name="hardware"))
    assert "hardware" in rendered.prompt
    assert "https://discord.com/channels/100/10/1" in rendered.prompt


def test_render_prompt_omits_context_section_when_no_context() -> None:
    rendered = render_prompt(a_request(context_messages=()))
    assert "CONTEXT" not in rendered.prompt


def test_render_prompt_includes_context_section_when_context_present() -> None:
    context = (a_message(50, content="earlier message"),)
    rendered = render_prompt(a_request(context_messages=context))
    assert "CONTEXT" in rendered.prompt
    assert "earlier message" in rendered.prompt
    assert "[c1]" in rendered.prompt
    assert "[50]" not in rendered.prompt


def test_render_prompt_related_claims_section_shows_none_when_empty() -> None:
    rendered = render_prompt(a_request(related_claims=()))
    assert "RELATED EXISTING CLAIMS" in rendered.prompt
    assert "none" in rendered.prompt


def test_render_prompt_related_claims_section_lists_claims() -> None:
    rendered = render_prompt(a_request(related_claims=(a_claim(5),)))
    assert "[claim:5]" in rendered.prompt
    assert "(fact)" in rendered.prompt
    assert "IP30" in rendered.prompt
    assert "Octane2 needs PROM 6.5" in rendered.prompt


def test_render_prompt_includes_reactions_and_attachments() -> None:
    reactions = (ReactionRow(message_id=1, emoji="\U0001f44d", count=3),)
    attachments = (
        AttachmentRow(
            id=1,
            message_id=1,
            filename="photo.png",
            content_type="image/png",
            size=10,
            url="https://example.com/photo.png",
            sha256=None,
            local_path=None,
        ),
    )
    rendered = render_prompt(a_request(reactions=reactions, attachments=attachments))
    assert "\U0001f44d\u00d73" in rendered.prompt
    assert "photo.png" in rendered.prompt


def test_render_prompt_redacts_opted_out_authors_but_keeps_message_ref() -> None:
    messages = (
        a_message(1, author_id=1, author_name="alice", content="normal"),
        a_message(2, author_id=2, author_name="bob", content="secret"),
    )
    rendered = render_prompt(a_request(messages=messages, opted_out_user_ids=frozenset({2})))
    assert "[m2]" in rendered.prompt
    assert "bob" not in rendered.prompt
    assert "secret" not in rendered.prompt
    assert "[redacted]" in rendered.prompt


def test_render_prompt_redacts_opted_out_context_authors() -> None:
    context = (a_message(50, author_id=2, author_name="bob", content="secret context"),)
    rendered = render_prompt(a_request(context_messages=context, opted_out_user_ids=frozenset({2})))
    assert "bob" not in rendered.prompt
    assert "secret context" not in rendered.prompt


def test_render_prompt_token_estimate_matches_formula() -> None:
    rendered = render_prompt(a_request())
    expected = math.ceil(len(rendered.system + rendered.prompt) / 4)
    assert rendered.token_estimate == expected


def test_render_prompt_is_deterministic() -> None:
    request = a_request()
    first = render_prompt(request)
    second = render_prompt(request)
    assert first == second


def test_render_prompt_snapshot() -> None:
    context = (a_message(50, author_id=1, author_name="alice", content="earlier context"),)
    reactions = (ReactionRow(message_id=1, emoji="\U0001f44d", count=2),)
    attachments = (
        AttachmentRow(
            id=1,
            message_id=1,
            filename="jumpers.png",
            content_type="image/png",
            size=10,
            url="https://example.com/jumpers.png",
            sha256=None,
            local_path=None,
        ),
    )
    messages = (
        a_message(1, author_id=1, author_name="alice", content="What PROM does an Octane2 need?"),
        a_message(2, author_id=2, author_name="bob", content="6.5 works fine."),
    )
    related_claims = (a_claim(5),)
    request = a_request(
        messages=messages,
        context_messages=context,
        attachments=attachments,
        reactions=reactions,
        related_claims=related_claims,
        channel_name="hardware",
    )
    rendered = render_prompt(request)
    assert rendered.system == SYSTEM_PROMPT
    assert rendered.version == "v5"
    assert rendered.prompt == (
        "CHANNEL: hardware\n"
        "\n"
        "PERMALINK: https://discord.com/channels/100/10/1\n"
        "\n"
        "CONTEXT (do not cite):\n"
        "[c1] member-A @ 2026-01-01T00:00:00+00:00:\n"
        "earlier context\n"
        "\n"
        "EXCHANGE:\n"
        "[m1] member-A @ 2026-01-01T00:00:00+00:00:\n"
        "What PROM does an Octane2 need?\n"
        "Reactions: \U0001f44d\u00d72\n"
        "Attachments: jumpers.png\n"
        "\n"
        "[m2] member-B @ 2026-01-01T00:00:00+00:00:\n"
        "6.5 works fine.\n"
        "\n"
        "RELATED EXISTING CLAIMS:\n"
        "[claim:5] (fact) IP30: Octane2 needs PROM 6.5"
    )
    assert rendered.token_estimate == math.ceil(len(rendered.system + rendered.prompt) / 4)


def test_readme_contains_system_prompt_verbatim() -> None:
    readme = Path(__file__).resolve().parents[2] / "README.md"
    assert SYSTEM_PROMPT in readme.read_text()


def test_render_prompt_maps_exchange_refs_to_message_ids() -> None:
    messages = (a_message(706732704137478123), a_message(706733682781847611))
    rendered = render_prompt(a_request(messages=messages))
    assert rendered.refs == {"m1": 706732704137478123, "m2": 706733682781847611}
    assert "[m1]" in rendered.prompt
    assert "[m2]" in rendered.prompt
    assert "[706733682781847611]" not in rendered.prompt


def test_render_prompt_context_refs_are_not_citable() -> None:
    context = (a_message(50, content="earlier message"),)
    rendered = render_prompt(a_request(context_messages=context))
    assert rendered.refs == {"m1": 1, "m2": 2}


def test_system_prompt_forbids_adding_specifics() -> None:
    assert "Do not add specifics" in SYSTEM_PROMPT
    assert "your own knowledge" in SYSTEM_PROMPT


def test_system_prompt_still_extracts_generously() -> None:
    assert "Extract generously" in SYSTEM_PROMPT


def test_system_prompt_keeps_v2_framing_minimal() -> None:
    assert SYSTEM_PROMPT.startswith(
        "You are reading an archived exchange from a hobbyist SGI/IRIX community."
    )
    assert "general software" not in SYSTEM_PROMPT


def test_system_prompt_preserves_uncertainty() -> None:
    assert "reportedly" in SYSTEM_PROMPT
    assert "lower the confidence" in SYSTEM_PROMPT


def test_system_prompt_protects_private_individuals() -> None:
    assert "member-A" in SYSTEM_PROMPT
    assert "a community member" in SYSTEM_PROMPT
    assert "Businesses and resellers may be named" in SYSTEM_PROMPT


def test_system_prompt_keeps_market_history_in_scope() -> None:
    assert "prices" in SYSTEM_PROMPT
    assert "sales" in SYSTEM_PROMPT


def test_render_prompt_replaces_author_names_with_stable_pseudonyms() -> None:
    context = (a_message(50, author_id=2, author_name="bob", content="earlier"),)
    messages = (
        a_message(1, author_id=1, author_name="alice", content="q"),
        a_message(2, author_id=2, author_name="bob", content="a"),
        a_message(3, author_id=1, author_name="alice", content="thanks"),
    )
    rendered = render_prompt(a_request(messages=messages, context_messages=context))
    assert "alice" not in rendered.prompt
    assert "bob" not in rendered.prompt
    assert "[c1] member-A @" in rendered.prompt
    assert "[m1] member-B @" in rendered.prompt
    assert "[m2] member-A @" in rendered.prompt
    assert "[m3] member-B @" in rendered.prompt


def test_render_prompt_replaces_mentions_with_pseudonyms() -> None:
    messages = (
        a_message(1, author_id=11, author_name="alice", content="hi"),
        a_message(2, author_id=22, author_name="bob", content="<@11> and <@!11> try <@99>"),
    )
    rendered = render_prompt(a_request(messages=messages))
    assert "member-A and member-A try another member" in rendered.prompt
    assert "<@" not in rendered.prompt


def test_render_prompt_redacts_mentions_of_opted_out_members() -> None:
    messages = (
        a_message(1, author_id=11, author_name="alice", content="hi"),
        a_message(2, author_id=22, author_name="bob", content="ask <@11>"),
    )
    rendered = render_prompt(a_request(messages=messages, opted_out_user_ids=frozenset({11})))
    assert "ask [redacted]" in rendered.prompt
    assert "member-A" in rendered.prompt


def test_pseudonym_letters_extend_past_z() -> None:
    from infovore.extract.prompt import pseudonym

    assert pseudonym(0) == "member-A"
    assert pseudonym(25) == "member-Z"
    assert pseudonym(26) == "member-AA"
    assert pseudonym(27) == "member-AB"
