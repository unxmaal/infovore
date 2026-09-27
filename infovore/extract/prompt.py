import hashlib
import math
from dataclasses import dataclass

from infovore.extract.protocol import ExtractionRequest
from infovore.rows import AttachmentRow, MessageRow, ReactionRow

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = (
    "You are reading an archived exchange from a hobbyist SGI/IRIX community.\n"
    "\n"
    "Your job is to capture domain knowledge a general-purpose LLM would not "
    "already have: specific part numbers, jumper settings, PROM/firmware "
    "versions, IRIX quirks and workarounds, repair procedures, compatibility "
    "facts, and sources for software and manuals. Generic computing knowledge "
    "is not wanted.\n"
    "\n"
    "Extract generously. A later closed-book novelty probe is the filter, not "
    "you: your job is to notice everything specific and supported by the "
    "messages, not to decide whether it is already widely known.\n"
    "\n"
    "Each claim must be specific and supported by the messages. Each claim "
    "carries a probe_question that asks for the fact without revealing it, so "
    "the fact can be tested for later without leaking the answer.\n"
    "\n"
    "If a claim corrects one of the supplied related existing claims, cite "
    "that claim's id in supersedes. If the community corrects itself within "
    "this exchange, extract only the corrected version, never the original "
    "mistake.\n"
    "\n"
    "Chatter, opinions, and questions that are never answered yield zero "
    "claims.\n"
    "\n"
    "Every claim must list in sources the refs (m1, m2, ...) of the "
    "messages in this exchange that support it. Never cite a message from "
    "the CONTEXT section (refs c1, c2, ...): those messages are read-only "
    "background from a prior exchange and cannot be cited.\n"
    "\n"
    "Reactions are provided as a weak signal of community agreement, not "
    "proof.\n"
    "\n"
    "Output ONLY a JSON object matching the given schema. No other text."
)

PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    prompt: str
    version: str
    token_estimate: int
    refs: dict[str, int]


def permalink(guild_id: int, channel_id: int, message_id: int) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def _redacted_author_and_content(
    message: MessageRow, opted_out_user_ids: frozenset[int]
) -> tuple[str, str]:
    if message.author_id in opted_out_user_ids:
        return "[redacted]", "[redacted]"
    return message.author_name_at_time, message.content


def _reactions_line(message_id: int, reactions: tuple[ReactionRow, ...]) -> str | None:
    matches = [reaction for reaction in reactions if reaction.message_id == message_id]
    if not matches:
        return None
    return "Reactions: " + ", ".join(f"{r.emoji}\u00d7{r.count}" for r in matches)


def _attachments_line(message_id: int, attachments: tuple[AttachmentRow, ...]) -> str | None:
    matches = [attachment for attachment in attachments if attachment.message_id == message_id]
    if not matches:
        return None
    return "Attachments: " + ", ".join(a.filename for a in matches)


def _render_message(
    ref: str,
    message: MessageRow,
    opted_out_user_ids: frozenset[int],
    reactions: tuple[ReactionRow, ...] = (),
    attachments: tuple[AttachmentRow, ...] = (),
) -> str:
    author, content = _redacted_author_and_content(message, opted_out_user_ids)
    lines = [f"[{ref}] {author} @ {message.created_at.isoformat()}:", content]
    reactions_line = _reactions_line(message.id, reactions)
    if reactions_line is not None:
        lines.append(reactions_line)
    attachments_line = _attachments_line(message.id, attachments)
    if attachments_line is not None:
        lines.append(attachments_line)
    return "\n".join(lines)


def _render_related_claims(request: ExtractionRequest) -> str:
    if not request.related_claims:
        return "none"
    return "\n".join(
        f"[claim:{claim.id}] ({claim.kind.value}) {claim.subject}: {claim.statement}"
        for claim in request.related_claims
    )


def render_prompt(request: ExtractionRequest) -> RenderedPrompt:
    guild_id = request.messages[0].guild_id
    link = permalink(guild_id, request.exchange.channel_id, request.exchange.first_message_id)
    sections = [
        f"CHANNEL: {request.channel_name}",
        f"PERMALINK: {link}",
    ]
    if request.context_messages:
        rendered_context = "\n\n".join(
            _render_message(f"c{index}", message, request.opted_out_user_ids)
            for index, message in enumerate(request.context_messages, start=1)
        )
        sections.append(f"CONTEXT (do not cite):\n{rendered_context}")
    refs = {f"m{index}": message.id for index, message in enumerate(request.messages, start=1)}
    rendered_exchange = "\n\n".join(
        _render_message(
            ref, message, request.opted_out_user_ids, request.reactions, request.attachments
        )
        for ref, message in zip(refs, request.messages, strict=True)
    )
    sections.append(f"EXCHANGE:\n{rendered_exchange}")
    sections.append(f"RELATED EXISTING CLAIMS:\n{_render_related_claims(request)}")
    prompt = "\n\n".join(sections)
    token_estimate = math.ceil(len(SYSTEM_PROMPT + prompt) / 4)
    return RenderedPrompt(
        system=SYSTEM_PROMPT,
        prompt=prompt,
        version=PROMPT_VERSION,
        token_estimate=token_estimate,
        refs=refs,
    )
