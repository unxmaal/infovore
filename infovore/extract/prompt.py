import hashlib
import math
import re
from dataclasses import dataclass

from infovore.extract.protocol import ExtractionRequest
from infovore.rows import AttachmentRow, MessageRow, ReactionRow

PROMPT_VERSION = "v8"

SYSTEM_PROMPT = (
    "You are reading an archived exchange from a hobbyist SGI/IRIX community.\n"
    "\n"
    "Your job is to capture domain knowledge a general-purpose LLM would not "
    "already have: specific part numbers, jumper settings, PROM/firmware "
    "versions, IRIX quirks and workarounds, repair procedures, compatibility "
    "facts, sources for software and manuals, and market history (prices, "
    "sales, listings, sellers). Generic computing knowledge is not wanted.\n"
    "\n"
    "Extract generously. A later closed-book novelty probe is the filter, not "
    "you: your job is to notice everything specific and supported by the "
    "messages, not to decide whether it is already widely known.\n"
    "\n"
    "Each claim must be specific and supported by the messages. Do not add "
    "specifics that are not in the messages from your own knowledge. If a "
    "message hedges (probably, I think, might), keep that in the statement "
    "(reportedly, probably) and lower the confidence. Each claim carries a "
    "probe_question that asks for the fact without revealing it, so the fact "
    "can be tested for later without leaking the answer.\n"
    "\n"
    "Authors appear as pseudonyms (member-A, member-B, ...). Never name "
    "people in a claim; say a community member instead. Businesses and "
    "resellers may be named.\n"
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

SYSTEM_PROMPT_V6 = (
    "You are reading an archived exchange from a hobbyist SGI/IRIX community.\n"
    "\n"
    "Record durable technical facts: statements that stay true and useful to "
    "someone who never reads this conversation. Part numbers, jumper and "
    "switch settings, PROM/firmware versions, IRIX quirks and the workarounds "
    "for them, repair and installation procedures, compatibility between "
    "specific parts, where software and manuals can be obtained, and what a "
    "model sold for and when. Generic computing knowledge is not wanted.\n"
    "\n"
    "Every claim is about a THING. Its grammatical subject must be the "
    "hardware, the software, the part or the procedure, never a person and "
    "never an unnamed someone. Write 'The SGI O2 power supply can be "
    "substituted with a Meanwell modular unit', not 'a member found that it "
    "can'. Who said it is recorded in sources and the permalink; it does not "
    "belong in the sentence. Authors appear as pseudonyms (member-A, "
    "member-B, ...) and must never be named. Businesses and resellers may be "
    "named, and a price needs its date.\n"
    "\n"
    "An occasion is not a fact. One person's purchase, one machine's "
    "behaviour on one afternoon, what somebody intends to try next, an "
    "unanswered question, and an opinion are all occasions and yield nothing, "
    "however specific they are. Ask of each claim whether it would still be "
    "worth reading in ten years by someone holding the same hardware.\n"
    "\n"
    "Record only what the messages establish. If the exchange supports only a "
    "hedged or disputed statement, leave it out rather than hedging it in "
    "words. Never add specifics from your own knowledge.\n"
    "\n"
    "Most exchanges yield nothing, and zero claims is the correct and common "
    "answer for ordinary conversation. Never return more than five claims "
    "from one exchange; if more seem available, keep the most durable.\n"
    "\n"
    "If a claim corrects one of the supplied related existing claims, cite "
    "that claim's id in supersedes. If the community corrects itself within "
    "this exchange, record only the corrected version, never the original "
    "mistake.\n"
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

SYSTEM_PROMPT_V7 = SYSTEM_PROMPT_V6.replace(
    # v6 refused anything hedged, which in an archive where almost every real
    # fact arrives attributed and tentative also refused the facts: measured
    # 26.5% recovery of v5's cited sources against v5's own 83% ceiling, and
    # nothing at all on 20 of 40 exchanges (issue #165).
    "Record only what the messages establish. If the exchange supports only a "
    "hedged or disputed statement, leave it out rather than hedging it in "
    "words. Never add specifics from your own knowledge.",
    "Record only what the messages establish, and never add specifics from "
    "your own knowledge. A fact stated tentatively is still a fact: "
    "'the ucontext struct reportedly holds no program counter' is worth "
    "recording, written as a statement about the struct, with the "
    "uncertainty carried in confidence rather than in the words. What "
    "yields nothing is a GUESS about what might be true, a plan, a "
    "question, or a statement the exchange itself disputes and never "
    "settles.",
)

SYSTEM_PROMPT_V8 = SYSTEM_PROMPT_V7.replace(
    # v7 still refused "one machine's behaviour on one afternoon", which is
    # how a fault and its symptoms arrive in a chat archive. Measured: v7
    # dropped "faulty IR pipe on an Onyx: irsaudit fails and X11 crashes
    # soon after login", which is the most useful shape of fact here
    # (issue #165). An occasion can be EVIDENCE for a fact about the class.
    "An occasion is not a fact. One person's purchase, one machine's "
    "behaviour on one afternoon, what somebody intends to try next, an "
    "unanswered question, and an opinion are all occasions and yield nothing, "
    "however specific they are.",
    "An occasion is not a fact, but an occasion can be the evidence for "
    "one. The event itself is never the claim: one person's purchase, a "
    "dispute with a seller, what somebody intends to try next, an "
    "unanswered question and an opinion all yield nothing. But when one "
    "machine's behaviour tells you something about that model or part, "
    "record it: a fault and the symptoms it produces, a part that turned "
    "out to fit, a command that did or did not work, a limit someone ran "
    "into. Write it as a statement about the hardware or software rather "
    "than about the occasion.",
)

PROMPTS = {
    "v5": SYSTEM_PROMPT,
    "v6": SYSTEM_PROMPT_V6,
    "v7": SYSTEM_PROMPT_V7,
    "v8": SYSTEM_PROMPT_V8,
}

# Bumping this halts live extraction until the new version is promoted
# (`runner.PromptNotPromotedError`), so a candidate prompt ships selectable
# and unpromoted, and only this constant decides what production sends.
LIVE_PROMPT_VERSION = PROMPT_VERSION


class UnknownPromptVersionError(KeyError):
    pass


def system_prompt(version: str = LIVE_PROMPT_VERSION) -> str:
    try:
        return PROMPTS[version]
    except KeyError as error:
        raise UnknownPromptVersionError(version) from error


PROMPT_SHA256 = hashlib.sha256(PROMPTS[PROMPT_VERSION].encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    prompt: str
    version: str
    token_estimate: int
    refs: dict[str, int]


def permalink(guild_id: int, channel_id: int, message_id: int) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


MENTION = re.compile(r"<@!?(\d+)>")
REDACTED = "[redacted]"
UNKNOWN_MEMBER = "another member"


def pseudonym(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return f"member-{letters}"


def _pseudonyms(request: ExtractionRequest) -> dict[int, str]:
    names: dict[int, str] = {}
    for message in (*request.context_messages, *request.messages):
        if message.author_id in request.opted_out_user_ids or message.author_id in names:
            continue
        names[message.author_id] = pseudonym(len(names))
    return names


def _replace_mentions(
    content: str, names: dict[int, str], opted_out_user_ids: frozenset[int]
) -> str:
    def replace(match: re.Match[str]) -> str:
        user_id = int(match.group(1))
        if user_id in opted_out_user_ids:
            return REDACTED
        return names.get(user_id, UNKNOWN_MEMBER)

    return MENTION.sub(replace, content)


def _redacted_author_and_content(
    message: MessageRow, opted_out_user_ids: frozenset[int], names: dict[int, str]
) -> tuple[str, str]:
    if message.author_id in opted_out_user_ids:
        return REDACTED, REDACTED
    return names[message.author_id], _replace_mentions(message.content, names, opted_out_user_ids)


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
    names: dict[int, str],
    reactions: tuple[ReactionRow, ...] = (),
    attachments: tuple[AttachmentRow, ...] = (),
) -> str:
    author, content = _redacted_author_and_content(message, opted_out_user_ids, names)
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


def render_prompt(request: ExtractionRequest, version: str = LIVE_PROMPT_VERSION) -> RenderedPrompt:
    guild_id = request.messages[0].guild_id
    link = permalink(guild_id, request.exchange.channel_id, request.exchange.first_message_id)
    sections = [
        f"CHANNEL: {request.channel_name}",
        f"PERMALINK: {link}",
    ]
    names = _pseudonyms(request)
    if request.context_messages:
        rendered_context = "\n\n".join(
            _render_message(f"c{index}", message, request.opted_out_user_ids, names)
            for index, message in enumerate(request.context_messages, start=1)
        )
        sections.append(f"CONTEXT (do not cite):\n{rendered_context}")
    refs = {f"m{index}": message.id for index, message in enumerate(request.messages, start=1)}
    rendered_exchange = "\n\n".join(
        _render_message(
            ref,
            message,
            request.opted_out_user_ids,
            names,
            request.reactions,
            request.attachments,
        )
        for ref, message in zip(refs, request.messages, strict=True)
    )
    sections.append(f"EXCHANGE:\n{rendered_exchange}")
    sections.append(f"RELATED EXISTING CLAIMS:\n{_render_related_claims(request)}")
    prompt = "\n\n".join(sections)
    system = system_prompt(version)
    token_estimate = math.ceil(len(system + prompt) / 4)
    return RenderedPrompt(
        system=system,
        prompt=prompt,
        version=version,
        token_estimate=token_estimate,
        refs=refs,
    )
