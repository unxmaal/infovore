import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from infovore.rows import AttachmentRow, MessageRow, ReactionRow
from infovore.triage.rules import DEFAULT_RULES, TriageRules

# Structural regexes are code-like enough to stay in code; their weights live
# in TriageRules (see infovore/triage/rules.toml).
IRIX_VERSION = re.compile(r"\b6\.5\.\d{1,2}[mf]?\b|\birix\s*[3-6]\.\d", re.IGNORECASE)
PART_NUMBER = re.compile(r"\b0\d{2}-\d{4}-\d{3}\b")
UNIX_PATH = re.compile(
    r"(?<![\w/])/(?:usr|var|etc|opt|dev|stand|sbin|bin|lib|tmp|hw|proc)/[\w./+-]+"
)
CODE = re.compile(r"```|`[^`\n]+`")

TRIAGE_VERSION = DEFAULT_RULES.version


@dataclass(frozen=True)
class TriageResult:
    score: float
    reasons: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class _CompiledRules:
    domain_terms: re.Pattern[str]
    archive_link: re.Pattern[str]
    gif_link: re.Pattern[str]
    laughter: re.Pattern[str]


@lru_cache(maxsize=8)
def _compile(rules: TriageRules) -> _CompiledRules:
    return _CompiledRules(
        domain_terms=re.compile(rf"\b(?:{'|'.join(rules.domain_terms)})\b", re.IGNORECASE),
        archive_link=re.compile(
            rf"(?:https?|ftp)://\S*(?:{'|'.join(rules.archive_link_hosts)})\S*",
            re.IGNORECASE,
        ),
        gif_link=re.compile(rf"https?://\S*(?:{'|'.join(rules.gif_hosts)})\S*", re.IGNORECASE),
        laughter=re.compile(rf"^\W*(?:{'|'.join(rules.laughter_tokens)})\W*$", re.IGNORECASE),
    )


def _distinct_domain_terms(text: str, domain_terms: re.Pattern[str]) -> int:
    return len({match.group(0).lower() for match in domain_terms.finditer(text)})


def _answered_question(messages: Sequence[MessageRow], answer_min_characters: int) -> bool:
    for index, question in enumerate(messages):
        if "?" not in question.content:
            continue
        for reply in messages[index + 1 :]:
            if (
                reply.author_id != question.author_id
                and len(reply.content.strip()) >= answer_min_characters
            ):
                return True
    return False


def _agreed_answer(
    messages: Sequence[MessageRow],
    reactions: Sequence[ReactionRow],
    agreement_emoji: frozenset[str],
) -> bool:
    later_ids = {message.id for message in messages[1:]}
    return any(
        reaction.message_id in later_ids and reaction.emoji in agreement_emoji
        for reaction in reactions
    )


def _share(messages: Sequence[MessageRow], predicate: re.Pattern[str] | int) -> float:
    if isinstance(predicate, int):
        hits = sum(1 for message in messages if len(message.content.strip()) < predicate)
    else:
        hits = sum(1 for message in messages if predicate.match(message.content.strip()))
    return hits / len(messages)


def score_exchange(
    messages: Sequence[MessageRow],
    reactions: Sequence[ReactionRow] = (),
    attachments: Sequence[AttachmentRow] = (),
    rules: TriageRules = DEFAULT_RULES,
) -> TriageResult:
    if not messages:
        return TriageResult(0.0, ())
    patterns = _compile(rules)
    text = "\n".join(message.content for message in messages)
    reasons: list[tuple[str, float]] = []
    terms = _distinct_domain_terms(text, patterns.domain_terms)
    if terms:
        reasons.append(
            ("domain_terms", min(rules.domain_term_cap, terms * rules.domain_term_weight))
        )
    for name, pattern, weight in (
        ("irix_version", IRIX_VERSION, rules.irix_version_weight),
        ("part_number", PART_NUMBER, rules.part_number_weight),
        ("unix_path", UNIX_PATH, rules.unix_path_weight),
        ("code", CODE, rules.code_weight),
        ("archive_link", patterns.archive_link, rules.archive_link_weight),
    ):
        if pattern.search(text):
            reasons.append((name, weight))
    if any(attachment.filename.lower().endswith(".pdf") for attachment in attachments):
        reasons.append(("pdf_attachment", rules.pdf_attachment_weight))
    if _answered_question(messages, rules.answer_min_characters):
        reasons.append(("answered_question", rules.answered_question_weight))
    if _agreed_answer(messages, reactions, rules.agreement_emoji):
        reasons.append(("agreed_answer", rules.agreed_answer_weight))
    if any(message.thread_id is not None for message in messages):
        reasons.append(("thread", rules.thread_weight))
    if len(text) >= rules.substantial_characters:
        reasons.append(("substantial", rules.substantial_weight))
    if (
        len(messages) > 1
        and _share(messages, rules.tiny_message_characters) > rules.tiny_share_threshold
    ):
        reasons.append(("mostly_tiny_messages", rules.tiny_penalty))
    if patterns.gif_link.search(text):
        reasons.append(("gif_links", rules.gif_penalty))
    if _share(messages, patterns.laughter) > rules.laughter_share_threshold:
        reasons.append(("laughter", rules.laughter_penalty))
    raw = sum(weight for _, weight in reasons)
    return TriageResult(round(min(1.0, max(0.0, raw)), 4), tuple(reasons))
