import re
from collections.abc import Sequence
from dataclasses import dataclass

from infovore.rows import AttachmentRow, MessageRow, ReactionRow

TRIAGE_VERSION = "t1"

DOMAIN_TERM_WEIGHT = 0.15
DOMAIN_TERM_CAP = 0.45
IRIX_VERSION_WEIGHT = 0.2
PART_NUMBER_WEIGHT = 0.3
UNIX_PATH_WEIGHT = 0.15
CODE_WEIGHT = 0.15
ARCHIVE_LINK_WEIGHT = 0.15
PDF_ATTACHMENT_WEIGHT = 0.15
ANSWERED_QUESTION_WEIGHT = 0.2
AGREED_ANSWER_WEIGHT = 0.05
THREAD_WEIGHT = 0.05
SUBSTANTIAL_WEIGHT = 0.1
SUBSTANTIAL_CHARACTERS = 400
ANSWER_MIN_CHARACTERS = 40
TINY_MESSAGE_CHARACTERS = 20
TINY_SHARE_THRESHOLD = 0.7
TINY_PENALTY = -0.2
GIF_PENALTY = -0.1
LAUGHTER_PENALTY = -0.1
LAUGHTER_SHARE_THRESHOLD = 0.3

DOMAIN_TERMS = re.compile(
    r"\b(?:sgi|silicon graphics|irix|indy|indigo2?|o2|octane2?|fuel|tezro|onyx[234]?|origin"
    r"|crimson|challenge|personal iris|ip\d{2}|r\d{4,5}|r1[0-6]k|hinv|inst|swmgr|nvram|prom"
    r"|xfs|xlv|mipspro|sash|gio|xio|xtalk|vpro|odyssey|impact|infinitereality|reality engine"
    r"|irixpro|nekoware|sgug|mipsel|mips)\b",
    re.IGNORECASE,
)
IRIX_VERSION = re.compile(r"\b6\.5\.\d{1,2}[mf]?\b|\birix\s*[3-6]\.\d", re.IGNORECASE)
PART_NUMBER = re.compile(r"\b0\d{2}-\d{4}-\d{3}\b")
UNIX_PATH = re.compile(
    r"(?<![\w/])/(?:usr|var|etc|opt|dev|stand|sbin|bin|lib|tmp|hw|proc)/[\w./+-]+"
)
CODE = re.compile(r"```|`[^`\n]+`")
ARCHIVE_LINK = re.compile(
    r"(?:https?|ftp)://\S*(?:ftp\.|archive\.org|techpubs|bitsavers|irix|sgi|nekochan)\S*",
    re.IGNORECASE,
)
GIF_LINK = re.compile(r"https?://\S*(?:tenor\.com|giphy\.com|\.gif\b)\S*", re.IGNORECASE)
LAUGHTER = re.compile(r"^\W*(?:lol+|lmao+|rofl|ha(?:ha)+|hehe+|xd+|kek)\W*$", re.IGNORECASE)
AGREEMENT_EMOJI = frozenset({"✅", "👍", "☑️", "✔️", "💯"})


@dataclass(frozen=True)
class TriageResult:
    score: float
    reasons: tuple[tuple[str, float], ...]


def _distinct_domain_terms(text: str) -> int:
    return len({match.group(0).lower() for match in DOMAIN_TERMS.finditer(text)})


def _answered_question(messages: Sequence[MessageRow]) -> bool:
    for index, question in enumerate(messages):
        if "?" not in question.content:
            continue
        for reply in messages[index + 1 :]:
            if (
                reply.author_id != question.author_id
                and len(reply.content.strip()) >= ANSWER_MIN_CHARACTERS
            ):
                return True
    return False


def _agreed_answer(messages: Sequence[MessageRow], reactions: Sequence[ReactionRow]) -> bool:
    later_ids = {message.id for message in messages[1:]}
    return any(
        reaction.message_id in later_ids and reaction.emoji in AGREEMENT_EMOJI
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
) -> TriageResult:
    if not messages:
        return TriageResult(0.0, ())
    text = "\n".join(message.content for message in messages)
    reasons: list[tuple[str, float]] = []
    terms = _distinct_domain_terms(text)
    if terms:
        reasons.append(("domain_terms", min(DOMAIN_TERM_CAP, terms * DOMAIN_TERM_WEIGHT)))
    for name, pattern, weight in (
        ("irix_version", IRIX_VERSION, IRIX_VERSION_WEIGHT),
        ("part_number", PART_NUMBER, PART_NUMBER_WEIGHT),
        ("unix_path", UNIX_PATH, UNIX_PATH_WEIGHT),
        ("code", CODE, CODE_WEIGHT),
        ("archive_link", ARCHIVE_LINK, ARCHIVE_LINK_WEIGHT),
    ):
        if pattern.search(text):
            reasons.append((name, weight))
    if any(attachment.filename.lower().endswith(".pdf") for attachment in attachments):
        reasons.append(("pdf_attachment", PDF_ATTACHMENT_WEIGHT))
    if _answered_question(messages):
        reasons.append(("answered_question", ANSWERED_QUESTION_WEIGHT))
    if _agreed_answer(messages, reactions):
        reasons.append(("agreed_answer", AGREED_ANSWER_WEIGHT))
    if any(message.thread_id is not None for message in messages):
        reasons.append(("thread", THREAD_WEIGHT))
    if len(text) >= SUBSTANTIAL_CHARACTERS:
        reasons.append(("substantial", SUBSTANTIAL_WEIGHT))
    if len(messages) > 1 and _share(messages, TINY_MESSAGE_CHARACTERS) > TINY_SHARE_THRESHOLD:
        reasons.append(("mostly_tiny_messages", TINY_PENALTY))
    if GIF_LINK.search(text):
        reasons.append(("gif_links", GIF_PENALTY))
    if _share(messages, LAUGHTER) > LAUGHTER_SHARE_THRESHOLD:
        reasons.append(("laughter", LAUGHTER_PENALTY))
    raw = sum(weight for _, weight in reasons)
    return TriageResult(round(min(1.0, max(0.0, raw)), 4), tuple(reasons))
