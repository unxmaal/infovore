import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from infovore.config import ConfigError
from infovore.rows import MessageRow

MENTION = re.compile(r"<@!?(\d+)>")
WIDTH = 4
MIN_NAME = 3
COMMON = frozenset(
    [
        "the",
        "and",
        "for",
        "you",
        "are",
        "but",
        "not",
        "was",
        "all",
        "can",
        "had",
        "her",
        "his",
        "its",
        "one",
        "our",
        "out",
        "who",
        "how",
        "why",
        "too",
        "now",
        "new",
        "she",
        "him",
        "has",
        "did",
        "get",
        "got",
        "let",
        "say",
        "see",
        "use",
        "any",
        "may",
        "way",
        "day",
        "yes",
        "this",
        "that",
        "with",
        "have",
        "from",
        "they",
        "them",
        "then",
        "than",
        "what",
        "when",
        "where",
        "will",
        "your",
        "just",
        "like",
        "some",
        "more",
        "most",
        "much",
        "very",
        "been",
        "were",
        "also",
        "into",
        "only",
        "over",
        "such",
        "here",
        "there",
        "these",
        "those",
        "would",
        "could",
        "should",
        "about",
        "which",
        "their",
        "other",
        "user",
        "users",
        "bot",
        "admin",
        "mod",
    ]
)


@dataclass(frozen=True)
class RenderedLine:
    ref: int
    message_id: int
    speaker: str
    text: str


@dataclass(frozen=True)
class Redacted:
    lines: list[RenderedLine]
    speakers: set[str]
    names: list[str]


def require_salt(salt: str | None) -> str:
    if not salt:
        raise ConfigError("INFOVORE_PSEUDONYM_SALT is required: names are redacted with it")
    return salt


def _digest(user_id: int, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()


def pseudonym(user_id: int, salt: str, width: int = WIDTH) -> str:
    return f"user-{_digest(user_id, salt)[:width]}"


def pseudonyms(user_ids: Iterable[int], salt: str, width: int = WIDTH) -> dict[int, str]:
    mapping: dict[int, str] = {}
    taken: set[str] = set()
    for user_id in sorted(set(user_ids)):
        size = width
        while (name := pseudonym(user_id, salt, size)) in taken:
            size += 1
        taken.add(name)
        mapping[user_id] = name
    return mapping


def _name_tokens(name: str) -> set[str]:
    parts = {name.strip(), *name.split()}
    return {p for p in parts if len(p) >= MIN_NAME and p.lower() not in COMMON}


def leaks(text: str, names: Sequence[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text, re.I) for n in names)


def redact_conversation(
    messages: Sequence[MessageRow], salt: str, drop: frozenset[int] = frozenset()
) -> Redacted:
    by_user = pseudonyms((m.author_id for m in messages), salt)
    replacements: dict[str, str] = {}
    for message in messages:
        for token in _name_tokens(message.author_name_at_time):
            replacements.setdefault(token.lower(), by_user[message.author_id])
    names = sorted(replacements, key=len, reverse=True)
    name_pattern = (
        re.compile(
            r"@?(?<!\w)(?:" + "|".join(re.escape(n) for n in names) + r")(?!\w)",
            re.I,
        )
        if names
        else None
    )

    def mention(match: re.Match[str]) -> str:
        user_id = int(match.group(1))
        return by_user.get(user_id) or pseudonym(user_id, salt)

    lines: list[RenderedLine] = []
    for message in messages:
        if message.author_id in drop:
            continue
        text = MENTION.sub(mention, message.content).strip()
        if name_pattern is not None:
            text = name_pattern.sub(lambda m: replacements[m.group(0).lstrip("@").lower()], text)
        if text:
            lines.append(RenderedLine(len(lines) + 1, message.id, by_user[message.author_id], text))
    return Redacted(lines, set(by_user.values()), names)
