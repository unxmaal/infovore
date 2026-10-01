import re
from dataclasses import dataclass

PERSON_NOUNS = (
    "community member",
    "a member",
    "one member",
    "another member",
    "a user",
    "one user",
    "a poster",
    "the poster",
    "the author",
    "someone",
    "somebody",
    "a contributor",
    "a maintainer",
    "a participant",
    "a commenter",
)

ATTRIBUTION_VERBS = (
    "reported",
    "reports",
    "said",
    "says",
    "stated",
    "states",
    "noted",
    "notes",
    "observed",
    "observes",
    "mentioned",
    "mentions",
    "claimed",
    "claims",
    "explained",
    "explains",
    "confirmed",
    "confirms",
    "asked",
    "asks",
    "wondered",
    "speculated",
    "suggested",
    "recalled",
    "described",
    "describes",
    "plans to",
    "intends to",
    "hopes to",
)

_PERSON_SUBJECT = re.compile(
    r"^\s*(?:a|an|the|one|another)?\s*(?:"
    + "|".join(
        re.escape(
            noun.split(maxsplit=1)[-1]
            if noun.split()[0] in {"a", "an", "the", "one", "another"}
            else noun
        )
        for noun in PERSON_NOUNS
    )
    + r")\b",
    re.IGNORECASE,
)

_ACCORDING_TO = re.compile(r"\baccording to\b", re.IGNORECASE)


@dataclass(frozen=True)
class ShapeCounts:
    total: int
    person_subject: int
    attributed: int

    @property
    def person_subject_share(self) -> float:
        return self.person_subject / self.total if self.total else 0.0

    @property
    def attributed_share(self) -> float:
        return self.attributed / self.total if self.total else 0.0


def has_person_subject(statement: str) -> bool:
    """Whether the claim's grammatical subject is a person rather than the
    hardware or software. A heuristic on the leading noun phrase, not a
    parser: it is the acceptance metric for issue #165, so it must be
    mechanical and cheap enough to run over the whole claims table."""
    return bool(_PERSON_SUBJECT.match(statement))


def attributes_to_a_speaker(statement: str) -> bool:
    """Looser than `has_person_subject`: a person anywhere as the source of
    the fact, including mid-sentence ('X, which a member confirmed') and
    'according to'."""
    lowered = statement.lower()
    if _ACCORDING_TO.search(lowered):
        return True
    for noun in PERSON_NOUNS:
        index = lowered.find(noun)
        while index != -1:
            tail = lowered[index + len(noun) : index + len(noun) + 40]
            if any(verb in tail for verb in ATTRIBUTION_VERBS):
                return True
            index = lowered.find(noun, index + 1)
    return False


def shape_counts(statements: list[str]) -> ShapeCounts:
    return ShapeCounts(
        total=len(statements),
        person_subject=sum(1 for text in statements if has_person_subject(text)),
        attributed=sum(1 for text in statements if attributes_to_a_speaker(text)),
    )
