from typing import Final

from infovore.claims.check import VERDICTS
from infovore.config import ConfigError

REJECTED_REVIEWS: Final = frozenset({"wrong", "made_up", "not_useful"})
UNCHECKED: Final = "unchecked"
SUPPORTED: Final = frozenset({"supported"})
ANY: Final = "any"
KNOWN: Final = frozenset(VERDICTS) | {UNCHECKED}
DEFAULT_VERDICTS: Final = "supported,uncheckable"
VERDICTS_HELP: Final = (
    "claim-check verdicts allowed on pages: a comma-separated subset of"
    " supported, uncheckable, unsupported_fact, low_overlap, unchecked, or 'any'"
    " (a 'good' human review always passes)"
)


def is_publishable(
    review: str | None, check: str | None, verdicts: frozenset[str] | None = None
) -> bool:
    if review in REJECTED_REVIEWS:
        return False
    if verdicts is None or review == "good":
        return True
    return (check or UNCHECKED) in verdicts


def parse_verdicts(text: str) -> frozenset[str] | None:
    if text == ANY:
        return None
    names = frozenset(part.strip() for part in text.split(",") if part.strip())
    unknown = names - KNOWN
    if not names or unknown:
        raise ConfigError(
            f"--verdicts takes '{ANY}' or a comma-separated subset of"
            f" {', '.join(sorted(KNOWN))}; got {text!r}"
        )
    return names
