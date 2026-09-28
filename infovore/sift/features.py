"""Message-level features for the trash classifier (issue #128 PR 3).

Mirrors `infovore.triage.bayes.features` (lower-cased word tokens, bounded to
`MAX_TOKEN_LENGTH`, plus virtual tokens), but built from a single message
rather than a whole exchange: `infovore.rows.MessageRow` already carries its
own `channel_id` and `reply_to_id` directly, so no exchange lookup is needed
to build a message's tokens (unlike the exchange-level classifier, which has
to resolve each exchange's `channel_id` and gather reactions/attachments
across every member message).

Virtual tokens:
- `CHAN_<channel_id>` -- so the classifier learns channel priors from data.
- `MLEN_<bucket>` -- a length bucket over the message's own character count
  (distinct from triage's `LEN_<bucket>`, which buckets an *exchange's*
  message count).
- `SIG_url`, `SIG_digits`, `SIG_code`, `SIG_question`, `SIG_reply` --
  structural signals: a URL, a digit (covers version strings like `6.5.22`
  and part numbers like `030-1234-001` alike -- any digit is evidence
  either way, and the Bayes model learns which), a code block or inline
  backtick, a question mark, and whether the message is itself a reply.
"""

import re

from infovore.rows import MessageRow

TOKEN = re.compile(r"[\w/][\w'./+-]*")
TRAILING_PUNCTUATION = ".,!?;:)'\"-"
MAX_TOKEN_LENGTH = 40

URL_RE = re.compile(r"https?://\S+")

_LENGTH_BUCKETS: tuple[tuple[int, str], ...] = (
    (10, "MLEN_0-10"),
    (40, "MLEN_11-40"),
    (120, "MLEN_41-120"),
)
_LENGTH_OVERFLOW_BUCKET = "MLEN_121+"


def message_length_bucket(content: str) -> str:
    length = len(content)
    for limit, bucket in _LENGTH_BUCKETS:
        if length <= limit:
            return bucket
    return _LENGTH_OVERFLOW_BUCKET


def _word_tokens(content: str) -> frozenset[str]:
    words = {token.rstrip(TRAILING_PUNCTUATION) for token in TOKEN.findall(content.lower())}
    return frozenset(word for word in words if word and len(word) <= MAX_TOKEN_LENGTH)


def _structural_signals(message: MessageRow) -> frozenset[str]:
    content = message.content
    signals: set[str] = set()
    if URL_RE.search(content):
        signals.add("SIG_url")
    if any(character.isdigit() for character in content):
        signals.add("SIG_digits")
    if "`" in content:
        signals.add("SIG_code")
    if "?" in content:
        signals.add("SIG_question")
    if message.reply_to_id is not None:
        signals.add("SIG_reply")
    return frozenset(signals)


def message_features(message: MessageRow, channel_id: int) -> frozenset[str]:
    """The token set `infovore.sift.train` trains and scores on for a single
    message: its own word tokens, structural signals, a `CHAN_<channel_id>`
    token (`channel_id` is taken as a parameter, not read off `message`, so a
    caller scoring many messages from an already-batched query need not
    round-trip through `message.channel_id` -- though in practice the two
    always agree), and a message-length bucket."""
    return frozenset(
        _word_tokens(message.content)
        | _structural_signals(message)
        | {f"CHAN_{channel_id}", message_length_bucket(message.content)}
    )
