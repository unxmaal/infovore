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

**Conversation context (issue #141).** Since #137 the maintainer labels a
message with its *conversation context* in view ("would anything be lost if
this message vanished?"), but the classifier above sees each message alone --
"that works" is junk in isolation but a meaningful confirmation after a
how-to. `build_context_tokens`/`exchange_context_tokens` add, as namespaced
virtual tokens so they never collide with a message's own words:

- `PREV_<tok>`/`NEXT_<tok>` -- the immediately preceding/following message's
  own word tokens, capped at `NEIGHBOUR_TOKEN_CAP` distinct tokens (sorted
  before capping, so the cap is deterministic) to limit length bias.
- `REPLYTO_<tok>` -- the same, for the message this one replies to, when
  `reply_to_id` resolves (the target may be outside the exchange -- see
  `exchange_context_tokens`).
- `*_FACT_url`/`*_FACT_version`/`*_FACT_partnumber`/`*_FACT_path`/
  `*_FACT_code` (same three namespaces) -- a neighbour's fact-shaped
  signals, finer-grained than the single-message `SIG_digits`.
- `POS_first`/`POS_early`/`POS_middle`/`POS_last` -- position bucket in the
  exchange; `EXSIZE_<bucket>` -- the exchange's size bucket.
- `CTX_is_reply`, `CTX_prev_ends_question`, `CTX_same_author_prev`.

A neighbour whose author has opted out contributes **no** tokens at all
(same rule as prompts): opted-out authors' messages are still scored/
trained like any other message today (unchanged by this issue), but they
must never leak content into a neighbour's features.

`FEATURE_SET_VERSION` is bumped whenever the feature set built by this
module changes shape -- `infovore.sift.train` stores it with the trained
ensemble and refuses to score with a model trained under a different
version (see `infovore.sift.train.FeatureSetMismatchError`).
"""

import re
import sqlite3
from collections.abc import Mapping, Sequence
from enum import StrEnum

from infovore.db.batch import exchange_inputs_for_ids
from infovore.db.raw import messages_by_ids
from infovore.rows import MessageRow

TOKEN = re.compile(r"[\w/][\w'./+-]*")
TRAILING_PUNCTUATION = ".,!?;:)'\"-"
MAX_TOKEN_LENGTH = 40

URL_RE = re.compile(r"https?://\S+")
_VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)*\b")
_PART_NUMBER_RE = re.compile(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")
_PATH_RE = re.compile(r"(?:/[\w.-]+){2,}|[A-Za-z]:\\[\w\\.-]+")

# Bumped whenever `message_features`/`build_context_tokens` changes the
# shape of the token set a model is trained/scored on (issue #141): version
# 1 is the pre-#141 feature set (no conversation-context tokens at all),
# version 2 adds them. `infovore.sift.train` stores this with the trained
# ensemble and refuses to score with a mismatched version.
FEATURE_SET_VERSION = 2


class FeatureSet(StrEnum):
    """The three selectable message feature sets (issue #144, following
    #141's own ablation: on the real corpus -- 1,184 human labels, ~238k
    citation labels -- conversation-context features measured *worse* than
    no context at all: combined out-of-fold AUC vs human labels 0.805 with
    context vs 0.854 without, `keep_lost@0.7` 13.9% vs 8.4%,
    `trash_caught@0.9` 19% vs 28%):

    - `PLAIN` -- the pre-#141 message features only: a message's own word
      tokens, structural signals (`SIG_*`), a `CHAN_<channel_id>` token,
      and a length bucket (`MLEN_*`). No conversation context at all --
      the default (`DEFAULT_FEATURE_SET`), since it's what measured best
      against real human labels.
    - `STRUCTURAL` -- `PLAIN` plus only the *shape* of the surrounding
      conversation: position/size buckets (`POS_*`, `EXSIZE_*`), the
      `CTX_*` flags, and a neighbour's fact-shape tokens (`*_FACT_*`) --
      but never a neighbour's own word tokens (no `PREV_`/`NEXT_`/
      `REPLYTO_<word>`), which is exactly what #141's ablation found hurts
      naive Bayes (a long, wordy neighbour swamps the model -- a
      length-bias-like failure).
    - `CONTEXT` -- everything `build_context_tokens`/`exchange_context_tokens`
      produce, unfiltered: #141's original, always-on-until-now behaviour.

    `infovore sift train --features` selects one; `infovore.sift.train`
    persists the chosen name alongside `FEATURE_SET_VERSION`
    (`message_combiner.feature_set_name`/`.feature_set_version`), and
    scoring builds exactly that set (`context_tokens_for_feature_set`)."""

    PLAIN = "plain"
    STRUCTURAL = "structural"
    CONTEXT = "context"


# `infovore sift train --features`'s default when unset: the pre-#141
# shape, which measured best against real human labels (see `FeatureSet`).
DEFAULT_FEATURE_SET = FeatureSet.PLAIN

# How many distinct neighbour word tokens (`PREV_`/`NEXT_`/`REPLYTO_`) a
# single message contributes, to limit length bias -- a long neighbour
# shouldn't get to outvote a short one purely on token count.
NEIGHBOUR_TOKEN_CAP = 30

_LENGTH_BUCKETS: tuple[tuple[int, str], ...] = (
    (10, "MLEN_0-10"),
    (40, "MLEN_11-40"),
    (120, "MLEN_41-120"),
)
_LENGTH_OVERFLOW_BUCKET = "MLEN_121+"

_SIZE_BUCKETS: tuple[tuple[int, str], ...] = (
    (2, "EXSIZE_1-2"),
    (5, "EXSIZE_3-5"),
    (15, "EXSIZE_6-15"),
)
_SIZE_OVERFLOW_BUCKET = "EXSIZE_16+"


def message_length_bucket(content: str) -> str:
    length = len(content)
    for limit, bucket in _LENGTH_BUCKETS:
        if length <= limit:
            return bucket
    return _LENGTH_OVERFLOW_BUCKET


def exchange_size_bucket(size: int) -> str:
    for limit, bucket in _SIZE_BUCKETS:
        if size <= limit:
            return bucket
    return _SIZE_OVERFLOW_BUCKET


def position_bucket(index: int, size: int) -> str:
    """`index`'s bucket among `size` position-ordered exchange messages:
    the first message is always `POS_first` (even alone, `size == 1`), the
    last (when distinct from the first) is `POS_last`, and everything else
    splits into `POS_early` (within the first third) or `POS_middle`."""
    if size <= 1 or index <= 0:
        return "POS_first"
    if index >= size - 1:
        return "POS_last"
    if index <= max(1, size // 3):
        return "POS_early"
    return "POS_middle"


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


def _capped_neighbour_tokens(namespace: str, content: str) -> frozenset[str]:
    words = sorted(_word_tokens(content))[:NEIGHBOUR_TOKEN_CAP]
    return frozenset(f"{namespace}_{word}" for word in words)


def _fact_shape_tokens(namespace: str, content: str) -> frozenset[str]:
    tokens: set[str] = set()
    if URL_RE.search(content):
        tokens.add(f"{namespace}_FACT_url")
    if _VERSION_RE.search(content):
        tokens.add(f"{namespace}_FACT_version")
    if _PART_NUMBER_RE.search(content):
        tokens.add(f"{namespace}_FACT_partnumber")
    if _PATH_RE.search(content):
        tokens.add(f"{namespace}_FACT_path")
    if "`" in content:
        tokens.add(f"{namespace}_FACT_code")
    return frozenset(tokens)


def _neighbour_contribution(
    namespace: str, neighbour: MessageRow, opted_out: frozenset[int]
) -> frozenset[str]:
    if neighbour.author_id in opted_out:
        return frozenset()
    return _capped_neighbour_tokens(namespace, neighbour.content) | _fact_shape_tokens(
        namespace, neighbour.content
    )


def build_context_tokens(
    messages: Sequence[MessageRow],
    opted_out: frozenset[int],
    reply_targets: Mapping[int, MessageRow] | None = None,
) -> dict[int, frozenset[str]]:
    """The context token set for every message in one exchange, `messages`
    given in position order (as `infovore.db.batch.exchange_inputs_for_ids`
    already returns them). `reply_targets` resolves a `reply_to_id` that
    points outside this exchange (a batched lookup the caller performs once
    across many exchanges -- see `exchange_context_tokens`); a target inside
    the exchange is resolved from `messages` directly, and a `reply_to_id`
    that resolves nowhere at all is simply omitted. Pure and DB-free, so a
    caller who already has an exchange's messages in hand (e.g. a test, or a
    future caller with its own batching) never needs a connection."""
    by_id = {message.id: message for message in messages}
    targets = reply_targets if reply_targets is not None else {}
    size = len(messages)
    size_bucket = exchange_size_bucket(size)
    result: dict[int, frozenset[str]] = {}
    for index, message in enumerate(messages):
        tokens: set[str] = {position_bucket(index, size), size_bucket}
        if message.reply_to_id is not None:
            tokens.add("CTX_is_reply")

        previous = messages[index - 1] if index > 0 else None
        if previous is not None and previous.author_id not in opted_out:
            tokens |= _neighbour_contribution("PREV", previous, opted_out)
            if previous.content.rstrip().endswith("?"):
                tokens.add("CTX_prev_ends_question")
            if previous.author_id == message.author_id:
                tokens.add("CTX_same_author_prev")

        following = messages[index + 1] if index + 1 < size else None
        if following is not None:
            tokens |= _neighbour_contribution("NEXT", following, opted_out)

        if message.reply_to_id is not None:
            target = by_id.get(message.reply_to_id) or targets.get(message.reply_to_id)
            if target is not None:
                tokens |= _neighbour_contribution("REPLYTO", target, opted_out)

        result[message.id] = frozenset(tokens)
    return result


def exchange_context_tokens(
    conn: sqlite3.Connection,
    exchange_ids: Sequence[int],
    opted_out: frozenset[int],
) -> dict[int, frozenset[str]]:
    """`build_context_tokens`, batched across many exchanges at once
    (issue #141): each exchange's messages are loaded once, in position
    order, via `infovore.db.batch.exchange_inputs_for_ids` (the same
    set-based, chunked loader `infovore.triage.train` uses), and any
    `reply_to_id` pointing outside its own exchange is resolved with one
    more batched `infovore.db.raw.messages_by_ids` call across every such id
    in the whole request, not one query per message."""
    inputs = exchange_inputs_for_ids(conn, exchange_ids)
    known_ids = {message.id for exchange in inputs.values() for message in exchange.messages}
    external_ids = sorted(
        {
            message.reply_to_id
            for exchange in inputs.values()
            for message in exchange.messages
            if message.reply_to_id is not None and message.reply_to_id not in known_ids
        }
    )
    reply_targets = {message.id: message for message in messages_by_ids(conn, external_ids)}

    result: dict[int, frozenset[str]] = {}
    for exchange_id in exchange_ids:
        exchange = inputs.get(exchange_id)
        if exchange is None or not exchange.messages:
            continue
        result.update(build_context_tokens(exchange.messages, opted_out, reply_targets))
    return result


def structural_context_tokens(tokens: frozenset[str]) -> frozenset[str]:
    """Filters a full context-token set (`build_context_tokens`/
    `exchange_context_tokens`'s output for one message) down to the
    `FeatureSet.STRUCTURAL` subset (issue #144): position/size buckets and
    reply-shape flags (`POS_`, `EXSIZE_`, `CTX_`) and a neighbour's
    fact-shape tokens (`*_FACT_*`), dropping every neighbour *word* token
    (`PREV_`/`NEXT_`/`REPLYTO_` followed by a plain lowercase word). A word
    token never contains the literal `_FACT_` marker itself -- word tokens
    are always lower-cased (`_word_tokens`), `_FACT_` is not -- so
    `"_FACT_" in token` unambiguously picks out only the fact-shape
    tokens, never a neighbour's own vocabulary."""
    return frozenset(
        token
        for token in tokens
        if token.startswith(("POS_", "EXSIZE_", "CTX_")) or "_FACT_" in token
    )


def context_tokens_for_feature_set(
    feature_set: FeatureSet, full_context: frozenset[str]
) -> frozenset[str]:
    """`full_context` (one message's unfiltered `build_context_tokens`/
    `exchange_context_tokens` output) narrowed to what `feature_set`
    actually uses -- the single choke point both training
    (`infovore.sift.train.build_message_examples`) and scoring
    (`infovore.sift.train._score_rows`) go through, so `structural`/
    `context` scoring can never drift from how they were trained. `PLAIN`
    always returns no context at all, regardless of `full_context`."""
    if feature_set is FeatureSet.PLAIN:
        return frozenset()
    if feature_set is FeatureSet.STRUCTURAL:
        return structural_context_tokens(full_context)
    return full_context


def message_features(
    message: MessageRow, channel_id: int, context: frozenset[str] = frozenset()
) -> frozenset[str]:
    """The token set `infovore.sift.train` trains and scores on for a single
    message: its own word tokens, structural signals, a `CHAN_<channel_id>`
    token (`channel_id` is taken as a parameter, not read off `message`, so a
    caller scoring many messages from an already-batched query need not
    round-trip through `message.channel_id` -- though in practice the two
    always agree), a message-length bucket, and -- when `context` is given
    (`build_context_tokens`/`exchange_context_tokens`, issue #141) -- that
    message's conversation-context tokens. `context` defaults to empty, so a
    caller that never resolves an exchange (or the ablation report's
    without-context baseline) gets exactly the pre-#141 feature set."""
    return frozenset(
        _word_tokens(message.content)
        | _structural_signals(message)
        | context
        | {f"CHAN_{channel_id}", message_length_bucket(message.content)}
    )
