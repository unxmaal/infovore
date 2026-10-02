import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from infovore.db.annotations import Annotation, record_annotation
from infovore.db.exchanges import exchange_message_ids, get_exchange
from infovore.db.raw import get_channel, messages_by_ids
from infovore.eval.slices import GOLD, GOLD_REPEATS, slice_ids
from infovore.extract.prompt import permalink
from infovore.rows import MessageRow

JUDGE_SCORER = "human_fact"
# What the page shows and asks IS the labelling function (RULE #309): bump
# this on any change to the instructions, the context shown, or the layout,
# so judgments made under different interfaces are never pooled by accident.
JUDGE_INTERFACE_VERSION = 1
FACT = "fact"
NO_FACT = "no_fact"
CONTEXT_SIZE = 3


class UnknownQueueItemError(LookupError):
    pass


class NotInExchangeError(ValueError):
    pass


@dataclass(frozen=True)
class QueueItem:
    exchange_id: int
    pass_number: int

    @property
    def source_ref(self) -> str:
        return f"judge:{GOLD}:{self.exchange_id}:pass{self.pass_number}"


@dataclass(frozen=True)
class JudgeMessage:
    id: int
    author: str
    created_at: str
    content: str
    link: str
    is_context: bool


@dataclass(frozen=True)
class ExchangeView:
    index: int
    total: int
    exchange_id: int
    channel: str
    messages: tuple[JudgeMessage, ...]
    marked: frozenset[int]
    done: bool


@dataclass(frozen=True)
class Agreement:
    exchanges: int
    messages: int
    agreed: int

    @property
    def rate(self) -> float | None:
        return self.agreed / self.messages if self.messages else None


def judging_queue(conn: sqlite3.Connection) -> list[QueueItem]:
    """The gold set in frozen order, then the repeats as a second pass,
    unannounced. Derived from the frozen slices, so it cannot drift."""
    return [QueueItem(i, 1) for i in slice_ids(conn, GOLD)] + [
        QueueItem(i, 2) for i in slice_ids(conn, GOLD_REPEATS)
    ]


def _item(conn: sqlite3.Connection, index: int) -> tuple[QueueItem, int]:
    queue = judging_queue(conn)
    if not 0 <= index < len(queue):
        raise UnknownQueueItemError(f"no queue item {index}; the queue has {len(queue)}")
    return queue[index], len(queue)


def _labels(conn: sqlite3.Connection, source_ref: str) -> dict[int, str]:
    """Latest judgment per message for one pass. Annotations are
    append-only, so changing a mark appends and the newest row wins."""
    latest: dict[int, str] = {}
    for row in conn.execute(
        "SELECT subject_id, label FROM annotations WHERE scorer = ? AND source_ref = ?"
        " AND subject_kind = 'message' ORDER BY id",
        (JUDGE_SCORER, source_ref),
    ):
        latest[row["subject_id"]] = row["label"]
    return latest


def _message(row: MessageRow, guild_id: int, is_context: bool) -> JudgeMessage:
    return JudgeMessage(
        id=row.id,
        author=row.author_name_at_time,
        created_at=row.created_at.isoformat(),
        content=row.content,
        link=permalink(guild_id, row.channel_id, row.id),
        is_context=is_context,
    )


def exchange_view(conn: sqlite3.Connection, index: int) -> ExchangeView:
    item, total = _item(conn, index)
    exchange = get_exchange(conn, item.exchange_id)
    assert exchange is not None and exchange.id is not None
    channel = get_channel(conn, exchange.channel_id)
    own = messages_by_ids(conn, exchange_message_ids(conn, exchange.id))
    guild_id = own[0].guild_id if own else 0
    context = []
    if exchange.parent_exchange_id is not None:
        parent_ids = exchange_message_ids(conn, exchange.parent_exchange_id)[-CONTEXT_SIZE:]
        context = messages_by_ids(conn, parent_ids)
    labels = _labels(conn, item.source_ref)
    return ExchangeView(
        index=index,
        total=total,
        exchange_id=exchange.id,
        channel=channel.name if channel is not None else str(exchange.channel_id),
        messages=tuple(
            [_message(m, guild_id, True) for m in context]
            + [_message(m, guild_id, False) for m in own]
        ),
        marked=frozenset(mid for mid, label in labels.items() if label == FACT),
        done=bool(labels),
    )


def submit(conn: sqlite3.Connection, index: int, fact_ids: Iterable[int], at: datetime) -> int:
    """Record a judgment for EVERY message of the exchange: the marked ones
    as fact, the rest as no_fact. Writing the negatives explicitly is the
    point: an exchange nobody opened stays unjudged instead of silently
    reading as "no facts" (the sift import defect, FINDING #173)."""
    item, _ = _item(conn, index)
    own = set(exchange_message_ids(conn, item.exchange_id))
    if not own:
        # Nothing would be written, so the item would never count as done and
        # the page would offer it forever. Refuse rather than no-op.
        raise NotInExchangeError(f"exchange {item.exchange_id} has no messages")
    facts = set(fact_ids)
    stray = facts - own
    if stray:
        raise NotInExchangeError(f"messages {sorted(stray)} are not in exchange {item.exchange_id}")
    with conn:
        for message_id in sorted(own):
            record_annotation(
                conn,
                Annotation(
                    subject_kind="message",
                    subject_id=message_id,
                    scorer=JUDGE_SCORER,
                    scorer_version=JUDGE_INTERFACE_VERSION,
                    reproducibility="recorded",
                    label=FACT if message_id in facts else NO_FACT,
                    source_ref=item.source_ref,
                ),
                at,
            )
    return len(own)


def progress(conn: sqlite3.Connection) -> tuple[int, int]:
    queue = judging_queue(conn)
    done = sum(1 for item in queue if _labels(conn, item.source_ref))
    return done, len(queue)


def self_agreement(conn: sqlite3.Connection) -> Agreement:
    """Message-level agreement between the two passes over each repeated
    exchange: the noise floor no extractor can be measured below (#190)."""
    exchanges = messages = agreed = 0
    for exchange_id in slice_ids(conn, GOLD_REPEATS):
        first = _labels(conn, QueueItem(exchange_id, 1).source_ref)
        second = _labels(conn, QueueItem(exchange_id, 2).source_ref)
        if not first or not second:
            continue
        exchanges += 1
        for message_id in first.keys() & second.keys():
            messages += 1
            agreed += first[message_id] == second[message_id]
    return Agreement(exchanges, messages, agreed)


def fact_counts(conn: sqlite3.Connection) -> tuple[int, int]:
    """Messages judged fact and no_fact on the first pass, latest mark wins."""
    facts = no_facts = 0
    for item in judging_queue(conn):
        if item.pass_number != 1:
            continue
        for label in _labels(conn, item.source_ref).values():
            if label == FACT:
                facts += 1
            else:
                no_facts += 1
    return facts, no_facts


def first_unjudged(conn: sqlite3.Connection) -> int | None:
    """Where to resume: the first queue position with no judgment yet."""
    for index, item in enumerate(judging_queue(conn)):
        if not _labels(conn, item.source_ref):
            return index
    return None
