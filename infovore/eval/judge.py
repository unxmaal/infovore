import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from infovore.db.annotations import Annotation, record_annotation
from infovore.db.batch import exchange_inputs_for_ids
from infovore.db.channel_filter import exclude_channels_clause, excluded_exchange_ids
from infovore.db.exchanges import claimable_condition, exchange_message_ids, get_exchange
from infovore.db.raw import get_channel, messages_by_ids
from infovore.eval.slices import BUILD, GOLD, GOLD_REPEATS, HOLDOUT, REJECTED, slice_ids
from infovore.extract.prompt import permalink
from infovore.rows import MessageRow
from infovore.triage.human import trainable_counts
from infovore.triage.lexicon import load_lexicon, score_lexicon

JUDGE_SCORER = "human_exchange"
# What the page shows and asks IS the labelling function (RULE #309): bump
# this on any change to the instructions, the context shown, or the layout,
# so judgments made under different interfaces are never pooled by accident.
JUDGE_INTERFACE_VERSION = 3
RELEVANT = "relevant"
IRRELEVANT = "irrelevant"
BAD_GROUPING = "bad_grouping"
LABELS = (RELEVANT, IRRELEVANT, BAD_GROUPING)
UNCERTAIN = "uncertain"
LIKELY_IRRELEVANT = "likely-irrelevant"
LIKELY_IRRELEVANT_POOL = 2000
UNCERTAIN_LIMIT = 200
RELEVANCE_TARGET = 200
CONTEXT_SIZE = 3
_REF = "judge:"


class UnknownQueueItemError(LookupError):
    pass


class NotInExchangeError(ValueError):
    pass


class InvalidLabelError(ValueError):
    pass


@dataclass(frozen=True)
class QueueItem:
    exchange_id: int
    slice_name: str
    position: int

    @property
    def source_ref(self) -> str:
        return f"{_REF}{self.slice_name}:{self.position}"


QueueBuilder = Callable[[sqlite3.Connection], list[QueueItem]]


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
    label: str | None


@dataclass(frozen=True)
class Agreement:
    exchanges: int
    agreed: int

    @property
    def rate(self) -> float | None:
        return self.agreed / self.exchanges if self.exchanges else None


@dataclass(frozen=True)
class SliceProgress:
    name: str
    done: int
    total: int


def frozen_queue(conn: sqlite3.Connection) -> list[QueueItem]:
    """Gold in frozen order, its repeats right after, unannounced, then the
    rest of s1. Derived from the frozen slices, so it cannot drift."""
    gold = slice_ids(conn, GOLD)
    in_gold = set(gold)
    plan = [
        (GOLD, gold),
        (GOLD_REPEATS, slice_ids(conn, GOLD_REPEATS)),
        (BUILD, [i for i in slice_ids(conn, BUILD) if i not in in_gold]),
    ]
    return [
        QueueItem(exchange_id, name, position)
        for name, ids in plan
        for position, exchange_id in enumerate(ids, start=1)
    ]


_LATEST_SCORE = (
    "(SELECT a.score FROM annotations a WHERE a.subject_kind = 'exchange'"
    " AND a.subject_id = exchanges.id AND a.scorer = ? AND a.reproducibility = 'derived'"
    " ORDER BY a.scorer_version DESC, a.id DESC LIMIT 1)"
)


def uncertain_queue(
    max_retries: int,
    min_p_lore: float,
    exclude_channels: frozenset[str],
    scorer: str | None = None,
) -> QueueBuilder:
    """Unjudged exchanges extraction could still claim, the ones the scorer is
    least sure about first (uncertainty sampling). The ref is the exchange id,
    since a rank shifts as items are judged and would alias another exchange.
    `scorer` reads that scorer's latest derived annotation; None reads p_lore."""

    def build(conn: sqlite3.Connection) -> list[QueueItem]:
        condition, params = claimable_condition(max_retries, None, min_p_lore, exclude_channels)
        score, score_params = ("p_lore", []) if scorer is None else (_LATEST_SCORE, [scorer])
        rows = conn.execute(
            f"SELECT id FROM (SELECT id, {score} AS s FROM exchanges WHERE {condition})"
            " WHERE s IS NOT NULL"
            " AND id NOT IN (SELECT subject_id FROM annotations"
            " WHERE subject_kind = 'exchange' AND scorer = ?)"
            " ORDER BY ABS(s - 0.5), id LIMIT ?",
            (*score_params, *params, JUDGE_SCORER, UNCERTAIN_LIMIT),
        ).fetchall()
        return [QueueItem(row["id"], UNCERTAIN, row["id"]) for row in rows]

    return build


_UNJUDGED = (
    "id NOT IN (SELECT subject_id FROM annotations WHERE subject_kind = 'exchange' AND scorer = ?)"
)


def c1_queue(exclude_channels: frozenset[str]) -> QueueBuilder:
    def build(conn: sqlite3.Connection) -> list[QueueItem]:
        return _c1_items(conn, exclude_channels)

    return build


def _c1_items(conn: sqlite3.Connection, exclude_channels: frozenset[str]) -> list[QueueItem]:
    clause, params = exclude_channels_clause("exchanges.channel_id", exclude_channels)
    allowed = {
        row["id"] for row in conn.execute(f"SELECT id FROM exchanges WHERE 1 = 1{clause}", params)
    }
    judged = {
        row["subject_id"]
        for row in conn.execute(
            "SELECT subject_id FROM annotations WHERE subject_kind = 'exchange' AND scorer = ?",
            (JUDGE_SCORER,),
        )
    }
    return [
        QueueItem(exchange_id, REJECTED, position)
        for position, exchange_id in enumerate(slice_ids(conn, REJECTED), start=1)
        if exchange_id not in judged and exchange_id in allowed
    ]


def likely_irrelevant_queue(exclude_channels: frozenset[str]) -> QueueBuilder:
    def build(conn: sqlite3.Connection) -> list[QueueItem]:
        return _likely_irrelevant_items(conn, exclude_channels)

    return build


def _likely_irrelevant_items(
    conn: sqlite3.Connection, exclude_channels: frozenset[str]
) -> list[QueueItem]:
    """Unjudged exchanges outside s2 and gold, lowest lexicon share first then lowest p_lore;
    only the LIKELY_IRRELEVANT_POOL lowest p_lore are scored, so no whole-corpus pass."""
    clause, excl_params = exclude_channels_clause("exchanges.channel_id", exclude_channels)
    rows = conn.execute(
        f"SELECT id, p_lore FROM exchanges WHERE p_lore IS NOT NULL AND {_UNJUDGED}"
        f" AND id NOT IN (SELECT exchange_id FROM eval_slices WHERE name IN (?, ?)){clause}"
        " ORDER BY p_lore, id LIMIT ?",
        (JUDGE_SCORER, HOLDOUT, GOLD, *excl_params, LIKELY_IRRELEVANT_POOL),
    ).fetchall()
    lexicon = load_lexicon()
    inputs = exchange_inputs_for_ids(conn, [row["id"] for row in rows])
    ranked = sorted(
        (score_lexicon(lexicon, inputs[row["id"]].messages).share, row["p_lore"], row["id"])
        for row in rows
    )
    return [
        QueueItem(exchange_id, LIKELY_IRRELEVANT, exchange_id)
        for _, _, exchange_id in ranked[:UNCERTAIN_LIMIT]
    ]


def _item(queue: list[QueueItem], index: int) -> QueueItem:
    if not 0 <= index < len(queue):
        raise UnknownQueueItemError(f"no queue item {index}; the queue has {len(queue)}")
    return queue[index]


def _latest(conn: sqlite3.Connection, exchange_id: int, ref_like: str) -> str | None:
    row = conn.execute(
        "SELECT label FROM annotations WHERE scorer = ? AND subject_kind = 'exchange'"
        " AND subject_id = ? AND source_ref LIKE ? ORDER BY id DESC LIMIT 1",
        (JUDGE_SCORER, exchange_id, ref_like),
    ).fetchone()
    return None if row is None else str(row["label"])


def _judged_refs(conn: sqlite3.Connection) -> set[str]:
    return {
        row["source_ref"]
        for row in conn.execute(
            "SELECT DISTINCT source_ref FROM annotations"
            " WHERE scorer = ? AND subject_kind = 'exchange'",
            (JUDGE_SCORER,),
        )
    }


def _message(row: MessageRow, guild_id: int, is_context: bool) -> JudgeMessage:
    return JudgeMessage(
        id=row.id,
        author=row.author_name_at_time,
        created_at=row.created_at.isoformat(),
        content=row.content,
        link=permalink(guild_id, row.channel_id, row.id),
        is_context=is_context,
    )


def exchange_view(conn: sqlite3.Connection, queue: list[QueueItem], index: int) -> ExchangeView:
    item = _item(queue, index)
    exchange = get_exchange(conn, item.exchange_id)
    assert exchange is not None and exchange.id is not None
    channel = get_channel(conn, exchange.channel_id)
    own = messages_by_ids(conn, exchange_message_ids(conn, exchange.id))
    guild_id = own[0].guild_id if own else 0
    context = []
    if exchange.parent_exchange_id is not None:
        parent_ids = exchange_message_ids(conn, exchange.parent_exchange_id)[-CONTEXT_SIZE:]
        context = messages_by_ids(conn, parent_ids)
    return ExchangeView(
        index=index,
        total=len(queue),
        exchange_id=exchange.id,
        channel=channel.name if channel is not None else str(exchange.channel_id),
        messages=tuple(
            [_message(m, guild_id, True) for m in context]
            + [_message(m, guild_id, False) for m in own]
        ),
        label=_latest(conn, item.exchange_id, item.source_ref),
    )


def submit(
    conn: sqlite3.Connection,
    queue: list[QueueItem],
    index: int,
    exchange_id: int,
    label: str,
    at: datetime,
) -> None:
    """Append one exchange-level judgment. The exchange id rides along so a
    queue that moved under the page (uncertain mode drops what is judged)
    cannot credit the label to a different exchange."""
    item = _item(queue, index)
    if label not in LABELS:
        raise InvalidLabelError(f"label must be one of {', '.join(LABELS)}")
    if item.exchange_id != exchange_id:
        raise NotInExchangeError(f"queue item {index} is not exchange {exchange_id}")
    if not exchange_message_ids(conn, exchange_id):
        raise NotInExchangeError(f"exchange {exchange_id} has no messages")
    with conn:
        record_annotation(
            conn,
            Annotation(
                subject_kind="exchange",
                subject_id=exchange_id,
                scorer=JUDGE_SCORER,
                scorer_version=JUDGE_INTERFACE_VERSION,
                reproducibility="recorded",
                label=label,
                source_ref=item.source_ref,
            ),
            at,
        )


def progress(conn: sqlite3.Connection, queue: list[QueueItem]) -> tuple[int, int]:
    judged = _judged_refs(conn)
    return sum(1 for item in queue if item.source_ref in judged), len(queue)


def queue_stats(
    conn: sqlite3.Connection,
    queue_name: str,
    queue: list[QueueItem],
    exclude_channels: frozenset[str] = frozenset(),
) -> dict[str, int | str]:
    """Header numbers for a queue rebuilt after every submit, where a position
    within it means nothing: judged is counted from the labels themselves."""
    judged = conn.execute(
        "SELECT COUNT(DISTINCT subject_id) FROM annotations"
        " WHERE scorer = ? AND subject_kind = 'exchange' AND source_ref LIKE ?",
        (JUDGE_SCORER, f"{_REF}{queue_name}:%"),
    ).fetchone()[0]
    needed = trainable_needed(conn, exclude_channels)
    return {
        "queue": queue_name,
        "judged": judged,
        "queued": len(queue),
        "relevant_needed": needed[RELEVANT],
        "irrelevant_needed": needed[IRRELEVANT],
        "target": RELEVANCE_TARGET,
    }


def first_unjudged(conn: sqlite3.Connection, queue: list[QueueItem]) -> int | None:
    judged = _judged_refs(conn)
    for index, item in enumerate(queue):
        if item.source_ref not in judged:
            return index
    return None


def slice_progress(conn: sqlite3.Connection) -> list[SliceProgress]:
    judged = _judged_refs(conn)
    totals: dict[str, list[int]] = {}
    for item in frozen_queue(conn):
        entry = totals.setdefault(item.slice_name, [0, 0])
        entry[0] += item.source_ref in judged
        entry[1] += 1
    return [SliceProgress(name, done, total) for name, (done, total) in totals.items()]


def uncertain_judged(conn: sqlite3.Connection) -> int:
    return sum(1 for ref in _judged_refs(conn) if ref.startswith(f"{_REF}{UNCERTAIN}:"))


def label_counts(
    conn: sqlite3.Connection, exclude_channels: frozenset[str] = frozenset()
) -> dict[str, int]:
    """Exchanges per label, newest judgment wins, repeats excluded so the
    repeated exchanges are not double-weighted."""
    counts = dict.fromkeys(LABELS, 0)
    repeats = f"{_REF}{GOLD_REPEATS}:%"
    denied = excluded_exchange_ids(conn, exclude_channels)
    for row in conn.execute(
        "SELECT label, subject_id FROM annotations a WHERE scorer = ?"
        " AND subject_kind = 'exchange' AND source_ref NOT LIKE ?"
        " AND id = (SELECT MAX(id) FROM annotations WHERE scorer = a.scorer"
        " AND subject_kind = 'exchange' AND subject_id = a.subject_id"
        " AND source_ref NOT LIKE ?)",
        (JUDGE_SCORER, repeats, repeats),
    ):
        if row["subject_id"] not in denied:
            counts[row["label"]] += 1
    return counts


def labels_needed(counts: dict[str, int]) -> dict[str, int]:
    """How many more of each class reach SpamAssassin's Bayes minimum."""
    return {label: max(0, RELEVANCE_TARGET - counts[label]) for label in (RELEVANT, IRRELEVANT)}


def trainable_needed(
    conn: sqlite3.Connection, exclude_channels: frozenset[str] = frozenset()
) -> dict[str, int]:
    relevant, irrelevant = trainable_counts(conn, exclude_channels)
    return labels_needed({RELEVANT: relevant, IRRELEVANT: irrelevant})


def self_agreement(conn: sqlite3.Connection) -> Agreement:
    """Exchange-level agreement between the two showings of each repeated
    gold exchange: the noise floor no scorer can be measured below (#190)."""
    exchanges = agreed = 0
    for exchange_id in slice_ids(conn, GOLD_REPEATS):
        first = _latest(conn, exchange_id, f"{_REF}{GOLD}:%")
        second = _latest(conn, exchange_id, f"{_REF}{GOLD_REPEATS}:%")
        if first is None or second is None:
            continue
        exchanges += 1
        agreed += first == second
    return Agreement(exchanges, agreed)
