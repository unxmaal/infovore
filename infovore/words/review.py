import sqlite3
from collections import Counter
from collections.abc import Iterator, Sequence

from wordfreq import top_n_list

from infovore.db.batch import exchange_inputs_for_ids
from infovore.db.reviewed_words import approved_words, decided_words
from infovore.triage.lexicon import load_lexicon, tokens

DEFAULT_TOP_N = 10000
BATCH = 2000
_LATEST_CASCADE = (
    "SELECT a.subject_id FROM annotations a"
    " JOIN (SELECT subject_id, MAX(id) AS last FROM annotations"
    "  WHERE subject_kind = 'exchange' AND label IS NOT NULL"
    "  AND scorer LIKE 'relevance\\_%' ESCAPE '\\' GROUP BY subject_id) l ON l.last = a.id"
    " JOIN current_exchanges e ON e.id = a.subject_id"
    " WHERE a.scorer = 'relevance_residue' AND a.label = 'residue' ORDER BY a.subject_id"
)


class UnknownWordError(ValueError):
    pass


def undecided_ids(conn: sqlite3.Connection) -> list[int]:
    return [row["subject_id"] for row in conn.execute(_LATEST_CASCADE)]


def _conversation_tokens(conn: sqlite3.Connection) -> Iterator[list[str]]:
    ids = undecided_ids(conn)
    for start in range(0, len(ids), BATCH):
        inputs = exchange_inputs_for_ids(conn, ids[start : start + BATCH])
        for exchange in inputs.values():
            yield tokens("\n".join(m.content for m in exchange.messages))


def build_pile(conn: sqlite3.Connection) -> Counter[str]:
    pile: Counter[str] = Counter()
    for found in _conversation_tokens(conn):
        pile.update(found)
    return pile


def common_words(top_n: int) -> frozenset[str]:
    return frozenset(top_n_list("en", top_n))


def candidates(conn: sqlite3.Connection, top_n: int = DEFAULT_TOP_N) -> list[tuple[str, int]]:
    terms = load_lexicon(conn).terms
    skip = common_words(top_n) | terms | decided_words(conn)
    kept = [
        (word, count)
        for word, count in build_pile(conn).items()
        if word not in skip and not (word.endswith("s") and word[:-1] in terms)
    ]
    return sorted(kept, key=lambda item: (-item[1], item[0]))


def gain(conn: sqlite3.Connection, approved: frozenset[str]) -> tuple[int, int]:
    total = hits = 0
    for found in _conversation_tokens(conn):
        total += 1
        hits += any(t in approved or (t.endswith("s") and t[:-1] in approved) for t in found)
    return total, hits


def reviewed_gain(conn: sqlite3.Connection) -> tuple[int, int]:
    return gain(conn, approved_words(conn))


class ReviewQueue:
    def __init__(self, items: Sequence[tuple[str, int]]) -> None:
        self.items = list(items)
        self.known = {word for word, _ in self.items}
        self.decided: set[str] = set()
        self.start = 0

    @property
    def remaining(self) -> int:
        return len(self.items) - len(self.decided)

    def page(self, size: int) -> list[tuple[str, int]]:
        found: list[tuple[str, int]] = []
        for index in range(self.start, len(self.items)):
            if len(found) == size:
                break
            if self.items[index][0] not in self.decided:
                found.append(self.items[index])
        return found

    def check(self, words: Sequence[str]) -> None:
        for word in words:
            if word not in self.known:
                raise UnknownWordError(f"unknown word: {word}")

    def mark_decided(self, words: Sequence[str]) -> None:
        self.check(words)
        self.decided.update(words)
        while self.start < len(self.items) and self.items[self.start][0] in self.decided:
            self.start += 1
