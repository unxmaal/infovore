import re
import sqlite3
from collections.abc import Sequence

_RANGE = re.compile(r"^(\d+)-(\d+)$")
_PLAIN = re.compile(r"^\d+$")


class InvalidRunSelectorError(Exception):
    def __init__(self, token: str) -> None:
        super().__init__(f"invalid run selector: {token!r}")
        self.token = token


class NoTrialBatchError(Exception):
    pass


def parse_run_tokens(tokens: Sequence[str]) -> list[int]:
    """Parse run-id tokens into a sorted, de-duplicated list of run ids.

    Each token is either a bare run id ("220") or an inclusive range
    ("220-419"). Shared by `label --from-runs`, `review --run-ids`, and
    `probe --run-id`.
    """
    ids: set[int] = set()
    for token in tokens:
        range_match = _RANGE.match(token)
        if range_match is not None:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start > end:
                raise InvalidRunSelectorError(token)
            ids.update(range(start, end + 1))
            continue
        if _PLAIN.match(token) is None:
            raise InvalidRunSelectorError(token)
        ids.add(int(token))
    return sorted(ids)


def latest_trial_batch_run_ids(conn: sqlite3.Connection) -> list[int]:
    """Return every run id from the most recent `extract --mode trial` invocation."""
    row = conn.execute(
        "SELECT batch_id FROM extraction_runs"
        " WHERE mode = 'trial' AND batch_id IS NOT NULL"
        " ORDER BY batch_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise NoTrialBatchError()
    rows = conn.execute(
        "SELECT id FROM extraction_runs WHERE mode = 'trial' AND batch_id = ? ORDER BY id",
        (row["batch_id"],),
    ).fetchall()
    return [r["id"] for r in rows]


def resolve_run_selector(conn: sqlite3.Connection, tokens: Sequence[str] | None) -> list[int]:
    """Resolve a run selector: explicit tokens, or the latest trial batch when empty."""
    if not tokens:
        return latest_trial_batch_run_ids(conn)
    return parse_run_tokens(tokens)
