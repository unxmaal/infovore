import sqlite3
from datetime import datetime
from typing import Final

from infovore.db.codec import to_db_time

RATINGS: Final = (0, 1, 2, 3)


def record_rating(conn: sqlite3.Connection, author_id: int, rating: int, now: datetime) -> None:
    if rating not in RATINGS:
        raise ValueError(f"rating must be one of {RATINGS}, got {rating!r}")
    conn.execute(
        "INSERT INTO author_ratings (author_id, rating, rated_at) VALUES (?, ?, ?)",
        (author_id, rating, to_db_time(now)),
    )


def ratings(conn: sqlite3.Connection) -> dict[int, int]:
    rows = conn.execute("SELECT author_id, rating FROM current_author_ratings")
    return {row[0]: row[1] for row in rows}
