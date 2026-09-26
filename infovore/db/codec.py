from datetime import UTC, datetime
from typing import overload


@overload
def to_db_time(value: datetime) -> str: ...
@overload
def to_db_time(value: None) -> None: ...
def to_db_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("datetime must carry a timezone")
    return value.astimezone(UTC).isoformat()


@overload
def from_db_time(value: str) -> datetime: ...
@overload
def from_db_time(value: None) -> None: ...
def from_db_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)
