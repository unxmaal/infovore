import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from infovore.config import ConfigError

ENV: Final = "INFOVORE_REPUTATION_PEOPLE"


@dataclass(frozen=True)
class People:
    accounts: dict[int, str]
    banned: frozenset[str]
    members: dict[str, tuple[int, ...]]

    def person(self, author_id: int) -> str:
        return self.accounts.get(author_id, str(author_id))

    def representative(self, person: str) -> int:
        return min(self.members[person]) if person in self.members else int(person)

    def is_banned(self, person: str) -> bool:
        return person in self.banned


def people_path(env: Mapping[str, str], flag: str | None) -> Path | None:
    if flag is not None:
        return Path(flag)
    return Path(env[ENV]) if env.get(ENV) else None


def load_people(path: Path | None) -> People:
    if path is None or not path.exists():
        return People({}, frozenset(), {})
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"people file: invalid TOML: {error}") from error
    extra = set(data) - {"person"}
    if extra:
        raise ConfigError(f"people file: unexpected keys {sorted(extra)}")
    entries = data.get("person", [])
    if not isinstance(entries, list):
        raise ConfigError("people file: 'person' must be an array of tables")
    accounts: dict[int, str] = {}
    members: dict[str, tuple[int, ...]] = {}
    banned: set[str] = set()
    for entry in entries:
        key = _identifier(entry, members)
        listed = _accounts(entry, key, accounts)
        members[key] = tuple(sorted(listed))
        flag = entry.get("banned", False)
        if not isinstance(flag, bool):
            raise ConfigError("people file: 'banned' must be a boolean")
        if flag:
            banned.add(key)
    return People(accounts, frozenset(banned), members)


def _identifier(entry: Mapping[str, Any], seen: Mapping[str, object]) -> str:
    key = entry.get("id")
    if not isinstance(key, str) or not key:
        raise ConfigError("people file: every person needs a non-empty string id")
    if key in seen:
        raise ConfigError("people file: duplicate person id")
    return key


def _accounts(entry: Mapping[str, Any], key: str, taken: dict[int, str]) -> list[int]:
    listed = entry.get("accounts", [])
    if not isinstance(listed, list):
        raise ConfigError("people file: 'accounts' must be a list of integers")
    for account in listed:
        if isinstance(account, bool) or not isinstance(account, int):
            raise ConfigError("people file: 'accounts' must be a list of integers")
        if account in taken:
            raise ConfigError("people file: an account is listed more than once")
        taken[account] = key
    return listed
