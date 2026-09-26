from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

DEFAULT_QUIET_GAP_MINUTES = 30
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_EXCHANGE_MAX_MESSAGES = 50
DEFAULT_OPT_OUT_ROLE_NAME = "no-archive"
DEFAULT_INCLUDE_BOT_MESSAGES = False


class Stage(StrEnum):
    EXTRACT = "extract"
    PROBE = "probe"
    JUDGE = "judge"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class StageSettings:
    backend: str
    model: str
    concurrency: int
    timeout_seconds: float
    options: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    discord_token: str = field(repr=False)
    guild_id: int
    channel_ids: tuple[int, ...]
    db_path: Path
    scratch_dir: Path
    stages: Mapping[Stage, StageSettings]
    quiet_gap_minutes: int = DEFAULT_QUIET_GAP_MINUTES
    batch_size: int = DEFAULT_BATCH_SIZE
    max_retries: int = DEFAULT_MAX_RETRIES
    exchange_max_messages: int = DEFAULT_EXCHANGE_MAX_MESSAGES
    opt_out_role_name: str = DEFAULT_OPT_OUT_ROLE_NAME
    include_bot_messages: bool = DEFAULT_INCLUDE_BOT_MESSAGES
