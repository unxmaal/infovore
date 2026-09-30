import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from infovore.source.protocol import DiscordSource
from infovore.triage.rules import DEFAULT_RULES, RulesError, TriageRules, load_rules

DEFAULT_QUIET_GAP_MINUTES = 30
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_EXCHANGE_MAX_MESSAGES = 50
DEFAULT_OPT_OUT_ROLE_NAME = "no-archive"
DEFAULT_INCLUDE_BOT_MESSAGES = False
DEFAULT_SCRATCH_DIR = "scratch"
DEFAULT_STAGE_CONCURRENCY = 2
DEFAULT_PROBE_BATCH_SIZE = 10
DEFAULT_STAGE_TIMEOUT_SECONDS = 60.0
DEFAULT_TRIAGE_MIN_SCORE = 0.3
DEFAULT_TRIAGE_MIN_P_LORE = 0.5

# Worker processes for triage's CPU-bound scoring (issue #113): `p_lore`,
# rule-based `score_exchange`, and the `--suggest-terms` corpus
# document-frequency scan. `1` means "no process pool" (in-process, what the
# test suite defaults to); otherwise defaults to every core the machine has.
DEFAULT_WORKERS = os.cpu_count() or 1

STAGE_ENV_KEYS = ("BACKEND", "MODEL", "CONCURRENCY", "TIMEOUT")


class Stage(StrEnum):
    EXTRACT = "extract"
    PROBE = "probe"
    JUDGE = "judge"


class SourceKind(StrEnum):
    DISCORD = "discord"
    EXPORT = "export"


class ConfigError(Exception):
    pass


DEFAULT_STAGE_BACKENDS = {
    Stage.EXTRACT: ("claude_cli", "sonnet"),
    Stage.PROBE: ("claude_cli", "sonnet"),
    Stage.JUDGE: ("claude_cli", "haiku"),
}


@dataclass(frozen=True)
class StageSettings:
    backend: str
    model: str
    concurrency: int
    timeout_seconds: float
    options: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    db_path: Path
    scratch_dir: Path
    stages: Mapping[Stage, StageSettings]
    source: SourceKind = SourceKind.DISCORD
    discord_token: str = field(default="", repr=False)
    guild_id: int | None = None
    channel_ids: tuple[int, ...] = ()
    export_dir: Path | None = None
    quiet_gap_minutes: int = DEFAULT_QUIET_GAP_MINUTES
    batch_size: int = DEFAULT_BATCH_SIZE
    max_retries: int = DEFAULT_MAX_RETRIES
    exchange_max_messages: int = DEFAULT_EXCHANGE_MAX_MESSAGES
    opt_out_role_name: str = DEFAULT_OPT_OUT_ROLE_NAME
    include_bot_messages: bool = DEFAULT_INCLUDE_BOT_MESSAGES
    triage_min_score: float = DEFAULT_TRIAGE_MIN_SCORE
    triage_min_p_lore: float = DEFAULT_TRIAGE_MIN_P_LORE
    triage_rules: TriageRules = DEFAULT_RULES
    workers: int = DEFAULT_WORKERS
    exclude_channels: frozenset[str] = field(default_factory=frozenset)


def _parse_int(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_bool(raw: str) -> bool | None:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _require(raw: str | None, name: str, errors: list[str]) -> str:
    if not raw:
        errors.append(f"{name} is required")
        return ""
    return raw


def _require_positive_int(raw: str | None, name: str, errors: list[str]) -> int:
    text = _require(raw, name, errors)
    if not text:
        return 0
    return _positive_int(text, name, errors)


def _positive_int(text: str, name: str, errors: list[str]) -> int:
    parsed = _parse_int(text)
    if parsed is None:
        errors.append(f"{name} must be an integer")
        return 0
    if parsed <= 0:
        errors.append(f"{name} must be positive")
        return 0
    return parsed


def _optional_positive_int(raw: str | None, default: int, name: str, errors: list[str]) -> int:
    if raw is None:
        return default
    return _positive_int(raw, name, errors) or default


def _optional_positive_float(
    raw: str | None, default: float, name: str, errors: list[str]
) -> float:
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        errors.append(f"{name} must be a number")
        return default
    if parsed <= 0:
        errors.append(f"{name} must be positive")
        return default
    return parsed


def _optional_unit_float(raw: str | None, default: float, name: str, errors: list[str]) -> float:
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        errors.append(f"{name} must be a number")
        return default
    if not 0.0 <= parsed <= 1.0:
        errors.append(f"{name} must be between 0 and 1")
        return default
    return parsed


def _optional_bool(raw: str | None, default: bool, name: str, errors: list[str]) -> bool:
    if raw is None:
        return default
    parsed = _parse_bool(raw)
    if parsed is None:
        errors.append(f"{name} must be a boolean")
        return default
    return parsed


def _parse_channel_ids(raw: str | None, errors: list[str]) -> tuple[int, ...]:
    text = raw or ""
    if not text.strip():
        return ()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return ()
    ids: list[int] = []
    bad = False
    for part in parts:
        parsed = _parse_int(part)
        if parsed is None or parsed <= 0:
            bad = True
            continue
        ids.append(parsed)
    if bad:
        errors.append("INFOVORE_CHANNEL_IDS must be a comma-separated list of positive integers")
        return ()
    return tuple(ids)


def normalize_channel_names(raw: str | None) -> frozenset[str]:
    """Comma-separated channel names -> a lowercased, `#`-stripped set
    (issue #138): shared between `INFOVORE_EXCLUDE_CHANNELS` and `sift
    export`/`sift serve --new`'s `--channels`, so the same string normalizes
    the same way whether it names a denylist or an include filter. Any
    non-empty name is accepted here; unknown-channel validation (against
    `infovore.db.channel_filter.known_channel_names`) is the caller's job,
    since only some callers have a database connection to validate against."""
    text = raw or ""
    names: set[str] = set()
    for part in text.split(","):
        name = part.strip()
        if name.startswith("#"):
            name = name[1:].strip()
        if name:
            names.add(name.lower())
    return frozenset(names)


def _optional_guild_id(raw: str | None, errors: list[str]) -> int | None:
    if raw is None or not raw.strip():
        return None
    return _positive_int(raw, "INFOVORE_GUILD_ID", errors) or None


def resolve_guild_id(settings: Settings, source: DiscordSource) -> int:
    if settings.guild_id is not None:
        return settings.guild_id
    guild_ids = getattr(source, "guild_ids", None)
    ids: frozenset[int] = guild_ids() if guild_ids is not None else frozenset()
    if len(ids) == 1:
        return next(iter(ids))
    raise ConfigError("INFOVORE_GUILD_ID is required: it could not be inferred from the export")


def _stage_options(env: Mapping[str, str], prefix: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for key in sorted(env):
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix in STAGE_ENV_KEYS:
            continue
        options[suffix.lower()] = env[key]
    return options


def _parse_stage(env: Mapping[str, str], stage: Stage, errors: list[str]) -> StageSettings:
    prefix = f"INFOVORE_{stage.value.upper()}_"
    default_backend, default_model = DEFAULT_STAGE_BACKENDS[stage]
    backend = env.get(prefix + "BACKEND", default_backend)
    model = env.get(prefix + "MODEL", default_model)
    concurrency = _optional_positive_int(
        env.get(prefix + "CONCURRENCY"), DEFAULT_STAGE_CONCURRENCY, f"{prefix}CONCURRENCY", errors
    )
    timeout_seconds = _optional_positive_float(
        env.get(prefix + "TIMEOUT"),
        DEFAULT_STAGE_TIMEOUT_SECONDS,
        f"{prefix}TIMEOUT",
        errors,
    )
    return StageSettings(
        backend=backend,
        model=model,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        options=_stage_options(env, prefix),
    )


def _load_triage_rules(raw: str | None, errors: list[str]) -> TriageRules:
    if raw is None or not raw.strip():
        return DEFAULT_RULES
    try:
        return load_rules(raw)
    except RulesError as error:
        errors.append(str(error))
        return DEFAULT_RULES


def _parse_source_kind(raw: str | None, errors: list[str]) -> SourceKind:
    text = raw or SourceKind.DISCORD.value
    try:
        return SourceKind(text)
    except ValueError:
        errors.append("INFOVORE_SOURCE must be 'discord' or 'export'")
        return SourceKind.DISCORD


def load_settings(env: Mapping[str, str]) -> Settings:
    errors: list[str] = []

    source_kind = _parse_source_kind(env.get("INFOVORE_SOURCE"), errors)
    is_export = source_kind is SourceKind.EXPORT

    channel_ids = _parse_channel_ids(env.get("INFOVORE_CHANNEL_IDS"), errors)
    export_dir: Path | None = None
    if is_export:
        discord_token = env.get("INFOVORE_DISCORD_TOKEN", "")
        guild_id = _optional_guild_id(env.get("INFOVORE_GUILD_ID"), errors)
        export_dir_raw = _require(env.get("INFOVORE_EXPORT_DIR"), "INFOVORE_EXPORT_DIR", errors)
        if export_dir_raw:
            export_dir = Path(export_dir_raw)
    else:
        discord_token = _require(
            env.get("INFOVORE_DISCORD_TOKEN"), "INFOVORE_DISCORD_TOKEN", errors
        )
        guild_id = _require_positive_int(env.get("INFOVORE_GUILD_ID"), "INFOVORE_GUILD_ID", errors)
    db_path_raw = _require(env.get("INFOVORE_DB_PATH"), "INFOVORE_DB_PATH", errors)
    scratch_dir_raw = env.get("INFOVORE_SCRATCH_DIR", DEFAULT_SCRATCH_DIR)
    quiet_gap_minutes = _optional_positive_int(
        env.get("INFOVORE_QUIET_GAP_MINUTES"),
        DEFAULT_QUIET_GAP_MINUTES,
        "INFOVORE_QUIET_GAP_MINUTES",
        errors,
    )
    batch_size = _optional_positive_int(
        env.get("INFOVORE_BATCH_SIZE"), DEFAULT_BATCH_SIZE, "INFOVORE_BATCH_SIZE", errors
    )
    max_retries = _optional_positive_int(
        env.get("INFOVORE_MAX_RETRIES"), DEFAULT_MAX_RETRIES, "INFOVORE_MAX_RETRIES", errors
    )
    exchange_max_messages = _optional_positive_int(
        env.get("INFOVORE_EXCHANGE_MAX_MESSAGES"),
        DEFAULT_EXCHANGE_MAX_MESSAGES,
        "INFOVORE_EXCHANGE_MAX_MESSAGES",
        errors,
    )
    opt_out_role_name = env.get("INFOVORE_OPT_OUT_ROLE", DEFAULT_OPT_OUT_ROLE_NAME)
    include_bot_messages = _optional_bool(
        env.get("INFOVORE_INCLUDE_BOT_MESSAGES"),
        DEFAULT_INCLUDE_BOT_MESSAGES,
        "INFOVORE_INCLUDE_BOT_MESSAGES",
        errors,
    )
    triage_min_score = _optional_unit_float(
        env.get("INFOVORE_TRIAGE_MIN_SCORE"),
        DEFAULT_TRIAGE_MIN_SCORE,
        "INFOVORE_TRIAGE_MIN_SCORE",
        errors,
    )
    triage_min_p_lore = _optional_unit_float(
        env.get("INFOVORE_TRIAGE_MIN_P_LORE"),
        DEFAULT_TRIAGE_MIN_P_LORE,
        "INFOVORE_TRIAGE_MIN_P_LORE",
        errors,
    )
    stages = {stage: _parse_stage(env, stage, errors) for stage in Stage}
    triage_rules = _load_triage_rules(env.get("INFOVORE_TRIAGE_RULES"), errors)
    workers = _optional_positive_int(
        env.get("INFOVORE_WORKERS"), DEFAULT_WORKERS, "INFOVORE_WORKERS", errors
    )
    exclude_channels = normalize_channel_names(env.get("INFOVORE_EXCLUDE_CHANNELS"))

    if errors:
        raise ConfigError("; ".join(errors))

    return Settings(
        db_path=Path(db_path_raw),
        scratch_dir=Path(scratch_dir_raw),
        stages=stages,
        source=source_kind,
        discord_token=discord_token,
        guild_id=guild_id,
        channel_ids=channel_ids,
        export_dir=export_dir,
        quiet_gap_minutes=quiet_gap_minutes,
        batch_size=batch_size,
        max_retries=max_retries,
        exchange_max_messages=exchange_max_messages,
        opt_out_role_name=opt_out_role_name,
        include_bot_messages=include_bot_messages,
        triage_min_score=triage_min_score,
        triage_min_p_lore=triage_min_p_lore,
        triage_rules=triage_rules,
        workers=workers,
        exclude_channels=exclude_channels,
    )


def read_dotenv(path: Path | str) -> dict[str, str]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    result: dict[str, str] = {}
    for line in file_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key] = value
    return result


def settings_from_environment(environ: Mapping[str, str], dotenv_path: Path | str) -> Settings:
    merged = read_dotenv(dotenv_path)
    merged.update(environ)
    return load_settings(merged)
