from pathlib import Path

import pytest

from infovore.config import (
    ConfigError,
    Settings,
    SourceKind,
    Stage,
    StageSettings,
    load_settings,
    read_dotenv,
    resolve_guild_id,
    settings_from_environment,
)

REQUIRED_ENV = {
    "INFOVORE_DISCORD_TOKEN": "tok",
    "INFOVORE_GUILD_ID": "1",
    "INFOVORE_CHANNEL_IDS": "10,20",
    "INFOVORE_DB_PATH": "db.sqlite",
}


def make_stage_settings(**overrides: object) -> StageSettings:
    defaults: dict[str, object] = {
        "backend": "fake",
        "model": "m",
        "concurrency": 2,
        "timeout_seconds": 30.0,
    }
    defaults.update(overrides)
    return StageSettings(**defaults)  # type: ignore[arg-type]


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "discord_token": "super-secret-token",
        "guild_id": 1,
        "channel_ids": (1, 2),
        "db_path": Path("db.sqlite"),
        "scratch_dir": Path("scratch"),
        "stages": {stage: make_stage_settings() for stage in Stage},
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_stage_enum_members() -> None:
    assert {s.value for s in Stage} == {"extract", "probe", "judge"}


def test_stage_settings_defaults_options_to_empty_mapping() -> None:
    settings = make_stage_settings()
    assert settings.options == {}


def test_stage_settings_holds_custom_options() -> None:
    settings = make_stage_settings(options={"binary_path": "/usr/bin/claude"})
    assert settings.options == {"binary_path": "/usr/bin/claude"}


def test_settings_defaults() -> None:
    settings = make_settings()
    assert settings.quiet_gap_minutes == 30
    assert settings.max_retries == 3
    assert settings.exchange_max_messages == 50
    assert settings.opt_out_role_name == "no-archive"
    assert settings.include_bot_messages is False


def test_settings_token_hidden_from_repr() -> None:
    settings = make_settings(discord_token="super-secret-token")
    assert "super-secret-token" not in repr(settings)
    assert "super-secret-token" not in str(settings)


def test_settings_is_frozen() -> None:
    settings = make_settings()
    try:
        settings.guild_id = 2  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("Settings should be frozen")


def test_read_dotenv_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_dotenv(tmp_path / "missing.env") == {}


def test_read_dotenv_parses_key_value_lines(tmp_path: Path) -> None:
    envfile = tmp_path / ".env"
    envfile.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "FOO=bar",
                'QUOTED="quoted value"',
                "SINGLE='single value'",
                "  SPACED = spaced value  ",
                "NOEQUALS",
            ]
        )
    )
    assert read_dotenv(envfile) == {
        "FOO": "bar",
        "QUOTED": "quoted value",
        "SINGLE": "single value",
        "SPACED": "spaced value",
    }


def test_settings_from_environment_merges_dotenv_and_environ(tmp_path: Path) -> None:
    envfile = tmp_path / ".env"
    envfile.write_text("INFOVORE_DISCORD_TOKEN=from-file\nINFOVORE_BATCH_SIZE=7\n")
    environ = {**REQUIRED_ENV, "INFOVORE_DISCORD_TOKEN": "from-environ"}
    settings = settings_from_environment(environ, envfile)
    assert settings.discord_token == "from-environ"
    assert settings.batch_size == 7


def test_settings_from_environment_uses_dotenv_when_not_overridden(tmp_path: Path) -> None:
    envfile = tmp_path / ".env"
    envfile.write_text("INFOVORE_BATCH_SIZE=9\n")
    settings = settings_from_environment(REQUIRED_ENV, envfile)
    assert settings.batch_size == 9


def test_load_settings_happy_path_has_stage_defaults() -> None:
    settings = load_settings(REQUIRED_ENV)
    assert settings.discord_token == "tok"
    assert settings.guild_id == 1
    assert settings.channel_ids == (10, 20)
    assert settings.db_path == Path("db.sqlite")
    assert settings.scratch_dir == Path("scratch")
    assert settings.stages[Stage.EXTRACT] == StageSettings("claude_cli", "sonnet", 2, 60.0, {})
    assert settings.stages[Stage.PROBE] == StageSettings("claude_cli", "sonnet", 2, 60.0, {})
    assert settings.stages[Stage.JUDGE] == StageSettings("claude_cli", "haiku", 2, 60.0, {})


def test_load_settings_scratch_dir_from_env() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_SCRATCH_DIR": "/tmp/x"}
    settings = load_settings(env)
    assert settings.scratch_dir == Path("/tmp/x")


def test_load_settings_optional_overrides() -> None:
    env = {
        **REQUIRED_ENV,
        "INFOVORE_QUIET_GAP_MINUTES": "15",
        "INFOVORE_BATCH_SIZE": "5",
        "INFOVORE_MAX_RETRIES": "1",
        "INFOVORE_EXCHANGE_MAX_MESSAGES": "20",
        "INFOVORE_OPT_OUT_ROLE": "lurker",
        "INFOVORE_INCLUDE_BOT_MESSAGES": "true",
    }
    settings = load_settings(env)
    assert settings.quiet_gap_minutes == 15
    assert settings.batch_size == 5
    assert settings.max_retries == 1
    assert settings.exchange_max_messages == 20
    assert settings.opt_out_role_name == "lurker"
    assert settings.include_bot_messages is True


def test_load_settings_include_bot_messages_falsey_strings() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_INCLUDE_BOT_MESSAGES": "no"}
    assert load_settings(env).include_bot_messages is False


def test_load_settings_per_stage_backend_model_concurrency_timeout() -> None:
    env = {
        **REQUIRED_ENV,
        "INFOVORE_EXTRACT_BACKEND": "openai_compat",
        "INFOVORE_EXTRACT_MODEL": "gpt",
        "INFOVORE_EXTRACT_CONCURRENCY": "8",
        "INFOVORE_EXTRACT_TIMEOUT": "12.5",
    }
    settings = load_settings(env)
    stage = settings.stages[Stage.EXTRACT]
    assert stage.backend == "openai_compat"
    assert stage.model == "gpt"
    assert stage.concurrency == 8
    assert stage.timeout_seconds == 12.5


def test_load_settings_stage_extra_keys_become_options() -> None:
    env = {
        **REQUIRED_ENV,
        "INFOVORE_EXTRACT_BINARY_PATH": "/usr/bin/claude",
        "INFOVORE_JUDGE_BASE_URL": "http://localhost",
    }
    settings = load_settings(env)
    assert settings.stages[Stage.EXTRACT].options == {"binary_path": "/usr/bin/claude"}
    assert settings.stages[Stage.JUDGE].options == {"base_url": "http://localhost"}
    assert settings.stages[Stage.PROBE].options == {}


def test_load_settings_missing_required_reports_all() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_settings({})
    message = str(excinfo.value)
    assert "INFOVORE_DISCORD_TOKEN" in message
    assert "INFOVORE_GUILD_ID" in message
    assert "INFOVORE_CHANNEL_IDS" in message
    assert "INFOVORE_DB_PATH" in message


def test_load_settings_bad_int_reported() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_GUILD_ID": "notanint"}
    with pytest.raises(ConfigError, match="INFOVORE_GUILD_ID"):
        load_settings(env)


def test_load_settings_bad_bool_reported() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_INCLUDE_BOT_MESSAGES": "maybe"}
    with pytest.raises(ConfigError, match="INFOVORE_INCLUDE_BOT_MESSAGES"):
        load_settings(env)


def test_load_settings_empty_channel_list_reported() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_CHANNEL_IDS": " , , "}
    with pytest.raises(ConfigError, match="INFOVORE_CHANNEL_IDS"):
        load_settings(env)


def test_load_settings_channel_list_with_bad_value_reported() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_CHANNEL_IDS": "10,nope,20"}
    with pytest.raises(ConfigError, match="INFOVORE_CHANNEL_IDS"):
        load_settings(env)


def test_load_settings_channel_list_with_non_positive_value_reported() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_CHANNEL_IDS": "10,-1"}
    with pytest.raises(ConfigError, match="INFOVORE_CHANNEL_IDS"):
        load_settings(env)


@pytest.mark.parametrize(
    "key",
    [
        "INFOVORE_QUIET_GAP_MINUTES",
        "INFOVORE_BATCH_SIZE",
        "INFOVORE_MAX_RETRIES",
        "INFOVORE_EXCHANGE_MAX_MESSAGES",
    ],
)
def test_load_settings_non_positive_optional_int_reported(key: str) -> None:
    env = {**REQUIRED_ENV, key: "0"}
    with pytest.raises(ConfigError, match=key):
        load_settings(env)


@pytest.mark.parametrize(
    "key",
    [
        "INFOVORE_QUIET_GAP_MINUTES",
        "INFOVORE_BATCH_SIZE",
        "INFOVORE_MAX_RETRIES",
        "INFOVORE_EXCHANGE_MAX_MESSAGES",
    ],
)
def test_load_settings_non_integer_optional_int_reported(key: str) -> None:
    env = {**REQUIRED_ENV, key: "nope"}
    with pytest.raises(ConfigError, match=key):
        load_settings(env)


def test_load_settings_stage_concurrency_non_positive_reported() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_EXTRACT_CONCURRENCY": "0"}
    with pytest.raises(ConfigError, match="INFOVORE_EXTRACT_CONCURRENCY"):
        load_settings(env)


def test_load_settings_stage_concurrency_non_integer_reported() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_EXTRACT_CONCURRENCY": "nope"}
    with pytest.raises(ConfigError, match="INFOVORE_EXTRACT_CONCURRENCY"):
        load_settings(env)


def test_load_settings_stage_timeout_non_positive_reported() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_PROBE_TIMEOUT": "-1"}
    with pytest.raises(ConfigError, match="INFOVORE_PROBE_TIMEOUT"):
        load_settings(env)


def test_load_settings_stage_timeout_non_numeric_reported() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_PROBE_TIMEOUT": "nope"}
    with pytest.raises(ConfigError, match="INFOVORE_PROBE_TIMEOUT"):
        load_settings(env)


def test_load_settings_collects_every_problem_at_once() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_settings({"INFOVORE_CHANNEL_IDS": "", "INFOVORE_GUILD_ID": "bad"})
    message = str(excinfo.value)
    assert "INFOVORE_DISCORD_TOKEN" in message
    assert "INFOVORE_GUILD_ID" in message
    assert "INFOVORE_CHANNEL_IDS" in message
    assert "INFOVORE_DB_PATH" in message


def test_load_settings_never_leaks_secret_in_error_message() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_DISCORD_TOKEN": "", "INFOVORE_GUILD_ID": "bad"}
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env)
    assert "super-secret" not in str(excinfo.value)


def test_load_settings_defaults_source_to_discord() -> None:
    settings = load_settings(REQUIRED_ENV)
    assert settings.source is SourceKind.DISCORD
    assert settings.export_dir is None


def test_load_settings_invalid_source_reported() -> None:
    env = {**REQUIRED_ENV, "INFOVORE_SOURCE": "carrier-pigeon"}
    with pytest.raises(ConfigError, match="INFOVORE_SOURCE"):
        load_settings(env)


def test_load_settings_export_source_does_not_require_discord_token(tmp_path: Path) -> None:
    env = {
        "INFOVORE_SOURCE": "export",
        "INFOVORE_EXPORT_DIR": str(tmp_path),
        "INFOVORE_DB_PATH": "db.sqlite",
    }
    settings = load_settings(env)
    assert settings.source is SourceKind.EXPORT
    assert settings.discord_token == ""
    assert settings.guild_id is None
    assert settings.channel_ids == ()
    assert settings.export_dir == tmp_path


def test_load_settings_export_source_requires_export_dir() -> None:
    env = {"INFOVORE_SOURCE": "export", "INFOVORE_DB_PATH": "db.sqlite"}
    with pytest.raises(ConfigError, match="INFOVORE_EXPORT_DIR"):
        load_settings(env)


def test_load_settings_export_source_accepts_explicit_guild_and_channels(
    tmp_path: Path,
) -> None:
    env = {
        "INFOVORE_SOURCE": "export",
        "INFOVORE_EXPORT_DIR": str(tmp_path),
        "INFOVORE_DB_PATH": "db.sqlite",
        "INFOVORE_GUILD_ID": "42",
        "INFOVORE_CHANNEL_IDS": "1,2",
    }
    settings = load_settings(env)
    assert settings.guild_id == 42
    assert settings.channel_ids == (1, 2)


def test_load_settings_export_source_bad_guild_id_reported(tmp_path: Path) -> None:
    env = {
        "INFOVORE_SOURCE": "export",
        "INFOVORE_EXPORT_DIR": str(tmp_path),
        "INFOVORE_DB_PATH": "db.sqlite",
        "INFOVORE_GUILD_ID": "not-an-int",
    }
    with pytest.raises(ConfigError, match="INFOVORE_GUILD_ID"):
        load_settings(env)


def test_load_settings_export_source_whitespace_only_channel_ids_means_all(
    tmp_path: Path,
) -> None:
    env = {
        "INFOVORE_SOURCE": "export",
        "INFOVORE_EXPORT_DIR": str(tmp_path),
        "INFOVORE_DB_PATH": "db.sqlite",
        "INFOVORE_CHANNEL_IDS": " , , ",
    }
    settings = load_settings(env)
    assert settings.channel_ids == ()


def test_load_settings_export_source_bad_channel_ids_reported(tmp_path: Path) -> None:
    env = {
        "INFOVORE_SOURCE": "export",
        "INFOVORE_EXPORT_DIR": str(tmp_path),
        "INFOVORE_DB_PATH": "db.sqlite",
        "INFOVORE_CHANNEL_IDS": "1,nope",
    }
    with pytest.raises(ConfigError, match="INFOVORE_CHANNEL_IDS"):
        load_settings(env)


class _SourceWithGuildIds:
    def __init__(self, ids: frozenset[int]) -> None:
        self._ids = ids

    def guild_ids(self) -> frozenset[int]:
        return self._ids


class _SourceWithoutGuildIds:
    pass


def test_resolve_guild_id_returns_configured_value_without_asking_source() -> None:
    settings = make_settings(guild_id=7)
    assert resolve_guild_id(settings, _SourceWithoutGuildIds()) == 7  # type: ignore[arg-type]


def test_resolve_guild_id_infers_single_guild_from_source() -> None:
    settings = make_settings(guild_id=None)
    assert resolve_guild_id(settings, _SourceWithGuildIds(frozenset({5}))) == 5  # type: ignore[arg-type]


def test_resolve_guild_id_raises_when_source_has_no_guild_ids_method() -> None:
    settings = make_settings(guild_id=None)
    with pytest.raises(ConfigError, match="INFOVORE_GUILD_ID"):
        resolve_guild_id(settings, _SourceWithoutGuildIds())  # type: ignore[arg-type]


def test_resolve_guild_id_raises_when_source_has_no_guilds() -> None:
    settings = make_settings(guild_id=None)
    with pytest.raises(ConfigError, match="INFOVORE_GUILD_ID"):
        resolve_guild_id(settings, _SourceWithGuildIds(frozenset()))  # type: ignore[arg-type]


def test_resolve_guild_id_raises_when_source_has_multiple_guilds() -> None:
    settings = make_settings(guild_id=None)
    with pytest.raises(ConfigError, match="INFOVORE_GUILD_ID"):
        resolve_guild_id(settings, _SourceWithGuildIds(frozenset({1, 2})))  # type: ignore[arg-type]
