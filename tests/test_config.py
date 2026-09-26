from pathlib import Path

from infovore.config import Settings, Stage, StageSettings


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
