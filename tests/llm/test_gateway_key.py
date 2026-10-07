import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from infovore.config import ConfigError
from infovore.llm.gateway_key import (
    GatewayKeyError,
    headers,
    key,
    read_keychain,
    refusal,
)


def stored(home: Path, value: str) -> dict[str, str]:
    home.mkdir(parents=True, exist_ok=True)
    (home / "gateway.key").write_text(value + "\n", encoding="utf-8")
    return {"LOCALHARNESS_HOME": str(home)}


def no_keychain() -> str:
    raise AssertionError("the Keychain was read")


def test_infovore_env_var_wins_over_everything(tmp_path: Path) -> None:
    env = stored(tmp_path, "from-file") | {
        "INFOVORE_GATEWAY_API_KEY": " ours ",
        "SOHOT_GATEWAY_KEY": "sohot",
    }
    assert key(env, platform="darwin", keychain=no_keychain) == "ours"


def test_sohot_env_var_wins_over_the_stores(tmp_path: Path) -> None:
    env = stored(tmp_path, "from-file") | {"SOHOT_GATEWAY_KEY": "sohot\n"}
    assert key(env, platform="darwin", keychain=no_keychain) == "sohot"


def test_a_mac_reads_the_keychain_before_the_file(tmp_path: Path) -> None:
    env = stored(tmp_path, "from-file") | {"SOHOT_GATEWAY_KEY": "  "}
    assert key(env, platform="darwin", keychain=lambda: "from-keychain") == "from-keychain"


def test_an_empty_keychain_falls_back_to_the_file(tmp_path: Path) -> None:
    assert key(stored(tmp_path, "from-file"), platform="darwin", keychain=lambda: "") == "from-file"


def test_other_platforms_read_only_the_file(tmp_path: Path) -> None:
    assert key(stored(tmp_path, "from-file"), platform="linux", keychain=no_keychain) == "from-file"


def test_the_file_store_can_be_forced_on_a_mac(tmp_path: Path) -> None:
    env = stored(tmp_path, "from-file") | {"SOHOT_GATEWAY_STORE": "file"}
    assert key(env, platform="darwin", keychain=no_keychain) == "from-file"


def test_the_home_defaults_to_localharness_under_the_user_home(tmp_path: Path) -> None:
    stored(tmp_path / "localharness", "default-home")
    assert key({}, platform="linux", user_home=tmp_path) == "default-home"


def test_no_key_anywhere_sends_no_header(tmp_path: Path) -> None:
    env = {"LOCALHARNESS_HOME": str(tmp_path / "missing")}
    assert key(env, platform="darwin", keychain=lambda: "") == ""
    assert headers(env, platform="linux") == {}


def test_headers_carry_the_resolved_key_as_a_bearer_token() -> None:
    assert headers({"SOHOT_GATEWAY_KEY": "sk-x"}) == {"Authorization": "Bearer sk-x"}


def test_the_keychain_is_read_by_service_name_with_security() -> None:
    seen: list[list[str]] = []

    def run(argv: list[str], **kw: Any) -> SimpleNamespace:
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout="sk-kc\n")

    assert read_keychain(run=run, which=lambda _: "/usr/bin/security") == "sk-kc"
    assert seen == [["security", "find-generic-password", "-s", "localharness-gateway", "-w"]]


def test_a_missing_or_failing_keychain_reads_as_empty() -> None:
    def fails(argv: list[str], **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=44, stdout="")

    assert read_keychain(run=fails, which=lambda _: "/usr/bin/security") == ""
    assert read_keychain(run=fails, which=lambda _: None) == ""


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://h/v1", code, "nope", {}, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("code", [401, 403])
def test_a_refused_key_names_the_command_that_prints_it(code: int) -> None:
    error = refusal(http_error(code))
    assert isinstance(error, GatewayKeyError) and isinstance(error, ConfigError)
    assert "soh gateway key" in str(error) and str(code) in str(error)


@pytest.mark.parametrize("error", [http_error(500), http_error(404), OSError("down")])
def test_other_failures_are_not_key_refusals(error: OSError) -> None:
    assert refusal(error) is None
