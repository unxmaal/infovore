"""The LiteLLM gateway key, resolved the way SoHoT's own clients resolve it."""

import os
import shutil
import subprocess
import sys
import urllib.error
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from infovore.config import ConfigError

ENV_VARS: Final = ("INFOVORE_GATEWAY_API_KEY", "SOHOT_GATEWAY_KEY")
STORE_ENV: Final = "SOHOT_GATEWAY_STORE"
HOME_ENV: Final = "LOCALHARNESS_HOME"
SERVICE: Final = "localharness-gateway"
FILE_NAME: Final = "gateway.key"
HINT: Final = (
    "`soh gateway key` prints the serving machine's gateway key; "
    "set INFOVORE_GATEWAY_API_KEY or SOHOT_GATEWAY_KEY to it"
)


class GatewayKeyError(ConfigError):
    pass


def read_keychain(
    run: Callable[..., Any] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> str:
    if which("security") is None:
        return ""
    result = run(
        ["security", "find-generic-password", "-s", SERVICE, "-w"],
        capture_output=True,
        text=True,
    )
    return (result.stdout or "").strip() if result.returncode == 0 else ""


def _read_file(environ: Mapping[str, str], user_home: Path | None) -> str:
    raw = environ.get(HOME_ENV)
    home = Path(raw).expanduser() if raw else (user_home or Path.home()) / "localharness"
    try:
        return (home / FILE_NAME).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def key(
    environ: Mapping[str, str] | None = None,
    *,
    platform: str | None = None,
    keychain: Callable[[], str] = read_keychain,
    user_home: Path | None = None,
) -> str:
    environ = os.environ if environ is None else environ
    for name in ENV_VARS:
        value = (environ.get(name) or "").strip()
        if value:
            return value
    if (platform or sys.platform) == "darwin" and environ.get(STORE_ENV) != "file":
        value = keychain()
        if value:
            return value
    return _read_file(environ, user_home)


def headers(
    environ: Mapping[str, str] | None = None,
    *,
    platform: str | None = None,
    keychain: Callable[[], str] = read_keychain,
) -> dict[str, str]:
    value = key(environ, platform=platform, keychain=keychain)
    return {"Authorization": f"Bearer {value}"} if value else {}


def refusal(error: BaseException) -> GatewayKeyError | None:
    if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403):
        return GatewayKeyError(f"the gateway refused the key (HTTP {error.code}): {HINT}")
    return None
