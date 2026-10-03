import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "run-unattended.sh"
STUB = '#!/bin/sh\necho "$(basename "$0") $*" >> "$CALLS"\n'
UV_STUB = STUB + (
    'case "$*" in\n'
    '  *PROMPT_VERSION*) echo "$CODE_VERSION" ;;\n'
    '  *status*) [ -n "$LIVE_VERSION" ] && echo "live prompt version: $LIVE_VERSION" ;;\n'
    "esac\nexit 0\n"
)

pytestmark = pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh not installed")


def run_script(
    tmp_path: Path, *args: str, code_version: str = "v7", live_version: str = "v7"
) -> tuple[int, str, list[str]]:
    home = tmp_path / "home"
    (home / "projects/github/unxmaal/infovore").mkdir(parents=True)
    (home / ".infovore.env").write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("uv", "tmux", "caffeinate"):
        stub = bin_dir / name
        stub.write_text(UV_STUB if name == "uv" else STUB)
        stub.chmod(0o755)
    calls = tmp_path / "calls.txt"
    calls.write_text("")
    env = {
        "HOME": str(home),
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "CALLS": str(calls),
        "CODE_VERSION": code_version,
        "LIVE_VERSION": live_version,
    }
    done = subprocess.run(["zsh", str(SCRIPT), *args], env=env, capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr, calls.read_text().splitlines()


def test_start_refuses_without_approval(tmp_path: Path) -> None:
    code, out, calls = run_script(tmp_path, "start")

    assert code == 1
    assert "--i-approved" in out
    assert calls == []


def test_start_refuses_on_version_mismatch_and_never_promotes(tmp_path: Path) -> None:
    code, out, calls = run_script(
        tmp_path, "start", "--i-approved", code_version="v8", live_version="v7"
    )

    assert code == 1
    assert "code prompt is v8, promoted is v7" in out
    assert "promote deliberately" in out
    assert not any(c.startswith("uv run infovore promote") for c in calls)
    assert not any(c.startswith("tmux") for c in calls)


def test_start_refuses_when_nothing_is_promoted(tmp_path: Path) -> None:
    code, out, calls = run_script(tmp_path, "start", "--i-approved", live_version="")

    assert code == 1
    assert "promoted is none" in out
    assert not any(c.startswith("tmux") for c in calls)


def test_start_with_matching_versions_launches_extract_only_and_never_promotes(
    tmp_path: Path,
) -> None:
    code, _, calls = run_script(tmp_path, "start", "--i-approved")

    assert code == 0
    assert not any(c.startswith("uv run infovore promote") for c in calls)
    assert sum(c.startswith("tmux new-session") for c in calls) == 1
    assert not any("new-window" in c or "probe" in c for c in calls)


def test_unknown_command_prints_usage_and_exits_2(tmp_path: Path) -> None:
    code, out, calls = run_script(tmp_path, "bogus")

    assert code == 2
    assert "usage:" in out
    assert not any(c.startswith("tmux") for c in calls)
