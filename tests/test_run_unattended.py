import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "run-unattended.sh"
STUB = '#!/bin/sh\necho "$(basename "$0") $*" >> "$CALLS"\n'

pytestmark = pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh not installed")


def run_start(tmp_path: Path, *args: str, version: str | None) -> tuple[int, str, list[str]]:
    home = tmp_path / "home"
    (home / "projects/github/unxmaal/infovore").mkdir(parents=True)
    (home / ".infovore.env").write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("uv", "tmux", "caffeinate"):
        stub = bin_dir / name
        stub.write_text(STUB)
        stub.chmod(0o755)
    calls = tmp_path / "calls.txt"
    calls.write_text("")
    env = {"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin", "CALLS": str(calls)}
    if version is not None:
        env["INFOVORE_PROMPT_VERSION"] = version
    done = subprocess.run(
        ["zsh", str(SCRIPT), "start", *args], env=env, capture_output=True, text=True
    )
    return done.returncode, done.stderr, calls.read_text().splitlines()


def test_start_refuses_without_approval(tmp_path: Path) -> None:
    code, err, calls = run_start(tmp_path, version="v7")

    assert code == 1
    assert "--i-approved" in err
    assert calls == []


def test_start_refuses_without_prompt_version(tmp_path: Path) -> None:
    code, err, calls = run_start(tmp_path, "--i-approved", version=None)

    assert code == 1
    assert "INFOVORE_PROMPT_VERSION" in err
    assert calls == []


def test_start_with_approval_launches_extract_only_and_no_probe(tmp_path: Path) -> None:
    code, _, calls = run_start(tmp_path, "--i-approved", version="v7")

    assert code == 0
    assert any(c.startswith("uv run infovore promote --prompt-version v7") for c in calls)
    assert sum(c.startswith("tmux new-session") for c in calls) == 1
    assert not any("new-window" in c or "probe" in c for c in calls)
