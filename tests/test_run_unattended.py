import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "run-unattended.sh"
STUB = '#!/bin/sh\necho "$(basename "$0") $*" >> "$CALLS"\n'
UV_STUB = STUB + "exit 0\n"

pytestmark = pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh not installed")


def run_script(tmp_path: Path, *args: str, fail_first: bool = False) -> tuple[int, str, list[str]]:
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
    if fail_first:
        (bin_dir / "uv").write_text(
            STUB + 'n=$(wc -l < "$CALLS")\n[ "$n" -ge 2 ] && exit 0\nexit 1\n'
        )
        (bin_dir / "sleep").write_text("#!/bin/sh\nexit 0\n")
        (bin_dir / "sleep").chmod(0o755)
    env = {
        "HOME": str(home),
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "CALLS": str(calls),
    }
    done = subprocess.run(["zsh", str(SCRIPT), *args], env=env, capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr, calls.read_text().splitlines()


def test_start_refuses_without_approval(tmp_path: Path) -> None:
    code, out, calls = run_script(tmp_path, "start")

    assert code == 1
    assert "--i-approved" in out
    assert calls == []


def test_start_launches_run_loop_in_tmux_under_caffeinate(tmp_path: Path) -> None:
    code, _, calls = run_script(tmp_path, "start", "--i-approved")

    assert code == 0
    tmux = [c for c in calls if c.startswith("tmux new-session")]
    assert len(tmux) == 1
    assert "caffeinate -dimsu" in tmux[0]
    assert not any("new-window" in c or "probe" in c for c in calls)
    assert not any("promote" in c or "PROMPT_VERSION" in c or "status" in c for c in calls)


def test_start_never_invokes_extract_or_order(tmp_path: Path) -> None:
    _, _, calls = run_script(tmp_path, "start", "--i-approved")

    assert not any(c.startswith("uv") for c in calls)
    assert "--order" not in SCRIPT.read_text()
    assert "infovore extract" not in SCRIPT.read_text()


def test_loop_runs_infovore_run_and_restarts_on_failure(tmp_path: Path) -> None:
    code, _, calls = run_script(tmp_path, "loop", fail_first=True)

    assert code == 0
    assert calls == ["uv run infovore run", "uv run infovore run"]


def test_unknown_command_prints_usage_and_exits_2(tmp_path: Path) -> None:
    code, out, calls = run_script(tmp_path, "bogus")

    assert code == 2
    assert "usage:" in out
    assert not any(c.startswith("tmux") for c in calls)
