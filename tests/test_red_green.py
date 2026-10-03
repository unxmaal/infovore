import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "red_green.sh"

STUB_UV = """#!/bin/sh
if [ -f fix.txt ]; then
  echo "PASSED tests/test_x.py::test_fixed"
else
  echo "FAILED tests/test_x.py::test_fixed - assert"
fi
echo "PASSED tests/test_x.py::test_always"
"""


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t"}
    env |= {"GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    out = subprocess.run(
        ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    (path / "README").write_text("x")
    git(path, "add", "-A")
    git(path, "commit", "-qm", "base")
    return path


def run_script(repo: Path, base: str, head: str, **env: str) -> tuple[int, str]:
    stub_dir = repo.parent / "bin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "uv"
    stub.write_text(STUB_UV)
    stub.chmod(0o755)
    full = {**os.environ, **env, "PATH": f"{stub_dir}:{os.environ['PATH']}"}
    done = subprocess.run(
        [str(SCRIPT), base, head], cwd=repo, env=full, capture_output=True, text=True
    )
    return done.returncode, done.stdout


def commit_tests(repo: Path, with_fix: bool) -> tuple[str, str]:
    base = git(repo, "rev-parse", "HEAD")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("pass\n")
    if with_fix:
        (repo / "fix.txt").write_text("fixed")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "head")
    return base, git(repo, "rev-parse", "HEAD")


def test_reports_red_to_green_and_flags_never_red(repo: Path) -> None:
    base, head = commit_tests(repo, with_fix=True)

    code, out = run_script(repo, base, head)

    assert code == 0
    assert "red->green tests/test_x.py::test_fixed" in out
    assert "NEVER-RED tests/test_x.py::test_always" in out
    assert "red->green: 1, never red: 1, head tests: 2" in out


def test_refactor_marker_excuses_a_green_on_base_test(repo: Path) -> None:
    base, head = commit_tests(repo, with_fix=True)

    _, out = run_script(repo, base, head, RED_GREEN_REFACTOR="test_always")

    assert "green/green(refactor) tests/test_x.py::test_always" in out
    assert "never red: 0" in out


def test_exits_nonzero_when_head_is_not_green(repo: Path) -> None:
    base, head = commit_tests(repo, with_fix=False)

    code, out = run_script(repo, base, head)

    assert code == 1
    assert "HEAD-NOT-GREEN tests/test_x.py::test_fixed" in out


def test_no_changed_tests_is_a_noop(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    (repo / "other.txt").write_text("o")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "more")

    code, out = run_script(repo, base, "HEAD")

    assert code == 0
    assert "no changed test files" in out
