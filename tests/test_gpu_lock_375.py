import argparse
import contextlib
import dataclasses
import fcntl
import io
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from infovore import cli, doctor, gpu_lock
from infovore.config import ConfigError
from infovore.doctor import Host, Status

REPO = Path(__file__).resolve().parent.parent


def wrapper(home: Path, body: str, mode: int = 0o755) -> Path:
    path = home / "deploy" / "scripts" / "with-gpu-lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(mode)
    return path


@pytest.fixture
def lh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "lh"
    monkeypatch.setenv("LOCALHARNESS_HOME", str(home))
    monkeypatch.delenv(gpu_lock.HELD_ENV, raising=False)
    return home


def test_the_body_runs_while_with_gpu_lock_holds_and_the_holder_ends_after(
    lh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mark = tmp_path / "wrapper.pid"
    monkeypatch.setenv("MARK", str(mark))
    wrapper(lh, 'echo $$ > "$MARK"\nexec "$@"')
    with gpu_lock.take():
        pid = int(mark.read_text())
        os.kill(pid, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_infovore_never_takes_generation_lock_itself(lh: Path) -> None:
    lock = lh / "queue" / "generation.lock"
    lock.parent.mkdir(parents=True)
    wrapper(lh, 'exec "$@"')
    with open(lock, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with gpu_lock.take():
            pass
    texts = {p: p.read_text() for p in (REPO / "infovore").rglob("*.py")}
    assert not [p for p, text in texts.items() if "generation.lock" in text or "lockf" in text]


def test_a_missing_wrapper_is_a_configuration_error_naming_it(lh: Path) -> None:
    with pytest.raises(ConfigError, match=r"lh/deploy/scripts/with-gpu-lock"), gpu_lock.take():
        pytest.fail("ran without the GPU")


def test_a_wrapper_that_is_not_executable_is_refused(lh: Path) -> None:
    wrapper(lh, 'exec "$@"', mode=0o644)
    with pytest.raises(gpu_lock.GpuLockMissingError), gpu_lock.take():
        pytest.fail("ran without the GPU")


def test_inside_with_gpu_lock_it_does_not_wrap_again(
    lh: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(gpu_lock.HELD_ENV, "1")
    ran = []
    with gpu_lock.take():
        ran.append(True)
    assert ran == [True]


def test_a_wrapper_that_refuses_fails_without_running_the_body(lh: Path) -> None:
    wrapper(lh, "echo refusing >&2; exit 3")
    with pytest.raises(gpu_lock.GpuLockRefusedError, match="exited 3"), gpu_lock.take():
        pytest.fail("ran without the GPU")


def test_a_wrapper_that_never_takes_the_gpu_is_stopped(lh: Path) -> None:
    wrapper(lh, "echo busy\nexec sleep 60")
    with pytest.raises(gpu_lock.GpuLockRefusedError, match="exited -15"), gpu_lock.take():
        pytest.fail("ran without the GPU")


def test_a_failing_body_still_releases_the_gpu(
    lh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mark = tmp_path / "wrapper.pid"
    monkeypatch.setenv("MARK", str(mark))
    wrapper(lh, 'echo $$ > "$MARK"\nexec "$@"')
    with pytest.raises(RuntimeError, match="boom"), gpu_lock.take():
        raise RuntimeError("boom")
    with pytest.raises(ProcessLookupError):
        os.kill(int(mark.read_text()), 0)


def test_the_wrapper_defaults_to_the_sohot_deploy_checkout(tmp_path: Path) -> None:
    assert gpu_lock.wrapper_path({}, tmp_path) == (
        tmp_path / "localharness" / "deploy" / "scripts" / "with-gpu-lock"
    )
    assert gpu_lock.wrapper_path({"LOCALHARNESS_HOME": "/x"}, tmp_path) == Path(
        "/x/deploy/scripts/with-gpu-lock"
    )


GPU = [
    "claims extract --endpoint E --model M",
    "claims novelty --run 1 --tag-run 1",
    "relevance cascade --all",
    "relevance compare",
    "relevance embed-score",
    "relevance llm-score --endpoint E --model M",
    "wiki tag --runs 1 --endpoint E --model M --write",
    "wiki write --tag-run 1 --endpoint E --model M --write",
    "wiki write --tag-run 1 --endpoint E --model M --grouping embed",
    "wiki build --out o --grouping embed",
    "wiki stats --grouping embed",
]
NOT_GPU = [
    "claims extract --endpoint E --model M --dry-run",
    "claims report",
    "relevance cascade --all --explain 1",
    "relevance llm-score --endpoint E --model M --dry-run",
    "relevance mine",
    "wiki tag --runs 1 --endpoint E --model M",
    "wiki write --tag-run 1 --endpoint E --model M",
    "wiki build --out o",
    "wiki scan site",
    "status",
    "run",
]


def parsed(line: str) -> tuple[cli.Command, argparse.Namespace]:
    commands = cli.builtin_commands()
    args = cli._build_parser(commands, io.StringIO()).parse_args(line.split())
    return next(c for c in commands if c.name == args.command), args


@pytest.mark.parametrize("line", GPU)
def test_gpu_steps_take_the_gpu(line: str) -> None:
    assert cli.uses_gpu(*parsed(line))


@pytest.mark.parametrize("line", NOT_GPU)
def test_other_steps_leave_the_gpu_alone(line: str) -> None:
    assert not cli.uses_gpu(*parsed(line))


class Step:
    help = "a step"
    migrates = False

    def __init__(self, name: str, gpu: bool, events: list[str]) -> None:
        self.name, self._gpu, self.events = name, gpu, events

    def configure(self, parser: argparse.ArgumentParser) -> None:
        return None

    def gpu(self, args: argparse.Namespace) -> bool:
        return self._gpu

    async def run(self, context: cli.AppContext, args: argparse.Namespace) -> int:
        self.events.append("run")
        return 0


def env(tmp_path: Path) -> dict[str, str]:
    return {
        "INFOVORE_DB_PATH": str(tmp_path / "db.sqlite"),
        "INFOVORE_DISCORD_TOKEN": "t",
        "INFOVORE_GUILD_ID": "1",
        "INFOVORE_PSEUDONYM_SALT": "s",
    }


@pytest.mark.parametrize(("gpu", "expected"), [(True, ["take", "run", "give"]), (False, ["run"])])
def test_main_runs_a_gpu_step_inside_the_lock(
    gpu: bool, expected: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    @contextlib.contextmanager
    def take() -> Iterator[None]:
        events.append("take")
        yield
        events.append("give")

    monkeypatch.setattr(gpu_lock, "take", take)
    code = cli.main(["step"], env(tmp_path), None, commands=[Step("step", gpu, events)])
    assert (code, events) == (0, expected)


def test_main_refuses_a_gpu_step_when_with_gpu_lock_is_absent(lh: Path, tmp_path: Path) -> None:
    events: list[str] = []
    err = io.StringIO()
    code = cli.main(
        ["step"], env(tmp_path), None, stderr=err, commands=[Step("step", True, events)]
    )
    assert (code, events) == (cli.ExitCode.CONFIG, [])
    assert "with-gpu-lock" in err.getvalue()


def doctor_host(tmp_path: Path, environ: dict[str, str]) -> Host:
    return Host(environ=environ, home=tmp_path / "home")


def test_doctor_passes_when_with_gpu_lock_is_installed(tmp_path: Path) -> None:
    wrapper(tmp_path / "lh", 'exec "$@"')
    h = doctor_host(tmp_path, {"LOCALHARNESS_HOME": str(tmp_path / "lh")})
    results = doctor.check_gpu_lock(h)
    assert [(r.name, r.status) for r in results] == [("gpu.lock", Status.PASS)]


def test_doctor_fails_when_with_gpu_lock_is_missing(tmp_path: Path) -> None:
    results = doctor.check_gpu_lock(doctor_host(tmp_path, {}))
    assert [(r.name, r.status) for r in results] == [("gpu.lock", Status.FAIL)]
    assert str(tmp_path / "home" / "localharness" / "deploy") in results[0].detail
    assert "SoHoT" in results[0].hint


def test_doctor_no_longer_checks_lockf() -> None:
    assert not hasattr(doctor, "LOCKF")
    assert "lockf" not in {f.name for f in dataclasses.fields(Host)}
