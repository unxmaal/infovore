"""The Studio's GPU, taken through SoHoT's `scripts/with-gpu-lock` (#375)."""

import contextlib
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

from infovore.config import ConfigError

HELD_ENV: Final = "LH_GPU_LOCK_HELD"
HOME_ENV: Final = "LOCALHARNESS_HOME"
WRAPPER: Final = Path("deploy/scripts/with-gpu-lock")
READY: Final = "held\n"
HOLD: Final = f"import sys; sys.stdout.write({READY!r}); sys.stdout.flush(); sys.stdin.read()"


class GpuLockMissingError(ConfigError):
    pass


class GpuLockRefusedError(Exception):
    pass


def wrapper_path(environ: Mapping[str, str], home: Path) -> Path:
    raw = environ.get(HOME_ENV)
    return (Path(raw).expanduser() if raw else home / "localharness") / WRAPPER


@contextlib.contextmanager
def take() -> Iterator[None]:
    if os.environ.get(HELD_ENV) == "1":
        yield
        return
    wrapper = wrapper_path(os.environ, Path.home())
    if not os.access(wrapper, os.X_OK):
        raise GpuLockMissingError(
            f"{wrapper} is missing or not executable; infovore takes the GPU through SoHoT's "
            f"with-gpu-lock: check out SoHoT's deploy at {wrapper.parents[1]} or set {HOME_ENV}"
        )
    holder = subprocess.Popen(
        [str(wrapper), sys.executable, "-I", "-c", HOLD],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert holder.stdin is not None and holder.stdout is not None
    taken = False
    try:
        taken = holder.stdout.readline() == READY
        if taken:
            yield
    finally:
        if not taken:
            holder.terminate()
        holder.stdin.close()
        code = holder.wait()
    if not taken:
        raise GpuLockRefusedError(f"with-gpu-lock exited {code} without taking the GPU")
