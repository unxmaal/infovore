from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


class ProcessRunner(Protocol):
    async def run(
        self, argv: Sequence[str], stdin: str, cwd: str, timeout: float
    ) -> ProcessResult: ...
