import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

FILE: Final = "corrections.toml"


@dataclass(frozen=True)
class Correction:
    exchange: int
    statement: str
    replace: str | None

    @property
    def key(self) -> tuple[int, str]:
        return (self.exchange, self.statement)


def load_corrections(text: str | None = None) -> dict[tuple[int, str], Correction]:
    source = text if text is not None else Path(__file__).with_name(FILE).read_text()
    out: dict[tuple[int, str], Correction] = {}
    for entry in tomllib.loads(source).get("claim", []):
        drop = bool(entry.get("drop", False))
        replace = entry.get("replace")
        fix = Correction(int(entry["exchange"]), " ".join(entry["statement"].split()), replace)
        if drop == (replace is not None):
            raise ValueError(f"correction {fix.key} needs exactly one of drop or replace")
        if fix.key in out:
            raise ValueError(f"duplicate correction {fix.key}")
        out[fix.key] = fix
    return out
