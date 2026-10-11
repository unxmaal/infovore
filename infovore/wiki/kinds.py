import re
from collections.abc import Mapping
from typing import Final

OTHER: Final = "other"
KINDS: Final = (
    "machine",
    "cpu",
    "graphics",
    "os",
    "software",
    "part",
    "standard",
    "company",
    OTHER,
)
LABELS: Final = {
    "machine": "Machines",
    "cpu": "Processors",
    "graphics": "Graphics",
    "os": "Operating systems",
    "software": "Software",
    "part": "Parts and peripherals",
    "standard": "Standards and interfaces",
    "company": "Companies",
    OTHER: "Other",
}
_RULES: Final = (
    ("os", r"operating system|unix-like|linux distribution|\bkernel\b"),
    ("cpu", r"microprocessor|\bprocessors?\b|\bcpus?\b|instruction set|microarchitecture"),
    (
        "company",
        r"company|corporation|manufacturer|\bvendor\b|subsidiary|organi[sz]ation|conference",
    ),
    ("standard", r"\bstandards?\b|protocol|\binterface\b|\bbus\b|connector|specification|encoding"),
    (
        "graphics",
        r"graphics (?:card|board|subsystem|accelerator|processing unit|chip|hardware)"
        r"|video card|\bgpu\b|framebuffer",
    ),
    (
        "software",
        r"software|program|application|emulator|library|\bapi\b|compiler|file system|editor|browser"
        r"|\bshell\b|utility|window manager|desktop environment|language|video game|\bgame\b"
        r"|toolkit|framework|daemon|command|driver|format|implementation",
    ),
    (
        "part",
        r"component|memory|power supply|storage|\bdrives?\b|\bdisks?\b|peripheral|device"
        r"|circuit board|monitor|display|keyboard|mouse|module|\bchips?\b|cable|adapter"
        r"|\bcards?\b|\brom\b",
    ),
    (
        "machine",
        r"computer|workstation|server|supercomputer|console|terminal|mainframe|laptop|notebook",
    ),
)
_COMPILED: Final = tuple((kind, re.compile(rx, re.IGNORECASE)) for kind, rx in _RULES)
_MODEL = re.compile(r"^([A-Za-z]+)\d")
_VERSION = re.compile(r"\.\d+[a-z]*$")
_TRAILER = re.compile(
    r"\s+(?:by|for|from|developed|made|manufactured|used|produced|sold|introduced|released)\b",
    re.IGNORECASE,
)


def kind_from_description(description: str) -> str | None:
    head = _TRAILER.split(description, 1)[0]
    for kind, pattern in _COMPILED:
        if pattern.search(head):
            return kind
    return None


def parents(sizes: Mapping[str, int]) -> dict[str, str]:
    lookup = {name.casefold(): name for name in sizes}

    def bigger(candidate: str, name: str) -> str | None:
        parent = lookup.get(candidate.casefold())
        return parent if parent and sizes[parent] > sizes[name] else None

    out: dict[str, str] = {}
    for name in sizes:
        words = name.split()
        shorter = [_VERSION.sub("", name)]
        shorter += [" ".join(words[:k]) for k in range(len(words) - 1, 0, -1)]
        shorter += [m.group(1) for m in [_MODEL.match(name)] if m]
        found = next((p for s in shorter if s != name and (p := bigger(s, name))), None)
        if found:
            out[name] = found
    return out


def assign_kinds(
    sizes: Mapping[str, int], descriptions: Mapping[str, str], authority: Mapping[str, str]
) -> dict[str, str]:
    parent = parents(sizes)
    out: dict[str, str] = {}

    def resolve(name: str) -> str:
        if name not in out:
            kind = authority.get(name) or kind_from_description(descriptions.get(name, ""))
            if kind is None:
                kind = resolve(parent[name]) if name in parent else OTHER
            out[name] = kind
        return out[name]

    for name in sizes:
        resolve(name)
    return out


def kind_counts(kinds: Mapping[str, str], sizes: Mapping[str, int]) -> dict[str, tuple[int, int]]:
    out = {kind: (0, 0) for kind in KINDS}
    for name, kind in kinds.items():
        pages, claims = out[kind]
        out[kind] = (pages + 1, claims + sizes[name])
    return {kind: counts for kind, counts in out.items() if counts[0]}


def summarise_kinds(counts: Mapping[str, tuple[int, int]]) -> str:
    return ", ".join(f"{kind} {pages} ({claims})" for kind, (pages, claims) in counts.items())
