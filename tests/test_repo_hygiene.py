import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GITIGNORE = REPO / ".gitignore"

# Built at runtime, never written literally: a scanner's fixtures are by
# definition examples of what it detects, so a literal marker in this file
# would make it fail its own check.
OPEN_MARKER = "<" * 7
CLOSE_MARKER = ">" * 7

# `=======` is deliberately NOT a marker: at the start of a line it is a valid
# Markdown setext heading underline, and a real conflict always leaves the
# angle-bracket pair behind anyway.
MARKERS = (OPEN_MARKER, CLOSE_MARKER)

TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".sql", ".toml", ".yml", ".yaml", ".json", ".html", ".css", ".js", ".sh"}
)


def has_conflict_marker(text: str) -> bool:
    return any(line.startswith(MARKERS) for line in text.splitlines())


def ignored_names() -> frozenset[str]:
    """Read the ignore list from `.gitignore` rather than keeping a second copy
    here, so the two cannot drift. The file is plain: directory names and
    suffix globs, no negations and no nested ignores, which is what makes a
    filesystem walk a faithful stand-in for what a push would carry."""
    entries = {
        line.strip().rstrip("/")
        for line in GITIGNORE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    return frozenset(entries | {".git"})


def publishable_files() -> list[Path]:
    ignored = ignored_names()
    found: list[Path] = []
    for root, dirs, names in os.walk(REPO):
        dirs[:] = [
            name
            for name in dirs
            if name not in ignored and not any(name.endswith(e.lstrip("*")) for e in ignored)
        ]
        for name in names:
            path = Path(root) / name
            if path.suffix in TEXT_SUFFIXES:
                found.append(path)
    return found


def test_scanner_fires_on_a_conflict_block() -> None:
    text = f"a\n{OPEN_MARKER} HEAD\nours\n{'=' * 7}\ntheirs\n{CLOSE_MARKER} branch\nb\n"

    assert has_conflict_marker(text)


def test_scanner_does_not_fire_on_ordinary_text() -> None:
    assert not has_conflict_marker("a normal line\nanother one\n")


def test_scanner_does_not_fire_on_a_markdown_setext_heading() -> None:
    assert not has_conflict_marker("Data model\n=======\n\nsome prose\n")


def test_scanner_does_not_fire_on_a_marker_mid_line() -> None:
    assert not has_conflict_marker(f"shift left with {OPEN_MARKER} in prose\n")


def test_the_ignore_list_comes_from_gitignore() -> None:
    assert ".venv" in ignored_names()
    assert ".git" in ignored_names()
    assert "infovore" not in ignored_names()


def test_the_walk_reaches_tracked_source_and_skips_the_venv() -> None:
    paths = {path.relative_to(REPO).as_posix() for path in publishable_files()}

    assert "infovore/cli.py" in paths
    assert "README.md" in paths
    assert not any(path.startswith(".venv/") for path in paths)


def test_no_publishable_file_carries_a_conflict_marker() -> None:
    offenders = sorted(
        path.relative_to(REPO).as_posix()
        for path in publishable_files()
        if has_conflict_marker(path.read_text(encoding="utf-8", errors="replace"))
    )

    assert offenders == []
