import re
from pathlib import Path

import pytest

from infovore.db.fts import TOKENIZE, match_terms, quote

MIGRATIONS = Path(__file__).resolve().parents[2] / "infovore" / "db" / "migrations"
TOKENIZE_RE = re.compile(r'tokenize\s*=\s*"([^"]+)"')


def _fts_migrations() -> list[Path]:
    return sorted(
        p for p in MIGRATIONS.glob("*.sql") if "USING fts5" in p.read_text(encoding="utf-8")
    )


def test_there_is_more_than_one_fts_index_to_disagree() -> None:
    """The seam test below is vacuous if it only ever checks one file, so this
    pins that the duplication it guards actually exists."""
    assert len(_fts_migrations()) >= 2


@pytest.mark.parametrize("migration", _fts_migrations(), ids=lambda p: p.name)
def test_every_fts_index_tokenises_the_way_python_assumes(migration: Path) -> None:
    """The tokenchar list lives in SQL, and `match_terms` strips exactly those
    characters. Nothing connected the two: the strip set was the hand-written
    literal `-./` while the indexes declared `-./_`, so `irix_build_` was
    unmatchable (issue #179). Two indexes tokenising differently would also
    make one query string mean two things."""
    declared = TOKENIZE_RE.findall(migration.read_text(encoding="utf-8"))

    assert declared, f"{migration.name} creates an fts5 table with no explicit tokenize"
    assert set(declared) == {TOKENIZE}


def test_a_trailing_tokenchar_is_stripped() -> None:
    assert match_terms(["Octane2.", "PSU-", "irix_build_", "a/b/"]) == [
        "Octane2",
        "PSU",
        "irix_build",
        "a/b",
    ]


def test_a_leading_tokenchar_is_kept() -> None:
    """The negative control, and the reason stripping is right-hand only: the
    indexed token keeps the slash the message had, so a query that dropped it
    would stop matching paths."""
    assert match_terms(["/usr/people/eric"]) == ["/usr/people/eric"]


def test_an_interior_tokenchar_is_kept() -> None:
    assert match_terms(["6.5.22m"]) == ["6.5.22m"]


def test_a_term_of_only_tokenchars_is_dropped() -> None:
    assert match_terms(["-", "...", "_", "/"]) == []


def test_duplicates_collapse_in_first_seen_order() -> None:
    assert match_terms(["b", "a", "b.", "a"]) == ["b", "a"]


def test_quoting_doubles_embedded_quotes() -> None:
    assert quote('oct" OR x MATCH "a') == '"oct"" OR x MATCH ""a"'
