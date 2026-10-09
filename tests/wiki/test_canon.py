import pytest

from infovore.wiki.canon import alias_names, canonical, display_names
from infovore.wiki.topics import load_topics


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("SGI Indigo 2", "indigo2"),
        ("indigo2", "indigo2"),
        ("Indigo-2", "indigo2"),
        ("IRIX 6.5", "irix6.5"),
        ("IRIX 65", "irix65"),
        ("6.5.1", "6.5.1"),
        ("v1.2", "v1.2"),
        ("Mr.", "mr"),
        ("Sun", "sun"),
        ("Sun Microsystems Ultra 10", "ultra10"),
        ("Ultra 10", "ultra10"),
        ("", ""),
        ("  ", ""),
    ],
)
def test_canonical(value: str, expected: str) -> None:
    assert canonical(value) == expected


def test_display_names_picks_most_frequent_spelling() -> None:
    names = display_names(["Indigo 2", "indigo2", "Indigo2", "Indigo2", "Indy"])
    assert names == {"indigo2": "Indigo2", "indy": "Indy"}


def test_display_names_breaks_ties_by_string_order() -> None:
    assert display_names(["indigo", "Indigo"]) == {"indigo": "Indigo"}


def test_display_names_empty() -> None:
    assert display_names([]) == {}


def test_alias_names_maps_every_alias_to_its_topic() -> None:
    names = alias_names(load_topics())
    assert names[canonical("personal iris")] == "Indigo"
    assert names[canonical("indigo 2")] == "Indigo2"
