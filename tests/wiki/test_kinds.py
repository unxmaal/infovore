import pytest

from infovore.wiki.kinds import (
    KINDS,
    LABELS,
    assign_kinds,
    kind_counts,
    kind_from_description,
    parents,
    summarise_kinds,
)


@pytest.mark.parametrize(
    ("description", "kind"),
    [
        ("Computer series", "machine"),
        ("Workstation computer series", "machine"),
        ("Computer terminal and workstation family", "machine"),
        ("Computer operating system", "os"),
        ("Free and open-source Unix-like operating system", "os"),
        ("MIPS microprocessor", "cpu"),
        ("Family of 64-bit Intel microprocessors", "cpu"),
        ("Set of computer and peripheral connection standards", "standard"),
        ("1981-2009 American computing company", "company"),
        ("Internal computer component that provides power to other components", "part"),
        ("Form of computer memory", "part"),
        ("Main printed circuit board used for a computing device", "part"),
        ("Graphics card family by SGI", "graphics"),
        ("Cross-platform graphics API", "software"),
        ("Journaling file system for IRIX and Linux", "software"),
        ("Multi-system emulator", "software"),
        ("3D computer graphics software", "software"),
        ("Line of computers by Digital Equipment Corporation", "machine"),
        ("Linux gaming", None),
        ("", None),
    ],
)
def test_kind_from_description(description: str, kind: str | None) -> None:
    assert kind_from_description(description) == kind


def test_parents_nest_by_name_prefix_and_model_number_onto_a_bigger_page() -> None:
    sizes = {
        "IRIX": 100,
        "IRIX 6.5": 40,
        "IRIX 6.5.22": 5,
        "O2": 90,
        "O2 PSU": 4,
        "Octane": 80,
        "Octane2": 30,
        "Big Octane": 200,
        "R10000": 20,
        "Fibre Channel": 16,
        "FC": 26,
    }
    assert parents(sizes) == {
        "IRIX 6.5": "IRIX",
        "IRIX 6.5.22": "IRIX 6.5",
        "O2 PSU": "O2",
        "Octane2": "Octane",
    }
    assert parents({"irix": 3, "IRIX 6.5": 10}) == {}


def test_assign_kinds_prefers_authority_then_description_then_parent() -> None:
    sizes = {"IRIX": 100, "IRIX 6.5": 40, "IRIX 6.5.22": 5, "Octane": 80, "VPro": 10, "Nekoware": 7}
    descriptions = {
        "IRIX": "Computer operating system",
        "Octane": "Hydrocarbon",
        "IRIX 6.5": "Fuel",
    }
    kinds = assign_kinds(sizes, descriptions, {"Octane": "machine", "IRIX 6.5": "os"})
    assert kinds == {
        "IRIX": "os",
        "IRIX 6.5": "os",
        "IRIX 6.5.22": "os",
        "Octane": "machine",
        "VPro": "other",
        "Nekoware": "other",
    }
    counts = kind_counts(kinds, sizes)
    assert counts == {"machine": (1, 80), "os": (3, 145), "other": (2, 17)}
    assert list(counts) == ["machine", "os", "other"]
    assert summarise_kinds(counts) == "machine 1 (80), os 3 (145), other 2 (17)"
    assert set(LABELS) == set(KINDS) and KINDS[-1] == "other"
