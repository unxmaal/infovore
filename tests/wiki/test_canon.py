from collections import Counter

import pytest

from infovore.wiki.canon import (
    Merged,
    alias_names,
    canonical,
    display_names,
    merge_map,
    spelling_counts,
    summarise_merge,
)
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
        ("Indigo²", "indigo2"),
        ("bootp()", "bootp"),
        ("$ORIGIN", "origin"),
        ("*BSD", "bsd"),
        ("Intel® SC450NX", "intelsc450nx"),
        ("Let's Encrypt", "letsencrypt"),
        ("N\u2013brick", "nbrick"),
        ("`fx` command", "fxcommand"),
        ("O2机器", "o2机器"),
    ],
)
def test_canonical(value: str, expected: str) -> None:
    assert canonical(value) == expected


def test_display_names_picks_most_frequent_spelling() -> None:
    names = display_names(["Indigo 2", "indigo2", "Indigo2", "Indigo2", "Indy"])
    assert names == {"indigo2": "Indigo2", "indy": "Indy"}


def test_display_names_breaks_ties_by_string_order() -> None:
    assert display_names(["indigo", "Indigo"]) == {"indigo": "Indigo"}


def test_display_names_skips_tags_that_canonicalise_to_nothing() -> None:
    assert display_names(["  ", "Indy"]) == {"indy": "Indy"}


def test_display_names_empty() -> None:
    assert display_names([]) == {}


def test_alias_names_maps_every_alias_to_its_topic() -> None:
    names = alias_names(load_topics())
    assert names[canonical("personal iris")] == "Indigo"
    assert names[canonical("indigo 2")] == "Indigo2"


def spelled(pairs: dict[str, int]) -> dict[str, Counter[str]]:
    return spelling_counts(t for t, n in pairs.items() for _ in range(n))


def test_merge_map_folds_plurals_scripts_and_generic_nouns_onto_the_bigger_key() -> None:
    merge = merge_map(
        spelled(
            {
                "O2": 100,
                "O2s": 5,
                "O2机器": 2,
                "O2 systems": 3,
                "The O2": 1,
                "power supply": 20,
                "power supplies": 10,
                "patch": 10,
                "patches": 8,
                "box": 4,
                "boxes": 3,
                "IRIX": 50,
                "IRIX workstation": 2,
            }
        )
    )
    assert merge == {
        "o2s": "o2",
        "o2机器": "o2",
        "o2systems": "o2",
        "theo2": "o2",
        "powersupplies": "powersupply",
        "patches": "patch",
        "boxes": "box",
        "irixworkstation": "irix",
    }


def test_merge_map_leaves_acronyms_models_and_small_targets_alone() -> None:
    merge = merge_map(
        spelled(
            {
                "Windows": 50,
                "window": 2,
                "XFS": 20,
                "xf": 1,
                "VMS": 20,
                "VM": 10,
                "Challenge S": 10,
                "Challenge": 8,
                "emacs": 10,
                "eMac": 1,
                "DOSBox": 10,
                "DOS": 20,
                "Visual Workstation": 10,
                "visual": 1,
                "ss": 3,
                "s": 3,
                "class": 3,
                "clas": 3,
            }
        )
    )
    assert merge == {}


def test_merge_map_resolves_chains_and_stops_on_cycles() -> None:
    merge = merge_map(spelled({"Indy": 100, "Indys": 10, "Indys systems": 3}))
    assert merge == {"indys": "indy", "indyssystems": "indy"}
    assert merge_map(spelled({"Indy": 5, "as": 2, "a": 2})) == {}


def test_display_names_fold_merged_spellings_onto_the_target() -> None:
    merge = {"indys": "indy"}
    assert display_names(["Indys", "Indys", "Indys", "Indy", "INDY"], merge) == {"indy": "Indys"}
    assert display_names(["Indys", "Indy", "Indy"], merge) == {"indy": "Indy"}


def test_summarise_merge_counts_keys_and_claim_tags() -> None:
    spellings = spelled({"O2": 5, "O2s": 2, "o2s": 1, "Indy": 3})
    merged = summarise_merge(spellings, merge_map(spellings))
    assert merged == Merged(1, 3) and str(merged) == "1 keys, 3 claim-tags"
