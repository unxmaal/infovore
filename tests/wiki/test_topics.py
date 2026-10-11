import re

import pytest

from infovore.wiki.kinds import KINDS
from infovore.wiki.topics import disambiguations, load_topics, slug, topic_kinds, topic_redirects


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("My O2 boots slowly", {"O2"}),
        ("an o2+ with an R12k", {"O2", "R12000"}),
        ("R10k and R10000 are the same", {"R10000"}),
        ("Octane2 with R14K", {"Octane", "R14000"}),
        ("origin 350 and O350", {"Origin 350"}),
        ("running irix 6.5.30 on an Indy", {"IRIX", "IRIX 6.5", "Indy"}),
        ("6.5.22m patch set", {"IRIX 6.5"}),
        ("IRIX5.3 is old", {"IRIX", "IRIX 5.3"}),
        ("Silicon Graphics made it", {"SGI"}),
        ("no3 and wo2 are not o2s", set()),
        ("nothing relevant", set()),
    ],
)
def test_assign(text: str, expected: set[str]) -> None:
    assert load_topics().assign(text) == frozenset(expected)


def test_names_and_slugs_are_unique_and_url_safe() -> None:
    names = load_topics().names
    slugs = [slug(n) for n in names]
    assert len(set(names)) == len(names)
    assert len(set(slugs)) == len(slugs)
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", s) for s in slugs)


def test_slug() -> None:
    assert slug("IRIX 6.5") == "irix-6-5"
    assert slug("O2") == "o2"


def test_duplicate_names_are_rejected() -> None:
    text = '[[topic]]\nname = "A"\naliases = ["a"]\n[[topic]]\nname = "A"\naliases = ["b"]\n'
    with pytest.raises(ValueError, match="duplicate topic"):
        load_topics(text)


def test_topic_with_only_patterns_loads() -> None:
    topics = load_topics('[[topic]]\nname = "A"\npatterns = ["x+y"]\n')
    assert topics.assign("see xxy") == frozenset({"A"})


def test_kind_only_topics_label_without_tagging() -> None:
    text = '[[topic]]\nname = "A"\nkind = "cpu"\n[[topic]]\nname = "B"\naliases = ["b"]\n'
    topics = load_topics(text)
    assert topics.names == ["B"] and topics.assign("a b") == frozenset({"B"})
    assert topic_kinds(text) == {"A": "cpu"}
    with pytest.raises(ValueError, match="unknown kind for A"):
        topic_kinds('[[topic]]\nname = "A"\nkind = "gadget"\n')
    with pytest.raises(ValueError, match="duplicate topic"):
        topic_kinds('[[topic]]\nname = "A"\nkind = "cpu"\n[[topic]]\nname = "A"\nkind = "os"\n')
    kinds = topic_kinds()
    assert set(load_topics().names) <= set(kinds) and set(kinds.values()) <= set(KINDS)
    assert kinds["Octane"] == "machine" and kinds["R10000"] == "cpu" and kinds["IRIX"] == "os"


def test_redirects_and_disambiguations_load_and_are_checked() -> None:
    text = (
        '[[topic]]\nname = "Fibre Channel"\nkind = "standard"\n'
        'redirects = ["FC", "Fiber Channel"]\n'
        '[[topic]]\nname = "O2"\naliases = ["o2"]\n'
        '[[disambiguation]]\nname = "Ultra"\ncandidates = ["O2", "SCSI"]\n'
    )
    assert topic_redirects(text) == {"FC": "Fibre Channel", "Fiber Channel": "Fibre Channel"}
    assert disambiguations(text) == {"Ultra": ["O2", "SCSI"]}
    assert load_topics(text).names == ["O2"]
    with pytest.raises(ValueError, match="redirect O2 is already"):
        topic_redirects('[[topic]]\nname = "O2"\n[[topic]]\nname = "B"\nredirects = ["O2"]\n')
    with pytest.raises(ValueError, match="redirect x is already"):
        topic_redirects(
            '[[topic]]\nname = "A"\nredirects = ["x"]\n[[topic]]\nname = "B"\nredirects = ["x"]\n'
        )
    with pytest.raises(ValueError, match="disambiguation A is already"):
        disambiguations('[[topic]]\nname = "A"\n[[disambiguation]]\nname = "A"\ncandidates = []\n')
    assert disambiguations('[[topic]]\nname = "A"\n') == {}
    live = topic_redirects()
    assert live["FC"] == "Fibre Channel" and live["I2"] == "Indigo2" and live["6.5"] == "IRIX 6.5"
    assert disambiguations()["Ultra"] == ["Ultra 5", "Ultra64", "SCSI"]
