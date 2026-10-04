import re

import pytest

from infovore.wiki.topics import load_topics, slug


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
