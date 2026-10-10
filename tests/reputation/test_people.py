from pathlib import Path

import pytest

from infovore.config import ConfigError
from infovore.reputation.people import ENV, People, load_people, people_path


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "people.toml"
    path.write_text(text)
    return path


def test_the_flag_beats_the_environment() -> None:
    assert people_path({ENV: "/env"}, "/flag") == Path("/flag")


def test_the_environment_is_used_without_a_flag() -> None:
    assert people_path({ENV: "/env"}, None) == Path("/env")


def test_an_empty_or_missing_environment_value_means_no_file() -> None:
    assert people_path({ENV: ""}, None) is None
    assert people_path({}, None) is None


def test_no_path_or_a_missing_file_is_the_identity_map(tmp_path: Path) -> None:
    for people in (load_people(None), load_people(tmp_path / "nope.toml")):
        assert people.accounts == {}
        assert people.members == {}
        assert people.person(111) == "111"
        assert people.representative("111") == 111
        assert not people.is_banned("111")


def test_an_empty_file_is_the_identity_map(tmp_path: Path) -> None:
    assert load_people(write(tmp_path, "")).accounts == {}


def test_accounts_map_to_one_person_and_bans_are_recorded(tmp_path: Path) -> None:
    people = load_people(
        write(
            tmp_path,
            '[[person]]\nid = "a"\naccounts = [222, 111]\n\n'
            '[[person]]\nid = "b"\naccounts = [333]\nbanned = true\n',
        )
    )

    assert people.accounts == {111: "a", 222: "a", 333: "b"}
    assert people.members == {"a": (111, 222), "b": (333,)}
    assert people.person(222) == "a"
    assert people.person(999) == "999"
    assert people.representative("a") == 111
    assert people.is_banned("b")
    assert not people.is_banned("a")


def test_people_is_a_plain_value() -> None:
    people = People({}, frozenset({"b"}), {})
    assert people.is_banned("b")
    assert not people.is_banned("a")


@pytest.mark.parametrize(
    "text",
    [
        "[other]\nx = 1\n",
        'person = "x"\n',
        "[[person]]\naccounts = [1]\n",
        "[[person]]\nid = 1\naccounts = [1]\n",
        '[[person]]\nid = ""\naccounts = [1]\n',
        '[[person]]\nid = "a"\naccounts = 1\n',
        '[[person]]\nid = "a"\naccounts = ["1"]\n',
        '[[person]]\nid = "a"\naccounts = [true]\n',
        '[[person]]\nid = "a"\naccounts = [1]\n[[person]]\nid = "a"\naccounts = [2]\n',
        '[[person]]\nid = "a"\naccounts = [1]\n[[person]]\nid = "b"\naccounts = [1]\n',
        '[[person]]\nid = "a"\naccounts = [1, 1]\n',
        '[[person]]\nid = "a"\naccounts = [1]\nbanned = "yes"\n',
        "not toml [[[\n",
    ],
)
def test_a_malformed_file_is_a_config_error(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_people(write(tmp_path, text))
