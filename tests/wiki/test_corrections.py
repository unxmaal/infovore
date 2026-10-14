import pytest

from infovore.wiki.corrections import Correction, load_corrections

TEXT = """
[[claim]]
exchange = 12
statement = "The  O2 is   loud."
drop = true

[[claim]]
exchange = 12
statement = "The O2 has 1GB."
replace = "The O2 takes up to 1GB of RAM"
"""


def test_load_corrections_keys_by_exchange_and_normalised_statement() -> None:
    fixes = load_corrections(TEXT)
    assert fixes == {
        (12, "The O2 is loud."): Correction(12, "The O2 is loud.", None),
        (12, "The O2 has 1GB."): Correction(12, "The O2 has 1GB.", "The O2 takes up to 1GB of RAM"),
    }
    assert fixes[(12, "The O2 is loud.")].key == (12, "The O2 is loud.")
    assert load_corrections("# nothing yet\n") == {}
    assert isinstance(load_corrections(), dict)


@pytest.mark.parametrize(
    "text",
    [
        '[[claim]]\nexchange = 1\nstatement = "x"\n',
        '[[claim]]\nexchange = 1\nstatement = "x"\ndrop = true\nreplace = "y"\n',
        '[[claim]]\nexchange = 1\nstatement = "x"\ndrop = false\n',
        '[[claim]]\nexchange = 1\nstatement = "x"\ndrop = true\n'
        '[[claim]]\nexchange = 1\nstatement = " x "\nreplace = "y"\n',
    ],
)
def test_load_corrections_rejects_ambiguous_and_duplicate_entries(text: str) -> None:
    with pytest.raises(ValueError, match="correction"):
        load_corrections(text)
