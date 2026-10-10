from infovore.triage.lexicon import load_lexicon
from infovore.wiki.support import FLOOR, MIN_TOKENS, Dropped, keep_supported

LEXICON = load_lexicon()
INDIGO = "The Indigo2 uses an R10000 processor"


def keep(
    sentences: list[tuple[str, list[int]]], statements: list[str]
) -> tuple[list[tuple[str, list[int]]], list[str]]:
    kept, dropped = keep_supported(sentences, statements, LEXICON)
    return kept, [d.reason for d in dropped]


def test_defaults() -> None:
    assert FLOOR == 0.25 and MIN_TOKENS == 3


def test_paraphrase_is_kept() -> None:
    text = "Indigo2 systems utilize the R10000 processor."
    assert keep([(text, [0])], [INDIGO]) == ([(text, [0])], [])


def test_invented_fact_is_dropped() -> None:
    text = "The Indigo2 uses an R10000 processor with 512 MB of RAM."
    assert keep([(text, [0])], [INDIGO]) == ([], ["unsupported_fact"])


def test_unrelated_sentence_is_dropped_for_low_overlap() -> None:
    assert keep([("Octane has V12 graphics", [0])], [INDIGO]) == ([], ["low_overlap"])


def test_citation_problems_are_dropped_with_reasons() -> None:
    sentences = [(INDIGO, []), (INDIGO, [0, -1]), (INDIGO, [3]), ("Indigo2 R10000", [0])]
    assert keep(sentences, [INDIGO]) == (
        [],
        ["no_citation", "bad_citation", "bad_citation", "too_short"],
    )


def test_dropped_keeps_text_and_cited_statements() -> None:
    _, dropped = keep_supported([("Octane has V12 graphics", [0, 0])], [INDIGO], LEXICON)
    assert dropped == [Dropped("Octane has V12 graphics", [INDIGO], "low_overlap")]


def test_citations_are_deduplicated_and_only_cited_claims_count() -> None:
    assert keep([(INDIGO, [1, 0, 1])], ["x y", INDIGO]) == ([(INDIGO, [1, 0])], [])
    assert keep([("Octane has V12 graphics", [0])], [INDIGO, "Octane has V12 graphics"])[0] == []
