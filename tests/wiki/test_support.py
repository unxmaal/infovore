from infovore.wiki.support import SUPPORT, keep_supported, support

INDIGO = "Indigo2 R10000 processor"


def test_support_is_the_fraction_of_sentence_tokens_found_in_cited() -> None:
    assert support("Indigo2 R10000 cache", [INDIGO]) == 2 / 3
    assert support("Indigo2 R10000", [INDIGO, "cache"]) == 1.0
    assert support("quantum flux", [INDIGO]) == 0.0


def test_support_is_zero_for_a_sentence_without_tokens() -> None:
    assert support("the a of", [INDIGO]) == 0.0


def test_threshold_default() -> None:
    assert SUPPORT == 0.6


def test_paraphrase_is_kept() -> None:
    kept, dropped = keep_supported([("The Indigo2 has the R10000 processor.", [0])], [INDIGO])
    assert kept == [("The Indigo2 has the R10000 processor.", [0])] and dropped == 0


def test_invented_spec_is_dropped_and_counted() -> None:
    text = "The Indigo2 has 512MB of RAM and a 200MHz R4400"
    assert keep_supported([(text, [0])], [INDIGO]) == ([], 1)


def test_bad_citations_are_dropped_even_next_to_valid_ones() -> None:
    sentences = [(INDIGO, [0, -1]), (INDIGO, [0, 1]), (INDIGO, [5])]
    assert keep_supported(sentences, [INDIGO]) == ([], 3)


def test_empty_citation_is_dropped() -> None:
    assert keep_supported([(INDIGO, [])], [INDIGO]) == ([], 1)


def test_citations_are_deduplicated_in_first_occurrence_order() -> None:
    kept, dropped = keep_supported([(INDIGO, [1, 0, 1])], ["x y", INDIGO])
    assert kept == [(INDIGO, [1, 0])] and dropped == 0


def test_support_only_counts_cited_claims() -> None:
    sentences = [("Octane has V12 graphics", [0])]
    assert keep_supported(sentences, [INDIGO, "Octane has V12 graphics"]) == ([], 1)


def test_support_equal_to_the_threshold_is_kept() -> None:
    text = "Indigo2 R10000 cache"
    kept, dropped = keep_supported([(text, [0])], [INDIGO], threshold=2 / 3)
    assert kept == [(text, [0])] and dropped == 0


def test_sentence_with_fewer_than_three_tokens_is_dropped() -> None:
    assert keep_supported([("2", [0]), ("Indigo2 R10000", [0])], ["2 " + INDIGO]) == ([], 2)
