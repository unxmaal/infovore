from infovore.wiki.build import WikiClaim
from infovore.wiki.groups import group_claims

INDIGO = "The Indigo2 has an R10000 processor"
INDIGO_PARAPHRASE = "Indigo2 has the R10000 processor"
ALPHA = "Alpha 21264 architecture details"
ALPHA_PARAPHRASE = "Alpha 21364 architecture details"


def claim(i: int, text: str, date: str = "2026-01-01") -> WikiClaim:
    return WikiClaim(i, 1, date, "user-aaaa", text, frozenset())


def test_paraphrases_join_and_a_distinct_claim_stays_apart() -> None:
    a, b, c = claim(1, INDIGO), claim(2, INDIGO_PARAPHRASE), claim(3, "Tru64 runs on Alpha")
    groups = group_claims([a, b, c])
    assert [(g.lead, g.members) for g in groups] == [(a, (a, b)), (c, (c,))]


def test_groups_sort_by_size_first() -> None:
    a, b, c, d = claim(1, "Octane has V12"), claim(2, INDIGO), claim(3, INDIGO_PARAPHRASE), claim(4, "x")
    groups = group_claims([a, b, c, d])
    assert [len(g.members) for g in groups] == [2, 1, 1]
    assert [g.lead for g in groups] == [b, a, d]


def test_equal_sizes_sort_by_lead_date_then_claim_id() -> None:
    a = claim(5, INDIGO, "2026-01-02")
    b = claim(3, ALPHA, "2026-01-01")
    c = claim(2, "Octane has V12 graphics", "2026-01-01")
    assert [g.lead for g in group_claims([a, b, c])] == [c, b, a]


def test_threshold_is_inclusive_and_configurable() -> None:
    a, b = claim(1, "Indigo2 R10000 processor"), claim(2, "Indigo2 R10000 cache")
    assert len(group_claims([a, b])) == 1
    assert len(group_claims([a, b], threshold=0.6)) == 2


def test_statements_without_tokens_never_join() -> None:
    a, b = claim(1, "the a of"), claim(2, "the a of")
    assert [g.members for g in group_claims([a, b])] == [(a,), (b,)]


def test_empty_input_returns_nothing() -> None:
    assert group_claims([]) == []


def test_a_claim_joins_the_first_matching_group() -> None:
    a, b = claim(1, "Indigo2 R10000 processor"), claim(2, "Indigo2 R10000 cache fast")
    c = claim(3, "Indigo2 R10000 processor cache")
    groups = group_claims([a, b, c], threshold=0.6)
    assert [g.members for g in groups] == [(a, c), (b,)]
