import pytest

from infovore.triage.embed import EMBED_BATCH
from infovore.wiki import groups as groups_module
from infovore.wiki.build import WikiClaim
from infovore.wiki.groups import (
    ClaimGroup,
    cosine,
    group_claims,
    make_grouper,
    make_grouper_for,
    summarise,
)
from tests.triage.test_embed import FakeEmbedder

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
    a, b, c, d = (
        claim(1, "Octane has V12"),
        claim(2, INDIGO),
        claim(3, INDIGO_PARAPHRASE),
        claim(4, "x"),
    )
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


def test_groups_sort_by_distinct_speakers_then_conversations_before_size() -> None:
    one_voice = [
        WikiClaim(1, 1, "2026-01-01", "user-aaaa", "Octane has V12 graphics", frozenset()),
        WikiClaim(2, 1, "2026-01-01", "user-aaaa", "Octane has V12 graphics card", frozenset()),
        WikiClaim(3, 1, "2026-01-01", "user-aaaa", "Octane has the V12 graphics", frozenset()),
    ]
    two_voices = [
        WikiClaim(4, 2, "2026-01-03", "user-bbbb", INDIGO, frozenset()),
        WikiClaim(5, 3, "2026-01-05", "user-cccc", INDIGO_PARAPHRASE, frozenset()),
    ]
    groups = group_claims([*one_voice, *two_voices])

    assert [g.lead.claim_id for g in groups] == [4, 1]
    assert (groups[0].corroboration, groups[0].exchanges, groups[0].span) == (
        2,
        frozenset({2, 3}),
        ("2026-01-03", "2026-01-05"),
    )
    assert (groups[1].corroboration, groups[1].speakers) == (1, frozenset({"user-aaaa"}))


def test_vectors_replace_tokens_and_must_align() -> None:
    a, b, c = claim(1, "alpha"), claim(2, "beta"), claim(3, "gamma")
    vectors = [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]]

    groups = group_claims([a, b, c], vectors=vectors)

    assert [(g.lead, g.members) for g in groups] == [(a, (a, b)), (c, (c,))]
    assert len(group_claims([a, b, c], vectors=vectors, vector_threshold=0.999)) == 3
    with pytest.raises(ValueError, match="one vector per claim"):
        group_claims([a, b], vectors=vectors)


def test_cosine_handles_zero_vectors() -> None:
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_make_grouper_embeds_every_claim_once_in_batches() -> None:
    embedder = FakeEmbedder()
    claims = [claim(i, f"indigo2 claim {i}") for i in range(EMBED_BATCH + 3)]

    grouper = make_grouper(embedder, claims)
    groups = grouper(claims[:5])

    assert [len(call) for call in embedder.calls] == [EMBED_BATCH, 3]
    assert sum(len(g.members) for g in groups) == 5


def test_make_grouper_for_picks_jaccard_or_the_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    assert make_grouper_for("jaccard", []) is group_claims
    monkeypatch.setattr(groups_module, "load_embedder", lambda model, revision: FakeEmbedder())
    grouper = make_grouper_for("embed", [claim(1, "indigo2 r10000")])
    assert grouper is not group_claims and len(grouper([claim(1, "indigo2 r10000")])) == 1


def test_summarise_counts_groups_by_speakers_and_corroborated_claims() -> None:
    solo = ClaimGroup(claim(1, "a"), (claim(1, "a"),))
    pair = ClaimGroup(
        claim(2, "b"),
        (claim(2, "b"), WikiClaim(3, 2, "2026-01-01", "user-bbbb", "b", frozenset())),
    )
    trio = ClaimGroup(
        claim(4, "c"),
        (
            claim(4, "c"),
            WikiClaim(5, 2, "2026-01-01", "user-bbbb", "c", frozenset()),
            WikiClaim(6, 3, "2026-01-01", "user-cccc", "c", frozenset()),
        ),
    )

    summary = summarise([[solo, pair], [trio]])

    assert summary.by_speakers == {"1": 1, "2": 1, "3+": 1}
    assert (summary.groups, summary.corroborated_claims, summary.claims) == (3, 5, 6)
    assert str(summary) == (
        "groups 3 (1 speaker(s) 1, 2 speaker(s) 1, 3+ speaker(s) 1);"
        " claims in 2+ speaker groups 5/6 (83.3%)"
    )
    assert summarise([]).share == 0.0
