from infovore.rows import LabelRegime

ISOLATED_SOURCE_REFS = frozenset(
    {
        "sift-serve:batch-002",
        "sift-serve:batch-003",
        "sift:2026-09-27T22:21:27.583979+00:00:a",
    }
)


def regime_for_source_ref(source_ref: str | None) -> LabelRegime:
    """Which labelling regime a human label belongs to (issue #170). The
    listed batches were judged one message at a time with no thread around
    it, before PR #139; everything since shows the conversation, which is
    the standard Eric actually labels to. An unknown ref is assumed to have
    context, since that is what every current path provides."""
    if source_ref in ISOLATED_SOURCE_REFS:
        return LabelRegime.ISOLATED
    return LabelRegime.CONTEXT
