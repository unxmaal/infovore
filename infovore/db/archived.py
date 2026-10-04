SCORER_PREFIX = "relevance_"
STAGES = ("denylist", "no_text", "lexicon", "embed", "residue")
MIDDLE_SCORER = f"{SCORER_PREFIX}embed"
_SCORERS = ", ".join(f"'{SCORER_PREFIX}{stage}'" for stage in STAGES)
_RULED_OUT = (
    f"a.scorer IN ('{SCORER_PREFIX}denylist', '{SCORER_PREFIX}no_text')"
    f" OR (a.scorer = '{MIDDLE_SCORER}' AND a.label = 'irrelevant')"
)


def archived_clause(column: str = "exchanges.id") -> str:
    """SQL predicate, no parameters: the exchange's latest cascade run decided it
    relevant or left it in residue. Never cascaded, denylisted, embed-irrelevant
    and no-text exchanges are out. The one definition every reader shares."""
    return (
        f"{column} IN (SELECT a.subject_id FROM annotations a"
        f" WHERE a.subject_kind = 'exchange' AND a.scorer IN ({_SCORERS})"
        " AND a.created_at = (SELECT MAX(b.created_at) FROM annotations b"
        "  WHERE b.subject_kind = 'exchange' AND b.subject_id = a.subject_id"
        f"  AND b.scorer IN ({_SCORERS}))"
        f" GROUP BY a.subject_id HAVING SUM({_RULED_OUT}) = 0)"
    )
