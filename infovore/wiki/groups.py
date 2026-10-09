from collections.abc import Sequence
from dataclasses import dataclass

from infovore.claims.value import tokens
from infovore.wiki.build import WikiClaim


@dataclass(frozen=True)
class ClaimGroup:
    lead: WikiClaim
    members: tuple[WikiClaim, ...]


def group_claims(claims: Sequence[WikiClaim], threshold: float = 0.5) -> list[ClaimGroup]:
    lead_tokens: list[frozenset[str]] = []
    member_lists: list[list[WikiClaim]] = []
    for claim in claims:
        toks = tokens(claim.statement)
        target = next(
            (
                i
                for i, lead in enumerate(lead_tokens)
                if toks and lead and len(lead & toks) / len(lead | toks) >= threshold
            ),
            None,
        )
        if target is None:
            lead_tokens.append(toks)
            member_lists.append([claim])
        else:
            member_lists[target].append(claim)
    groups = [ClaimGroup(members[0], tuple(members)) for members in member_lists]
    groups.sort(key=lambda g: (-len(g.members), g.lead.date, g.lead.claim_id))
    return groups
