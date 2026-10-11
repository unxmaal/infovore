import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from infovore.claims.value import tokens
from infovore.triage.embed import DEFAULT_MODEL, DEFAULT_REVISION, EMBED_BATCH, Embedder
from infovore.triage.embed_backend import load_embedder
from infovore.wiki.build import WikiClaim

JACCARD_THRESHOLD: Final = 0.5
VECTOR_THRESHOLD: Final = 0.85
GROUPINGS: Final = ("jaccard", "embed")
GROUPING_HELP: Final = (
    "how near-duplicate claims are clustered: token jaccard, or bge-small cosine"
    " (needs `uv run --extra embed`)"
)
Vector = Sequence[float]


@dataclass(frozen=True)
class ClaimGroup:
    lead: WikiClaim
    members: tuple[WikiClaim, ...]

    @property
    def speakers(self) -> frozenset[str]:
        return frozenset(m.speaker for m in self.members)

    @property
    def exchanges(self) -> frozenset[int]:
        return frozenset(m.exchange_id for m in self.members)

    @property
    def span(self) -> tuple[str, str]:
        dates = sorted(m.date for m in self.members)
        return dates[0], dates[-1]

    @property
    def corroboration(self) -> int:
        return len(self.speakers)


Grouper = Callable[[Sequence[WikiClaim]], list[ClaimGroup]]


def cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def group_claims(
    claims: Sequence[WikiClaim],
    threshold: float = JACCARD_THRESHOLD,
    vectors: Sequence[Vector] | None = None,
    vector_threshold: float = VECTOR_THRESHOLD,
) -> list[ClaimGroup]:
    if vectors is not None and len(vectors) != len(claims):
        raise ValueError("one vector per claim is required")
    keys: list[object] = [tokens(c.statement) for c in claims] if vectors is None else list(vectors)
    leads: list[int] = []
    member_lists: list[list[WikiClaim]] = []
    for i, claim in enumerate(claims):
        target = next(
            (
                g
                for g, lead in enumerate(leads)
                if (
                    _jaccard(keys[lead], keys[i]) >= threshold  # type: ignore[arg-type]
                    if vectors is None
                    else cosine(vectors[lead], vectors[i]) >= vector_threshold
                )
            ),
            None,
        )
        if target is None:
            leads.append(i)
            member_lists.append([claim])
        else:
            member_lists[target].append(claim)
    groups = [ClaimGroup(members[0], tuple(members)) for members in member_lists]
    groups.sort(
        key=lambda g: (
            -g.corroboration,
            -len(g.exchanges),
            -len(g.members),
            g.lead.date,
            g.lead.claim_id,
        )
    )
    return groups


def claim_vectors(embedder: Embedder, claims: Sequence[WikiClaim]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for start in range(0, len(claims), EMBED_BATCH):
        batch = claims[start : start + EMBED_BATCH]
        for claim, vector in zip(batch, embedder.embed([c.statement for c in batch]), strict=True):
            out[claim.claim_id] = vector
    return out


def make_grouper(
    embedder: Embedder, claims: Sequence[WikiClaim], vector_threshold: float = VECTOR_THRESHOLD
) -> Grouper:
    vectors = claim_vectors(embedder, claims)

    def group(members: Sequence[WikiClaim]) -> list[ClaimGroup]:
        return group_claims(
            members,
            vectors=[vectors[c.claim_id] for c in members],
            vector_threshold=vector_threshold,
        )

    return group


@dataclass(frozen=True)
class Corroboration:
    groups: int
    by_speakers: dict[str, int]
    corroborated_claims: int
    claims: int

    @property
    def share(self) -> float:
        return self.corroborated_claims / self.claims if self.claims else 0.0

    def __str__(self) -> str:
        buckets = ", ".join(f"{k} speaker(s) {v}" for k, v in self.by_speakers.items())
        return (
            f"groups {self.groups} ({buckets});"
            f" claims in 2+ speaker groups {self.corroborated_claims}/{self.claims}"
            f" ({self.share:.1%})"
        )


def summarise(groups_by_page: Sequence[Sequence[ClaimGroup]]) -> Corroboration:
    by: dict[str, int] = {"1": 0, "2": 0, "3+": 0}
    total = corroborated = count = 0
    for groups in groups_by_page:
        for g in groups:
            count += 1
            total += len(g.members)
            by["1" if g.corroboration == 1 else "2" if g.corroboration == 2 else "3+"] += 1
            if g.corroboration >= 2:
                corroborated += len(g.members)
    return Corroboration(count, by, corroborated, total)


def make_grouper_for(grouping: str, claims: Sequence[WikiClaim]) -> Grouper:
    if grouping == "embed":
        return make_grouper(load_embedder(DEFAULT_MODEL, DEFAULT_REVISION), claims)
    return group_claims
