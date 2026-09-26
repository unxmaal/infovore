import re

from infovore.extract.protocol import (
    ExtractedClaim,
    ExtractionOutcome,
    ExtractionRequest,
    Failure,
    FailureKind,
    ProbeOutcome,
)
from infovore.rows import ClaimKind, ClaimRow, Novelty

FAIL_PREFIX = "FAIL: "

MARKER_KINDS: dict[str, ClaimKind] = {
    "FACT": ClaimKind.FACT,
    "CORRECTION": ClaimKind.CORRECTION,
    "PROCEDURE": ClaimKind.PROCEDURE,
    "REF": ClaimKind.REFERENCE,
}

SUPERSEDES_RE = re.compile(r"^(.*) \[supersedes (\d+)\]$")

PROBE_VERDICTS: dict[str, Novelty] = {
    "[known]": Novelty.KNOWN,
    "[partial]": Novelty.PARTIAL,
    "[contradicts]": Novelty.CONTRADICTS,
    "[unknown]": Novelty.UNKNOWN,
}

PROBE_FAIL_MARKER = "[probe-fail]"


def _parse_claim_line(line: str) -> tuple[ClaimKind, str, str] | None:
    for marker, kind in MARKER_KINDS.items():
        prefix = f"{marker}: "
        if not line.startswith(prefix):
            continue
        rest = line[len(prefix) :]
        if " :: " not in rest:
            return None
        subject, _, statement = rest.partition(" :: ")
        return kind, subject, statement
    return None


class MarkerExtractor:
    async def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        related_ids = {claim.id for claim in request.related_claims if claim.id is not None}
        claims: list[ExtractedClaim] = []
        for message in request.messages:
            if message.author_id in request.opted_out_user_ids:
                continue
            for line in message.content.splitlines():
                if line.startswith(FAIL_PREFIX):
                    fail_kind = FailureKind(line[len(FAIL_PREFIX) :])
                    retry_after = 60.0 if fail_kind is FailureKind.USAGE_LIMIT else None
                    return ExtractionOutcome(
                        claims=(),
                        model=None,
                        input_tokens=None,
                        output_tokens=None,
                        failure=Failure(
                            fail_kind, f"fake extractor failure: {fail_kind.value}", retry_after
                        ),
                    )
                parsed = _parse_claim_line(line)
                if parsed is None:
                    continue
                kind, subject, statement = parsed
                supersedes_claim_id: int | None = None
                if kind is ClaimKind.CORRECTION:
                    match = SUPERSEDES_RE.match(statement)
                    if match is not None:
                        statement = match.group(1)
                        candidate_id = int(match.group(2))
                        if candidate_id in related_ids:
                            supersedes_claim_id = candidate_id
                claims.append(
                    ExtractedClaim(
                        statement=statement,
                        subject=subject,
                        kind=kind,
                        confidence=0.9,
                        probe_question=f"What is known about {subject}?",
                        source_message_ids=(message.id,),
                        supersedes_claim_id=supersedes_claim_id,
                    )
                )
        return ExtractionOutcome(
            claims=tuple(claims),
            model="fake-marker",
            input_tokens=None,
            output_tokens=None,
            failure=None,
        )


class MarkerProbe:
    async def probe(self, claim: ClaimRow) -> ProbeOutcome:
        if PROBE_FAIL_MARKER in claim.statement:
            return ProbeOutcome(
                verdict=None,
                model=None,
                answer=None,
                failure=Failure(FailureKind.TRANSIENT, "fake probe failure", None),
            )
        verdict = Novelty.UNKNOWN
        for marker, value in PROBE_VERDICTS.items():
            if marker in claim.statement:
                verdict = value
                break
        return ProbeOutcome(verdict=verdict, model="fake-probe", answer="fake answer", failure=None)
