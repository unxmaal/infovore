import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from infovore.claims.extract import ClaimsReplyError

SYSTEM = (
    "Write a short wiki section about the topic using only the numbered claims. "
    "2 to 6 sentences, plain and factual, merging claims that say the same thing. "
    "Give each sentence the numbers of the claims it rests on. "
    "Add nothing the claims do not say."
)
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "s": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "array",
                "prefixItems": [
                    {"type": "string", "maxLength": 400},
                    {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
        }
    },
    "required": ["s"],
    "additionalProperties": False,
}
MAX_GROUPS: Final = 40


def prompt_hash() -> str:
    return hashlib.sha256(json.dumps([SYSTEM, SCHEMA], sort_keys=True).encode()).hexdigest()[:12]


def build_request(
    model: str, topic: str, section: str, statements: Sequence[str], max_tokens: int = 900
) -> dict[str, Any]:
    lines = [f"Topic: {topic}"]
    if section != "General":
        lines.append(f"Section: {section}")
    lines.append("")
    lines.extend(f"{i}. {s}" for i, s in enumerate(statements, 1))
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "section", "schema": SCHEMA, "strict": True},
        },
    }


def parse_reply(reply: Mapping[str, Any], count: int) -> list[tuple[str, list[int]]]:
    try:
        choice = reply["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ClaimsReplyError("truncated: reply hit max_tokens")
        body = json.loads(choice["message"]["content"])
        return [
            (str(text), [n - 1 if 1 <= n <= count else -1 for n in map(int, numbers)])
            for text, numbers in body["s"]
        ]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ClaimsReplyError(f"unusable reply: {error!r}") from error
