import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from infovore.claims.extract import ClaimsReplyError

SYSTEM = (
    "Each numbered line is a claim about vintage computing. For each, list the specific things "
    "it is about: machine models, parts, OS and versions, programs, companies, standards. Use the "
    "usual full name ('Indigo2', 'IRIX 6.5', 'R10000', 'Sun Ultra 10'). 0 to 4 per claim; skip "
    "generic words like 'computer'. Reply {\"t\":[[n,[name,...]],...]} with n the line number."
)
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "t": {
            "type": "array",
            "items": {
                "type": "array",
                "prefixItems": [
                    {"type": "integer"},
                    {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 60},
                        "maxItems": 4,
                    },
                ],
                "minItems": 2,
                "maxItems": 2,
            },
        }
    },
    "required": ["t"],
    "additionalProperties": False,
}
BATCH: Final = 20


def prompt_hash() -> str:
    return hashlib.sha256(json.dumps([SYSTEM, SCHEMA], sort_keys=True).encode()).hexdigest()[:12]


def build_request(model: str, statements: Sequence[str], max_tokens: int = 1200) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1)),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "tags", "schema": SCHEMA, "strict": True},
        },
    }


def parse_reply(reply: Mapping[str, Any], count: int) -> dict[int, list[str]]:
    try:
        choice = reply["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ClaimsReplyError("truncated: reply hit max_tokens")
        body = json.loads(choice["message"]["content"])
        result: dict[int, list[str]] = {i: [] for i in range(count)}
        for number, names in body["t"]:
            if 1 <= int(number) <= count:
                stripped = (str(name).strip() for name in names)
                result[int(number) - 1] = [name for name in stripped if name]
        return result
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ClaimsReplyError(f"unusable reply: {error!r}") from error
