from typing import Any

import pytest

from infovore.claims.extract import ClaimsReplyError
from infovore.wiki import tagger


def reply(content: str, finish: str = "stop") -> dict[str, Any]:
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}


def test_build_request_shape() -> None:
    assert tagger.build_request("m", ["first", "second"]) == {
        "model": "m",
        "temperature": 0,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": tagger.SYSTEM},
            {"role": "user", "content": "1. first\n2. second"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "tags", "schema": tagger.SCHEMA, "strict": True},
        },
    }


def test_build_request_custom_max_tokens() -> None:
    assert tagger.build_request("m", ["only"], max_tokens=42)["max_tokens"] == 42


def test_schema_is_strict_pairs_of_number_and_names() -> None:
    schema = tagger.SCHEMA
    assert schema["required"] == ["t"] and schema["additionalProperties"] is False
    pair = schema["properties"]["t"]["items"]
    assert pair["minItems"] == 2 and pair["maxItems"] == 2
    assert pair["prefixItems"][0] == {"type": "integer"}
    assert pair["prefixItems"][1]["maxItems"] == 4
    assert pair["prefixItems"][1]["items"] == {"type": "string", "maxLength": 60}


def test_prompt_hash_is_twelve_hex_and_stable() -> None:
    value = tagger.prompt_hash()
    assert len(value) == 12 and int(value, 16) >= 0
    assert value == tagger.prompt_hash()


def test_batch_size() -> None:
    assert tagger.BATCH == 20


def test_parse_reply_ignores_out_of_range_numbers() -> None:
    content = '{"t":[[0,["a"]],[1,["b"]],[3,["c"]]]}'
    assert tagger.parse_reply(reply(content), 2) == {0: ["b"], 1: []}


def test_parse_reply_strips_names_and_drops_blanks() -> None:
    content = '{"t":[[1,["  x  ","  ","y"]]]}'
    assert tagger.parse_reply(reply(content), 1) == {0: ["x", "y"]}


def test_parse_reply_maps_missing_positions_to_empty() -> None:
    assert tagger.parse_reply(reply('{"t":[[2,["z"]]]}'), 3) == {0: [], 1: ["z"], 2: []}


def test_parse_reply_rejects_invalid_json() -> None:
    with pytest.raises(ClaimsReplyError):
        tagger.parse_reply(reply("{not json"), 1)


def test_parse_reply_rejects_truncation() -> None:
    with pytest.raises(ClaimsReplyError, match="truncated"):
        tagger.parse_reply(reply('{"t":[]}', finish="length"), 1)


def test_parse_reply_accepts_other_finish_reasons() -> None:
    assert tagger.parse_reply(reply('{"t":[[1,["a"]]]}', finish="tool_calls"), 1) == {0: ["a"]}


def test_parse_reply_rejects_missing_structure() -> None:
    with pytest.raises(ClaimsReplyError):
        tagger.parse_reply({"choices": []}, 1)
