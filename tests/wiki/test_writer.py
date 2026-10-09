import json
from typing import Any

import pytest

from infovore.claims.extract import ClaimsReplyError
from infovore.wiki.writer import MAX_GROUPS, SCHEMA, SYSTEM, build_request, parse_reply, prompt_hash


def reply(content: str, finish: str = "stop") -> dict[str, Any]:
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}


def sentences(*entries: list[Any]) -> str:
    return json.dumps({"s": list(entries)})


def test_prompt_hash_is_twelve_hex_and_stable() -> None:
    value = prompt_hash()
    assert len(value) == 12 and int(value, 16) >= 0
    assert value == prompt_hash()


def test_max_groups() -> None:
    assert MAX_GROUPS == 40


def test_build_request_general_has_no_section_line() -> None:
    assert build_request("m", "O2", "General", ["a", "b"]) == {
        "model": "m",
        "temperature": 0,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Topic: O2\n\n1. a\n2. b"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "section", "schema": SCHEMA, "strict": True},
        },
    }


def test_build_request_names_a_non_general_section() -> None:
    request = build_request("m", "O2", "With Indy", ["a"], max_tokens=123)
    assert request["messages"][1]["content"] == "Topic: O2\nSection: With Indy\n\n1. a"
    assert request["max_tokens"] == 123


def test_parse_reply_returns_zero_based_positions() -> None:
    assert parse_reply(reply(sentences(["text", [1, 3]], ["more", [2]])), 3) == [
        ("text", [0, 2]),
        ("more", [1]),
    ]


def test_parse_reply_keeps_out_of_range_as_minus_one() -> None:
    assert parse_reply(reply(sentences(["text", [0, 4, 2]])), 3) == [("text", [-1, -1, 1])]


def test_parse_reply_rejects_invalid_json_and_missing_structure() -> None:
    with pytest.raises(ClaimsReplyError):
        parse_reply(reply("not json"), 3)
    with pytest.raises(ClaimsReplyError):
        parse_reply({"choices": []}, 3)


def test_parse_reply_rejects_truncation() -> None:
    with pytest.raises(ClaimsReplyError, match="truncated"):
        parse_reply(reply(sentences(["t", [1]]), finish="length"), 3)


def test_parse_reply_accepts_other_finish_reasons() -> None:
    assert parse_reply(reply(sentences(["t", [1]]), finish="stop"), 1) == [("t", [0])]
