from datetime import UTC, datetime

from infovore.rows import MessageRow
from infovore.sift.features import message_features

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _message(content: str, reply_to_id: int | None = None) -> MessageRow:
    return MessageRow(
        id=1,
        channel_id=1,
        guild_id=1,
        author_id=1,
        author_name_at_time="alice",
        author_is_bot=False,
        created_at=NOW,
        edited_at=None,
        content=content,
        reply_to_id=reply_to_id,
        thread_id=None,
        deleted_at=None,
        ingested_at=NOW,
        raw_json="{}",
    )


def test_message_features_includes_lowercased_word_tokens() -> None:
    tokens = message_features(_message("Hello WORLD"), channel_id=1)
    assert "hello" in tokens
    assert "world" in tokens


def test_message_features_includes_channel_token() -> None:
    tokens = message_features(_message("hi"), channel_id=42)
    assert "CHAN_42" in tokens


def test_message_features_includes_a_length_bucket() -> None:
    short = message_features(_message("hi"), channel_id=1)
    long = message_features(_message("x" * 200), channel_id=1)
    short_buckets = {token for token in short if token.startswith("MLEN_")}
    long_buckets = {token for token in long if token.startswith("MLEN_")}
    assert len(short_buckets) == 1
    assert len(long_buckets) == 1
    assert short_buckets != long_buckets


def test_message_features_flags_a_url() -> None:
    tokens = message_features(_message("check https://example.com/manual.pdf"), channel_id=1)
    assert "SIG_url" in tokens


def test_message_features_does_not_flag_url_when_absent() -> None:
    tokens = message_features(_message("no links here"), channel_id=1)
    assert "SIG_url" not in tokens


def test_message_features_flags_digit_shapes() -> None:
    tokens = message_features(_message("part 030-1234-001"), channel_id=1)
    assert "SIG_digits" in tokens


def test_message_features_does_not_flag_digits_when_absent() -> None:
    tokens = message_features(_message("no numbers at all"), channel_id=1)
    assert "SIG_digits" not in tokens


def test_message_features_flags_code_backticks() -> None:
    tokens = message_features(_message("run `make install`"), channel_id=1)
    assert "SIG_code" in tokens


def test_message_features_does_not_flag_code_when_absent() -> None:
    tokens = message_features(_message("plain text"), channel_id=1)
    assert "SIG_code" not in tokens


def test_message_features_flags_a_question_mark() -> None:
    tokens = message_features(_message("does anyone have the manual?"), channel_id=1)
    assert "SIG_question" in tokens


def test_message_features_does_not_flag_question_when_absent() -> None:
    tokens = message_features(_message("no question here"), channel_id=1)
    assert "SIG_question" not in tokens


def test_message_features_flags_reply_to_present() -> None:
    tokens = message_features(_message("yes exactly", reply_to_id=7), channel_id=1)
    assert "SIG_reply" in tokens


def test_message_features_does_not_flag_reply_when_absent() -> None:
    tokens = message_features(_message("standalone message"), channel_id=1)
    assert "SIG_reply" not in tokens


def test_message_features_drops_long_tokens_and_trailing_punctuation() -> None:
    tokens = message_features(_message("wow!! amazing,"), channel_id=1)
    assert "wow" in tokens
    assert "amazing" in tokens
    assert "wow!!" not in tokens
    assert "amazing," not in tokens
