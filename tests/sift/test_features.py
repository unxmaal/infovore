import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.batch import exchange_inputs_for_ids
from infovore.db.connection import migrate, open_database
from infovore.rows import MessageRow
from infovore.sift.features import (
    FEATURE_SET_VERSION,
    NEIGHBOUR_TOKEN_CAP,
    build_context_tokens,
    exchange_context_tokens,
    exchange_size_bucket,
    message_features,
    position_bucket,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
NOW_TEXT = NOW.isoformat()


def _message(
    content: str,
    reply_to_id: int | None = None,
    id: int = 1,
    author_id: int = 1,
) -> MessageRow:
    return MessageRow(
        id=id,
        channel_id=1,
        guild_id=1,
        author_id=author_id,
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


# --- issue #141: conversation-context features -------------------------------


def test_feature_set_version_is_two() -> None:
    assert FEATURE_SET_VERSION == 2


def test_message_features_default_context_matches_pre_141_shape() -> None:
    """No `context` argument at all -- the ablation baseline and any
    pre-#141 caller -- gets exactly the old token set, no CTX_/POS_/
    EXSIZE_/PREV_/NEXT_/REPLYTO_ tokens at all."""
    tokens = message_features(_message("hello world"), channel_id=1)
    assert not any(
        token.startswith(prefix)
        for token in tokens
        for prefix in ("CTX_", "POS_", "EXSIZE_", "PREV_", "NEXT_", "REPLYTO_")
    )


def test_message_features_unions_in_given_context_tokens() -> None:
    tokens = message_features(
        _message("hello world"), channel_id=1, context=frozenset({"POS_first"})
    )
    assert "POS_first" in tokens
    assert "hello" in tokens


# --- position_bucket / exchange_size_bucket -----------------------------------


def test_position_bucket_first_message_is_always_first() -> None:
    assert position_bucket(0, 1) == "POS_first"
    assert position_bucket(0, 10) == "POS_first"


def test_position_bucket_last_message() -> None:
    assert position_bucket(4, 5) == "POS_last"


def test_position_bucket_early_and_middle() -> None:
    assert position_bucket(1, 5) == "POS_early"
    assert position_bucket(2, 5) == "POS_middle"
    assert position_bucket(3, 10) == "POS_early"
    assert position_bucket(5, 10) == "POS_middle"


def test_exchange_size_bucket_thresholds() -> None:
    assert exchange_size_bucket(1) == "EXSIZE_1-2"
    assert exchange_size_bucket(2) == "EXSIZE_1-2"
    assert exchange_size_bucket(3) == "EXSIZE_3-5"
    assert exchange_size_bucket(5) == "EXSIZE_3-5"
    assert exchange_size_bucket(6) == "EXSIZE_6-15"
    assert exchange_size_bucket(15) == "EXSIZE_6-15"
    assert exchange_size_bucket(16) == "EXSIZE_16+"


# --- build_context_tokens (pure) ----------------------------------------------


def test_build_context_tokens_adds_prev_and_next_word_tokens() -> None:
    prev = _message("run pip install widget", id=1, author_id=1)
    focus = _message("that works", id=2, author_id=2)
    tokens = build_context_tokens([prev, focus], opted_out=frozenset())

    assert {"PREV_run", "PREV_pip", "PREV_install", "PREV_widget"} <= tokens[2]
    assert {"NEXT_that", "NEXT_works"} <= tokens[1]


def test_build_context_tokens_position_and_size_buckets() -> None:
    prev = _message("hi", id=1)
    focus = _message("that works", id=2, author_id=2)
    tokens = build_context_tokens([prev, focus], opted_out=frozenset())

    assert "POS_first" in tokens[1]
    assert "POS_last" in tokens[2]
    assert "EXSIZE_1-2" in tokens[1]
    assert "EXSIZE_1-2" in tokens[2]


def test_build_context_tokens_flags_same_author_as_prev() -> None:
    prev = _message("hi", id=1, author_id=9)
    same_author = _message("that works", id=2, author_id=9)
    different_author = _message("that works", id=2, author_id=1)

    assert "CTX_same_author_prev" in build_context_tokens([prev, same_author], frozenset())[2]
    assert (
        "CTX_same_author_prev"
        not in build_context_tokens([prev, different_author], frozenset())[2]
    )


def test_build_context_tokens_flags_prev_ends_with_question() -> None:
    question = _message("are you sure?", id=1)
    statement = _message("i am sure", id=1)
    focus = _message("that works", id=2, author_id=2)

    assert "CTX_prev_ends_question" in build_context_tokens([question, focus], frozenset())[2]
    assert (
        "CTX_prev_ends_question" not in build_context_tokens([statement, focus], frozenset())[2]
    )


def test_build_context_tokens_caps_distinct_neighbour_tokens() -> None:
    words = " ".join(f"word{i}" for i in range(NEIGHBOUR_TOKEN_CAP + 10))
    prev = _message(words, id=1)
    focus = _message("that works", id=2, author_id=2)

    tokens = build_context_tokens([prev, focus], frozenset())[2]
    prev_word_tokens = {token for token in tokens if token.startswith("PREV_word")}
    assert len(prev_word_tokens) == NEIGHBOUR_TOKEN_CAP


def test_build_context_tokens_excludes_opted_out_prev_neighbour() -> None:
    prev = _message("secret words here", id=1, author_id=42)
    focus = _message("that works", id=2, author_id=2)

    tokens = build_context_tokens([prev, focus], opted_out=frozenset({42}))[2]
    assert not any(token.startswith("PREV_") for token in tokens)
    assert "CTX_same_author_prev" not in tokens
    assert "CTX_prev_ends_question" not in tokens


def test_build_context_tokens_excludes_opted_out_next_neighbour() -> None:
    focus = _message("that works", id=1, author_id=2)
    following = _message("secret words here", id=2, author_id=42)

    tokens = build_context_tokens([focus, following], opted_out=frozenset({42}))[1]
    assert not any(token.startswith("NEXT_") for token in tokens)


def test_build_context_tokens_flags_reply_target_fact_shapes_and_words() -> None:
    reply_target = _message("the manual is at /etc/config/app.conf", id=1, author_id=5)
    other = _message("ignore me", id=2, author_id=6)
    focus = _message("that works", id=3, author_id=2, reply_to_id=1)

    tokens = build_context_tokens([reply_target, other, focus], frozenset())[3]
    assert "REPLYTO_manual" in tokens
    assert "REPLYTO_FACT_path" in tokens
    assert "CTX_is_reply" in tokens


def test_build_context_tokens_omits_reply_target_that_does_not_resolve() -> None:
    focus = _message("that works", id=1, reply_to_id=999)
    tokens = build_context_tokens([focus], frozenset())[1]
    assert not any(token.startswith("REPLYTO_") for token in tokens)
    assert "CTX_is_reply" in tokens


def test_build_context_tokens_resolves_reply_target_from_outside_the_exchange() -> None:
    external = _message("the manual is at /etc/config/app.conf", id=100, author_id=5)
    focus = _message("that works", id=1, reply_to_id=100)

    tokens = build_context_tokens([focus], frozenset(), reply_targets={100: external})[1]
    assert "REPLYTO_manual" in tokens
    assert "REPLYTO_FACT_path" in tokens


def test_build_context_tokens_reply_target_also_excluded_when_opted_out() -> None:
    external = _message("secret manual details", id=100, author_id=42)
    focus = _message("that works", id=1, reply_to_id=100)

    tokens = build_context_tokens([focus], frozenset({42}), reply_targets={100: external})[1]
    assert not any(token.startswith("REPLYTO_") for token in tokens)


# --- fact-shape signals --------------------------------------------------------


def test_build_context_tokens_flags_a_url_neighbour() -> None:
    prev = _message("check https://example.com/help", id=1)
    focus = _message("that works", id=2, author_id=2)
    tokens = build_context_tokens([prev, focus], frozenset())[2]
    assert "PREV_FACT_url" in tokens


def test_build_context_tokens_flags_a_version_neighbour() -> None:
    prev = _message("upgrade to 6.5.22 now", id=1)
    focus = _message("that works", id=2, author_id=2)
    tokens = build_context_tokens([prev, focus], frozenset())[2]
    assert "PREV_FACT_version" in tokens
    assert "PREV_FACT_partnumber" not in tokens


def test_build_context_tokens_flags_a_part_number_neighbour() -> None:
    prev = _message("order part 030-1234-001 please", id=1)
    focus = _message("that works", id=2, author_id=2)
    tokens = build_context_tokens([prev, focus], frozenset())[2]
    assert "PREV_FACT_partnumber" in tokens
    assert "PREV_FACT_version" not in tokens


def test_build_context_tokens_flags_a_code_neighbour() -> None:
    prev = _message("run `ls -la` now", id=1)
    focus = _message("that works", id=2, author_id=2)
    tokens = build_context_tokens([prev, focus], frozenset())[2]
    assert "PREV_FACT_code" in tokens


def test_build_context_tokens_no_fact_shapes_for_plain_text() -> None:
    prev = _message("nothing special here", id=1)
    focus = _message("that works", id=2, author_id=2)
    tokens = build_context_tokens([prev, focus], frozenset())[2]
    assert not any(token.startswith("PREV_FACT_") for token in tokens)


def test_build_context_tokens_flags_fact_shapes_for_next_too() -> None:
    focus = _message("that works", id=1, author_id=2)
    following = _message("see the manual at /etc/config/app.conf", id=2)
    tokens = build_context_tokens([focus, following], frozenset())[1]
    assert "NEXT_FACT_path" in tokens


# --- the issue's decisive fixture: "that works" after a how-to vs after chatter --


def test_that_works_is_identical_without_context_but_distinguishable_with_it() -> None:
    how_to = _message("run pip install widget then restart the service", id=1)
    chatter = _message("lol remember that time we did the thing", id=1)
    keep_focus = _message("that works", id=2, author_id=2)
    trash_focus = _message("that works", id=2, author_id=2)

    # Without context: the two "that works" messages are indistinguishable.
    keep_base = message_features(keep_focus, channel_id=1)
    trash_base = message_features(trash_focus, channel_id=1)
    assert keep_base == trash_base

    # With context: the neighbour's words tell them apart.
    keep_context = build_context_tokens([how_to, keep_focus], frozenset())[2]
    trash_context = build_context_tokens([chatter, trash_focus], frozenset())[2]
    keep_tokens = message_features(keep_focus, channel_id=1, context=keep_context)
    trash_tokens = message_features(trash_focus, channel_id=1, context=trash_context)
    assert keep_tokens != trash_tokens
    assert "PREV_install" in keep_tokens
    assert "PREV_install" not in trash_tokens


# --- exchange_context_tokens (DB-batched) -------------------------------------


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_database(tmp_path / "x.db")
    migrate(conn)
    return conn


def _insert_message(
    conn: sqlite3.Connection,
    message_id: int,
    author_id: int,
    content: str,
    reply_to_id: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
        " created_at, content, reply_to_id, ingested_at, raw_json)"
        " VALUES (?, 1, 1, ?, 'user', ?, ?, ?, ?, '{}')",
        (message_id, author_id, NOW_TEXT, content, reply_to_id, NOW_TEXT),
    )


def _insert_exchange(
    conn: sqlite3.Connection,
    exchange_id: int,
    message_ids: list[int],
) -> None:
    conn.execute(
        "INSERT INTO exchanges (id, channel_id, first_message_id, last_message_id,"
        " started_at, ended_at, message_count, grouping_rule, content_hash)"
        " VALUES (?, 1, ?, ?, ?, ?, ?, 'quiet_gap', ?)",
        (
            exchange_id,
            message_ids[0],
            message_ids[-1],
            NOW_TEXT,
            NOW_TEXT,
            len(message_ids),
            f"h{exchange_id}",
        ),
    )
    for position, message_id in enumerate(message_ids):
        conn.execute(
            "INSERT INTO exchange_messages (exchange_id, message_id, position) VALUES (?, ?, ?)",
            (exchange_id, message_id, position),
        )


def test_exchange_context_tokens_batches_across_multiple_exchanges(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _insert_message(conn, 1, 1, "hello")
    _insert_exchange(conn, 100, [1])
    _insert_message(conn, 2, 1, "world")
    _insert_exchange(conn, 200, [2])

    tokens = exchange_context_tokens(conn, [100, 200], opted_out=frozenset())

    assert "POS_first" in tokens[1]
    assert "POS_first" in tokens[2]


def test_exchange_context_tokens_resolves_reply_target_outside_the_exchange(
    tmp_path: Path,
) -> None:
    conn = _db(tmp_path)
    _insert_message(conn, 5, 9, "see the manual at /etc/config/app.conf")
    _insert_message(conn, 10, 2, "that works", reply_to_id=5)
    _insert_exchange(conn, 100, [10])

    tokens = exchange_context_tokens(conn, [100], opted_out=frozenset())

    assert "REPLYTO_manual" in tokens[10]
    assert "REPLYTO_FACT_path" in tokens[10]


def test_exchange_context_tokens_excludes_opted_out_neighbours(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _insert_message(conn, 1, 42, "secret words here")
    _insert_message(conn, 2, 2, "that works")
    _insert_exchange(conn, 100, [1, 2])
    conn.execute("INSERT INTO opt_outs (user_id, since) VALUES (42, ?)", (NOW_TEXT,))

    tokens = exchange_context_tokens(conn, [100], opted_out=frozenset({42}))

    assert not any(token.startswith("PREV_") for token in tokens[2])


def test_exchange_context_tokens_skips_an_unknown_exchange_id(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    tokens = exchange_context_tokens(conn, [999], opted_out=frozenset())
    assert tokens == {}


def test_exchange_context_tokens_uses_exchange_inputs_for_ids_ordering(tmp_path: Path) -> None:
    """Sanity check that the batched wrapper agrees with the same exchange
    loader `infovore.triage.train` uses -- same message order."""
    conn = _db(tmp_path)
    _insert_message(conn, 1, 1, "first")
    _insert_message(conn, 2, 1, "second")
    _insert_exchange(conn, 100, [1, 2])

    inputs = exchange_inputs_for_ids(conn, [100])
    tokens = exchange_context_tokens(conn, [100], opted_out=frozenset())

    assert [m.id for m in inputs[100].messages] == [1, 2]
    assert set(tokens) == {1, 2}
