import re
from pathlib import Path

from infovore.claims.redact import leaks, pseudonym, pseudonyms, redact_conversation
from tests.claims.seed import SALT, conversation, db, messages_of

LINES = [
    (11, "Alice Smith", "my Indy runs IRIX 6.5, thanks bobby"),
    (22, "bobby", "<@11> try the PROM, @Alice Smith said so"),
    (33, "xy", "ping <@!44> and <@22>, hello alice smith"),
]


def test_a_pseudonym_is_stable_salted_and_hides_the_id() -> None:
    first = pseudonym(11, SALT)

    assert re.fullmatch(r"user-[0-9a-f]{4}", first)
    assert first == pseudonym(11, SALT)
    assert first != pseudonym(11, "other-salt")
    assert first != pseudonym(12, SALT)


def test_colliding_pseudonyms_are_lengthened_deterministically() -> None:
    ids = list(range(1, 40))
    mapping = pseudonyms(ids, SALT, width=1)

    assert len(set(mapping.values())) == len(ids)
    assert pseudonyms(list(reversed(ids)), SALT, width=1) == mapping
    assert any(len(v) > len("user-a") for v in mapping.values())


def test_names_and_mentions_are_replaced_in_authors_and_text(tmp_path: Path) -> None:
    conn = db(tmp_path)
    eid, ids = conversation(conn, LINES, 1)
    messages = messages_of(conn, eid)

    redacted = redact_conversation(messages, SALT)
    shown = "\n".join(f"{line.speaker}: {line.text}" for line in redacted.lines)

    for real in ("Alice", "Smith", "bobby", "<@", "alice smith", "11>"):
        assert real not in shown
    a, b = pseudonym(11, SALT), pseudonym(22, SALT)
    assert redacted.lines[0].text == f"my Indy runs IRIX 6.5, thanks {b}"
    assert redacted.lines[1].text == f"{a} try the PROM, {a} said so"
    assert pseudonym(44, SALT) in redacted.lines[2].text
    assert [line.ref for line in redacted.lines] == [1, 2, 3]
    assert [line.message_id for line in redacted.lines] == ids
    assert redacted.speakers == {a, b, pseudonym(33, SALT)}


def test_short_names_are_not_pattern_replaced(tmp_path: Path) -> None:
    conn = db(tmp_path)
    eid, _ = conversation(conn, [(1, "xy", "a xy b"), (2, "pat", "hi")], 1)

    redacted = redact_conversation(messages_of(conn, eid), SALT)

    assert redacted.lines[0].text == "a xy b"


def test_blank_messages_are_dropped_and_refs_count_rendered_lines(tmp_path: Path) -> None:
    conn = db(tmp_path)
    eid, ids = conversation(conn, [(1, "ann", "  "), (2, "bob", "real text")], 1)

    redacted = redact_conversation(messages_of(conn, eid), SALT)

    assert [(line.ref, line.message_id) for line in redacted.lines] == [(1, ids[1])]


def test_leak_check_finds_names_in_model_output(tmp_path: Path) -> None:
    conn = db(tmp_path)
    eid, _ = conversation(conn, LINES, 1)
    redacted = redact_conversation(messages_of(conn, eid), SALT)

    assert leaks("BOBBY said it works", redacted.names)
    assert leaks("alice smith has an Indy", redacted.names)
    assert not leaks("user-1234 has an Indy", redacted.names)


def test_a_conversation_of_only_short_names_still_redacts_mentions(tmp_path: Path) -> None:
    conn = db(tmp_path)
    eid, _ = conversation(conn, [(1, "xy", "hey <@2>")], 1)

    redacted = redact_conversation(messages_of(conn, eid), SALT)

    assert redacted.names == []
    assert redacted.lines[0].text == f"hey {pseudonym(2, SALT)}"


def test_names_that_are_common_words_are_left_alone_in_text(tmp_path: Path) -> None:
    conn = db(tmp_path)
    eid, _ = conversation(conn, [(1, "The", "the Indy boots"), (2, "dave", "yes")], 1)

    redacted = redact_conversation(messages_of(conn, eid), SALT)

    assert redacted.lines[0].text == "the Indy boots"
    assert redacted.lines[0].speaker == pseudonym(1, SALT)
    assert redacted.names == ["dave"]
