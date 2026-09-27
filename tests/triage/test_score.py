from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from infovore.rows import AttachmentRow, MessageRow, ReactionRow
from infovore.triage.rules import DEFAULT_RULES
from infovore.triage.score import TRIAGE_VERSION, score_exchange

START = datetime(2026, 1, 1, tzinfo=UTC)


def msg(
    message_id: int,
    content: str,
    author_id: int = 1,
    thread_id: int | None = None,
) -> MessageRow:
    return MessageRow(
        id=message_id,
        channel_id=10,
        guild_id=9,
        author_id=author_id,
        author_name_at_time=f"user{author_id}",
        author_is_bot=False,
        created_at=START + timedelta(minutes=message_id),
        edited_at=None,
        content=content,
        reply_to_id=None,
        thread_id=thread_id,
        deleted_at=None,
        ingested_at=START,
        raw_json="{}",
    )


def signals(*messages: MessageRow, **kwargs: object) -> set[str]:
    result = score_exchange(list(messages), **kwargs)  # type: ignore[arg-type]
    return {name for name, _ in result.reasons}


def test_version_is_derived_from_the_rules_content_hash() -> None:
    assert DEFAULT_RULES.version == TRIAGE_VERSION
    assert TRIAGE_VERSION.startswith("r-")


def test_chatter_scores_zero() -> None:
    result = score_exchange([msg(1, "lol"), msg(2, "haha same", author_id=2), msg(3, "gm")])
    assert result.score == 0.0


def test_domain_terms_count_distinct_hits_up_to_a_cap() -> None:
    one = score_exchange([msg(1, "my Octane is back from the shop, very happy with it")])
    three = score_exchange(
        [msg(1, "swapped the Octane PROM and ran hinv, the IP30 board is fine now")]
    )
    many = score_exchange(
        [
            msg(
                1,
                "Indy Indigo2 O2 Octane Fuel Tezro Onyx Origin hinv inst swmgr nvram PROM XFS"
                " MIPSpro IP22 IP30 IP35 R10000 R12k",
            )
        ]
    )
    assert 0 < one.score < three.score
    assert dict(many.reasons)["domain_terms"] == pytest.approx(0.45)


def test_specifics() -> None:
    assert "irix_version" in signals(msg(1, "upgrade to 6.5.22 first, then 6.5.30"))
    assert "irix_version" in signals(msg(1, "this was on IRIX 5.3 originally"))
    assert "part_number" in signals(msg(1, "the PSU is 060-0035-003 not the older one"))
    assert "unix_path" in signals(msg(1, "check /var/adm/SYSLOG after boot"))
    assert "code" in signals(msg(1, "run `hinv -vm` and paste it"))
    assert "code" in signals(msg(1, "```\nsetenv console d\n```"))
    assert "archive_link" in signals(msg(1, "manual: https://techpubs.example/sgi/octane.pdf"))


def test_generic_numbers_and_links_are_not_specifics() -> None:
    found = signals(msg(1, "4.5 stars, call me at 555-1234, see https://example.com/cats"))
    assert not found & {"irix_version", "part_number", "archive_link"}


def test_question_answered_by_someone_else() -> None:
    answered = signals(
        msg(1, "how do I reset the PROM password on an O2?"),
        msg(2, "hold the reset button and use the command monitor to clear it", author_id=2),
    )
    unanswered = signals(msg(1, "how do I reset the PROM password on an O2?"))
    self_reply = signals(
        msg(1, "how do I reset the PROM password on an O2?"),
        msg(2, "never mind, found it in the manual after some digging", author_id=1),
    )
    assert "answered_question" in answered
    assert "answered_question" not in unanswered
    assert "answered_question" not in self_reply


def test_structure_bonuses() -> None:
    long_text = "the Octane needs the right PSU revision " * 12
    assert "thread" in signals(msg(1, "Octane PSU notes", thread_id=77))
    assert "substantial" in signals(msg(1, long_text))
    pdf = AttachmentRow(1, 1, "octane_owners_guide.pdf", "application/pdf", 10, "u", None, None)
    assert "pdf_attachment" in signals(msg(1, "here you go"), attachments=[pdf])
    agree = ReactionRow(2, "✅", 3)
    assert "agreed_answer" in signals(
        msg(1, "which PSU?"), msg(2, "the 060-0035-003 one", author_id=2), reactions=[agree]
    )


def test_noise_penalties_pull_scores_down() -> None:
    signal = "swapped the Octane PROM and ran hinv"
    clean = score_exchange([msg(1, signal)])
    noisy = score_exchange(
        [msg(1, signal)]
        + [msg(i, "lol", author_id=i) for i in range(2, 12)]
        + [msg(12, "https://tenor.com/view/cat-12345")]
    )
    reasons = dict(noisy.reasons)
    assert reasons["mostly_tiny_messages"] < 0
    assert reasons["gif_links"] < 0
    assert reasons["laughter"] < 0
    assert noisy.score < clean.score


def test_score_is_bounded_and_deterministic() -> None:
    everything = [
        msg(1, "Octane Fuel Tezro Onyx Origin IP30 hinv PROM 6.5.22 060-0035-003 /usr/sbin/inst"),
        msg(2, "```\nnvram netaddr\n``` https://bitsavers.org/pdf/sgi/x.pdf " * 20, author_id=2),
    ]
    first = score_exchange(everything)
    assert first.score == 1.0
    assert score_exchange(everything) == first


def test_empty_exchange_scores_zero() -> None:
    assert score_exchange([]).score == 0.0


def test_score_exchange_defaults_to_default_rules() -> None:
    default_result = score_exchange([msg(1, "here you go")])
    explicit_result = score_exchange([msg(1, "here you go")], rules=DEFAULT_RULES)
    assert default_result == explicit_result


def test_score_exchange_uses_provided_rules_weight() -> None:
    pdf = AttachmentRow(1, 1, "manual.pdf", "application/pdf", 10, "u", None, None)
    custom = replace(DEFAULT_RULES, pdf_attachment_weight=0.9, version="custom-a")
    result = score_exchange([msg(1, "here you go")], attachments=[pdf], rules=custom)
    assert dict(result.reasons)["pdf_attachment"] == 0.9


def test_score_exchange_domain_terms_are_driven_by_rules() -> None:
    custom = replace(
        DEFAULT_RULES, domain_terms=(*DEFAULT_RULES.domain_terms, "zorptech"), version="custom-b"
    )
    with_term = score_exchange([msg(1, "got a zorptech unit today")], rules=custom)
    without_term = score_exchange([msg(1, "got a zorptech unit today")])
    assert "domain_terms" in dict(with_term.reasons)
    assert "domain_terms" not in dict(without_term.reasons)


def test_score_exchange_archive_and_gif_hosts_are_driven_by_rules() -> None:
    custom = replace(
        DEFAULT_RULES,
        archive_link_hosts=("example-archive",),
        gif_hosts=("example-gif",),
        version="custom-c",
    )
    archive = signals(msg(1, "see https://example-archive.test/manual"), rules=custom)
    gif = signals(msg(1, "https://example-gif.test/cat"), rules=custom)
    assert "archive_link" in archive
    assert "gif_links" in gif


def test_score_exchange_laughter_tokens_are_driven_by_rules() -> None:
    custom = replace(DEFAULT_RULES, laughter_tokens=("giggle",), version="custom-d")
    found = signals(
        msg(1, "octane pdu"),
        msg(2, "giggle", author_id=2),
        msg(3, "giggle", author_id=3),
        rules=custom,
    )
    assert "laughter" in found
