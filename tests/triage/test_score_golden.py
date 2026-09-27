"""Golden regression test for `score_exchange` (issue #95, deliverable 1).

Pins the exact `TriageResult` (score *and* reasons, in order) that today's
hard-coded `infovore/triage/score.py` produces for a diverse set of fixtures:
domain terms (single/multiple/capped), IRIX versions, part numbers, unix
paths, code, archive links, gif links, laughter, tiny-message noise,
answered/unanswered/self-answered questions, agreed answers, threads,
substantial length, PDF attachments, and a combined "everything" case.

This must keep passing, unchanged, after rules.toml/TriageRules replace the
hard-coded constants: the shipped default rules are required to reproduce
today's scores exactly. Do not "fix" this test to match new output — if it
fails, the refactor changed behavior it shouldn't have.
"""

from datetime import UTC, datetime, timedelta

from infovore.rows import AttachmentRow, MessageRow, ReactionRow
from infovore.triage.score import TriageResult, score_exchange

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


def test_golden_empty_exchange() -> None:
    assert score_exchange([]) == TriageResult(0.0, ())


def test_golden_chatter_only() -> None:
    result = score_exchange([msg(1, "lol"), msg(2, "haha same", author_id=2), msg(3, "gm")])
    assert result == TriageResult(0.0, (("mostly_tiny_messages", -0.2), ("laughter", -0.1)))


def test_golden_single_domain_term() -> None:
    result = score_exchange([msg(1, "my Octane is back from the shop, very happy with it")])
    assert result == TriageResult(0.15, (("domain_terms", 0.15),))


def test_golden_three_domain_terms() -> None:
    result = score_exchange(
        [msg(1, "swapped the Octane PROM and ran hinv, the IP30 board is fine now")]
    )
    assert result == TriageResult(0.45, (("domain_terms", 0.45),))


def test_golden_many_domain_terms_capped() -> None:
    result = score_exchange(
        [
            msg(
                1,
                "Indy Indigo2 O2 Octane Fuel Tezro Onyx Origin hinv inst swmgr nvram PROM XFS"
                " MIPSpro IP22 IP30 IP35 R10000 R12k",
            )
        ]
    )
    assert result == TriageResult(0.45, (("domain_terms", 0.45),))


def test_golden_irix_version() -> None:
    result = score_exchange([msg(1, "upgrade to 6.5.22 first, then 6.5.30")])
    assert result == TriageResult(0.2, (("irix_version", 0.2),))


def test_golden_irix_version_old_style() -> None:
    result = score_exchange([msg(1, "this was on IRIX 5.3 originally")])
    assert result == TriageResult(0.35, (("domain_terms", 0.15), ("irix_version", 0.2)))


def test_golden_part_number() -> None:
    result = score_exchange([msg(1, "the PSU is 060-0035-003 not the older one")])
    assert result == TriageResult(0.3, (("part_number", 0.3),))


def test_golden_unix_path() -> None:
    result = score_exchange([msg(1, "check /var/adm/SYSLOG after boot")])
    assert result == TriageResult(0.15, (("unix_path", 0.15),))


def test_golden_code_inline() -> None:
    result = score_exchange([msg(1, "run `hinv -vm` and paste it")])
    assert result == TriageResult(0.3, (("domain_terms", 0.15), ("code", 0.15)))


def test_golden_code_block() -> None:
    result = score_exchange([msg(1, "```\nsetenv console d\n```")])
    assert result == TriageResult(0.15, (("code", 0.15),))


def test_golden_archive_link() -> None:
    result = score_exchange([msg(1, "manual: https://techpubs.example/sgi/octane.pdf")])
    assert result == TriageResult(0.45, (("domain_terms", 0.3), ("archive_link", 0.15)))


def test_golden_generic_numbers_and_links_are_not_specifics() -> None:
    result = score_exchange(
        [msg(1, "4.5 stars, call me at 555-1234, see https://example.com/cats")]
    )
    assert result == TriageResult(0.0, ())


def test_golden_answered_question() -> None:
    result = score_exchange(
        [
            msg(1, "how do I reset the PROM password on an O2?"),
            msg(
                2,
                "hold the reset button and use the command monitor to clear it",
                author_id=2,
            ),
        ]
    )
    assert result == TriageResult(0.5, (("domain_terms", 0.3), ("answered_question", 0.2)))


def test_golden_unanswered_question() -> None:
    result = score_exchange([msg(1, "how do I reset the PROM password on an O2?")])
    assert result == TriageResult(0.3, (("domain_terms", 0.3),))


def test_golden_self_reply_not_answered() -> None:
    result = score_exchange(
        [
            msg(1, "how do I reset the PROM password on an O2?"),
            msg(
                2,
                "never mind, found it in the manual after some digging",
                author_id=1,
            ),
        ]
    )
    assert result == TriageResult(0.3, (("domain_terms", 0.3),))


def test_golden_thread_and_substantial() -> None:
    result = score_exchange([msg(1, "the Octane needs the right PSU revision " * 12, thread_id=77)])
    assert result == TriageResult(
        0.3, (("domain_terms", 0.15), ("thread", 0.05), ("substantial", 0.1))
    )


def test_golden_pdf_attachment() -> None:
    pdf = AttachmentRow(1, 1, "octane_owners_guide.pdf", "application/pdf", 10, "u", None, None)
    result = score_exchange([msg(1, "here you go")], attachments=[pdf])
    assert result == TriageResult(0.15, (("pdf_attachment", 0.15),))


def test_golden_agreed_answer() -> None:
    agree = ReactionRow(2, "✅", 3)
    result = score_exchange(
        [msg(1, "which PSU?"), msg(2, "the 060-0035-003 one", author_id=2)],
        reactions=[agree],
    )
    assert result == TriageResult(0.35, (("part_number", 0.3), ("agreed_answer", 0.05)))


def test_golden_noise_penalties() -> None:
    result = score_exchange(
        [msg(1, "swapped the Octane PROM and ran hinv")]
        + [msg(i, "lol", author_id=i) for i in range(2, 12)]
        + [msg(12, "https://tenor.com/view/cat-12345")]
    )
    assert result == TriageResult(
        0.05,
        (
            ("domain_terms", 0.44999999999999996),
            ("mostly_tiny_messages", -0.2),
            ("gif_links", -0.1),
            ("laughter", -0.1),
        ),
    )


def test_golden_gif_link_giphy() -> None:
    result = score_exchange(
        [
            msg(1, "swapped the Octane PROM and ran hinv"),
            msg(2, "https://giphy.com/gifs/cat-party"),
        ]
    )
    assert result == TriageResult(
        0.35, (("domain_terms", 0.44999999999999996), ("gif_links", -0.1))
    )


def test_golden_everything_maxed() -> None:
    result = score_exchange(
        [
            msg(
                1,
                "Octane Fuel Tezro Onyx Origin IP30 hinv PROM 6.5.22 060-0035-003 /usr/sbin/inst",
            ),
            msg(
                2,
                "```\nnvram netaddr\n``` https://bitsavers.org/pdf/sgi/x.pdf " * 20,
                author_id=2,
            ),
        ]
    )
    assert result == TriageResult(
        1.0,
        (
            ("domain_terms", 0.45),
            ("irix_version", 0.2),
            ("part_number", 0.3),
            ("unix_path", 0.15),
            ("code", 0.15),
            ("archive_link", 0.15),
            ("substantial", 0.1),
        ),
    )


def test_golden_laughter_share() -> None:
    result = score_exchange(
        [
            msg(1, "octane pdu question"),
            msg(2, "lol", author_id=2),
            msg(3, "lmao", author_id=3),
            msg(4, "haha", author_id=4),
        ]
    )
    assert result == TriageResult(
        0.0,
        (
            ("domain_terms", 0.15),
            ("mostly_tiny_messages", -0.2),
            ("laughter", -0.1),
        ),
    )
