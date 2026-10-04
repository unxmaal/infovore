from datetime import UTC, datetime
from pathlib import Path

from infovore.db.reviewed_words import record_decisions
from infovore.triage.lexicon import load_lexicon, message_hits, tokens
from tests.triage.test_lexicon import corpus


def test_reviewed_words_join_the_lexicon_as_a_category_and_change_the_version(
    tmp_path: Path,
) -> None:
    conn = corpus(tmp_path / "x.db")
    base = load_lexicon()
    assert load_lexicon(conn).version == base.version
    assert load_lexicon(conn).sources["reviewed"] == 0

    record_decisions(conn, {"zorpish": True, "lunch": False, "disk": True}, datetime.now(UTC))
    merged = load_lexicon(conn)

    assert merged.version != base.version
    assert merged.version.startswith("lx-")
    assert merged.sources["reviewed"] == 1
    assert "zorpish" in merged.terms
    assert "lunch" not in merged.terms
    assert merged.size == base.size + 1
    assert message_hits(merged, "a zorpish thing") == ["zorpish"]


def test_tokens_reduce_urls_to_their_domain_word_and_drop_punctuation() -> None:
    assert "ebay" in tokens("see https://www.ebay.com/itm/123?x=1 now")
    assert "ebay" in tokens("http://ebay.co.uk/")
    assert "g5" in tokens("my G5 and 7200 ubr")
    assert "..." not in tokens("wait ... what")
    assert "-" not in tokens("a - b")
    assert tokens("a a b").count("a") == 2
