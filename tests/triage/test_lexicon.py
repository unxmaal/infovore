import sqlite3
from pathlib import Path

import pytest

from infovore.db.connection import migrate, open_database
from infovore.triage.human import load_gazetteer
from infovore.triage.lexicon import (
    LexiconError,
    load_lexicon,
    message_hits,
    mine_terms,
    parse_lexicon,
    score_lexicon,
)
from tests.triage.test_human import message

TOML = '[general]\nterms = ["disk", "scsi"]\n[mined]\nterms = ["nekoware"]\n'


def test_the_shipped_lexicon_has_every_source() -> None:
    lexicon = load_lexicon()

    assert lexicon.version.startswith("lx-")
    assert lexicon.sources["general"] > 100
    assert lexicon.sources["mined"] > 0
    assert lexicon.sources["gazetteer"] > 0
    assert lexicon.size == sum(lexicon.sources.values())


def test_the_version_changes_with_the_terms_and_the_gazetteer() -> None:
    gazetteer = load_gazetteer()
    a = parse_lexicon(TOML, gazetteer)
    b = parse_lexicon(TOML.replace("scsi", "scsi2"), gazetteer)

    assert a.version != b.version
    assert a.version == parse_lexicon(TOML, gazetteer).version


def test_a_malformed_lexicon_is_refused() -> None:
    gazetteer = load_gazetteer()
    with pytest.raises(LexiconError):
        parse_lexicon("[general]\nterms = []\n", gazetteer)
    with pytest.raises(LexiconError):
        parse_lexicon('[general]\nterms = [1]\n[mined]\nterms = []\n', gazetteer)


def test_hits_cover_terms_plurals_and_the_gazetteer() -> None:
    lexicon = parse_lexicon(TOML, load_gazetteer())

    assert message_hits(lexicon, "My Disks and a SCSI. chain") == ["disk", "scsi"]
    assert message_hits(lexicon, "running IRIX 6.5.22 on it") == ["gaz:irix_version"]
    assert message_hits(lexicon, "lol great lunch") == []


def test_the_score_is_the_share_of_messages_with_a_hit() -> None:
    lexicon = parse_lexicon(TOML, load_gazetteer())
    score = score_lexicon(lexicon, [message("scsi disk", mid=1), message("lunch", mid=2)])

    assert (score.share, score.hits, score.messages) == (0.5, 2, 2)
    empty = score_lexicon(lexicon, [])
    assert (empty.share, empty.hits, empty.messages) == (0.0, 0, 0)


def corpus(path: Path) -> sqlite3.Connection:
    conn = open_database(path)
    migrate(conn)
    for channel_id, name in ((1, "tech"), (2, "chat"), (3, "other")):
        conn.execute(
            "INSERT INTO channels (id, guild_id, parent_id, name, kind) VALUES (?, 1, NULL, ?, 'text')",
            (channel_id, name),
        )
    rows = [(1, "the zorp daemon crashed"), (1, "zorp again"), (2, "the lunch was good"),
            (2, "lunch again, zorp"), (3, "ignored zorp zorp")]
    for index, (channel_id, text) in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO messages (id, channel_id, guild_id, author_id, author_name_at_time,"
            " created_at, content, ingested_at, raw_json)"
            " VALUES (?, ?, 1, 1, 'a', '2026-01-01', ?, '2026-01-01', '{}')",
            (index, channel_id, text),
        )
    return conn


def test_mining_ranks_terms_by_log_odds_between_the_channel_sets(tmp_path: Path) -> None:
    lexicon = parse_lexicon(TOML, load_gazetteer())
    mined = mine_terms(corpus(tmp_path / "m.db"), ["tech"], ["chat"], min_count=2, lexicon=lexicon)

    assert [m.term for m in mined] == ["zorp"]
    assert (mined[0].tech, mined[0].off) == (2, 1)
    assert mined[0].log_odds > 0
    assert mine_terms(corpus(tmp_path / "m.db"), ["tech"], ["chat"], min_count=1, limit=1)[0].term == "zorp"
