CREATE VIRTUAL TABLE messages_fts USING fts5 (
    content,
    content = 'messages',
    content_rowid = 'id',
    tokenize = "unicode61 tokenchars '-./_'"
);

-- The tokenchars are not cosmetic: without them `6.5.22m` splits into three
-- useless numbers, `030-1234-567` into three, and `/usr/people` into two.
-- Mirrors claims_fts so the two indexes tokenise the same corpus the same way.

CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts (rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts (messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

-- This trigger is a PRIVACY MECHANISM, not hygiene. privacy.optout.redact_stored
-- overwrites messages.content with '[redacted]'. An external-content FTS5 index
-- keeps its own tokens, so without this the old words stay searchable and a
-- query could prove what a redacted message said.
CREATE TRIGGER messages_fts_update AFTER UPDATE OF content ON messages BEGIN
    INSERT INTO messages_fts (messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts (rowid, content) VALUES (new.id, new.content);
END;

-- The 1,381,473 existing messages predate this index, so triggers alone would
-- index nothing.
INSERT INTO messages_fts (rowid, content) SELECT id, content FROM messages;
