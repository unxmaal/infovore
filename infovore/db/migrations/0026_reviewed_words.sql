CREATE TABLE reviewed_words (
    id INTEGER PRIMARY KEY,
    word TEXT NOT NULL,
    tech INTEGER NOT NULL CHECK (tech IN (0, 1)),
    decided_at TEXT NOT NULL
);

CREATE INDEX reviewed_words_word ON reviewed_words (word, id);

CREATE TRIGGER reviewed_words_no_update BEFORE UPDATE ON reviewed_words BEGIN
    SELECT RAISE(ABORT, 'reviewed words are append-only');
END;

CREATE TRIGGER reviewed_words_no_delete BEFORE DELETE ON reviewed_words BEGIN
    SELECT RAISE(ABORT, 'reviewed words are append-only');
END;
