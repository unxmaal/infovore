CREATE TABLE speaker_drops (
    id INTEGER PRIMARY KEY,
    author_id INTEGER NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('drop', 'keep')),
    decided_at TEXT NOT NULL
);
CREATE INDEX speaker_drops_author ON speaker_drops (author_id, id);

CREATE VIEW current_speaker_drops AS
SELECT d.author_id, d.decision, d.decided_at
FROM speaker_drops d
WHERE d.id = (SELECT MAX(id) FROM speaker_drops WHERE author_id = d.author_id);

CREATE TRIGGER speaker_drops_no_update BEFORE UPDATE ON speaker_drops BEGIN
    SELECT RAISE(ABORT, 'speaker drops are append-only');
END;
CREATE TRIGGER speaker_drops_no_delete BEFORE DELETE ON speaker_drops BEGIN
    SELECT RAISE(ABORT, 'speaker drops are append-only');
END;
