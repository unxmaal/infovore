CREATE TABLE chunk_recipes (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    quiet_gap_seconds INTEGER NOT NULL,
    max_messages INTEGER NOT NULL,
    overlap INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (quiet_gap_seconds, max_messages, overlap)
);

-- The 203,634 existing exchanges were all produced with a 30 minute quiet gap,
-- a 50 message cap and 3 messages of overlap. Those constants were introduced
-- in one commit and never changed, so this recovers what was decided rather
-- than inventing a recipe nobody chose.
INSERT INTO chunk_recipes (version, quiet_gap_seconds, max_messages, overlap, created_at)
VALUES (1, 1800, 50, 3, '2026-10-01T00:00:00+00:00');

ALTER TABLE exchanges ADD COLUMN chunk_recipe INTEGER REFERENCES chunk_recipes (version);

UPDATE exchanges SET chunk_recipe = 1;
