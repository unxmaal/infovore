-- rebuild: foreign_keys off

CREATE TABLE chunk_recipes_v2 (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    quiet_gap_seconds INTEGER NOT NULL,
    max_messages INTEGER NOT NULL,
    overlap INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    gap_percentile REAL NOT NULL DEFAULT 0,
    gap_floor_seconds INTEGER NOT NULL DEFAULT 0,
    gap_ceiling_seconds INTEGER NOT NULL DEFAULT 0,
    fold_factor REAL NOT NULL DEFAULT 0,
    fold_size INTEGER NOT NULL DEFAULT 0,
    UNIQUE (quiet_gap_seconds, max_messages, overlap, gap_percentile, gap_floor_seconds,
            gap_ceiling_seconds, fold_factor, fold_size)
);
INSERT INTO chunk_recipes_v2 (version, quiet_gap_seconds, max_messages, overlap, created_at)
SELECT version, quiet_gap_seconds, max_messages, overlap, created_at FROM chunk_recipes;
DROP TABLE chunk_recipes;
ALTER TABLE chunk_recipes_v2 RENAME TO chunk_recipes;

-- Per-channel gap = 95th percentile of the channel's own gaps up to 6 h, floored at 30 min;
-- groups of at most 2 messages fold into a larger neighbour within 4 x that gap.
INSERT INTO chunk_recipes (version, quiet_gap_seconds, max_messages, overlap, created_at,
                           gap_percentile, gap_floor_seconds, gap_ceiling_seconds,
                           fold_factor, fold_size)
VALUES (2, 1800, 50, 3, '2026-10-03T00:00:00+00:00', 95, 1800, 21600, 4, 2);

ALTER TABLE exchanges ADD COLUMN superseded_by_recipe INTEGER REFERENCES chunk_recipes (version);

CREATE TABLE superseded_exchange_messages (
    exchange_id INTEGER NOT NULL REFERENCES exchanges (id),
    message_id INTEGER NOT NULL REFERENCES messages (id),
    position INTEGER NOT NULL,
    PRIMARY KEY (exchange_id, message_id)
);

CREATE VIEW all_exchange_messages AS
SELECT exchange_id, message_id, position FROM exchange_messages
UNION ALL
SELECT exchange_id, message_id, position FROM superseded_exchange_messages;

CREATE TABLE channel_chunk_gaps (
    recipe INTEGER NOT NULL REFERENCES chunk_recipes (version),
    channel_id INTEGER NOT NULL,
    gap_seconds INTEGER NOT NULL,
    PRIMARY KEY (recipe, channel_id)
);

CREATE TABLE exchange_remap (
    recipe INTEGER NOT NULL REFERENCES chunk_recipes (version),
    old_exchange_id INTEGER NOT NULL REFERENCES exchanges (id),
    new_exchange_id INTEGER REFERENCES exchanges (id),
    kind TEXT NOT NULL CHECK (kind IN ('one_to_one', 'merged', 'split', 'ambiguous')),
    shared_messages INTEGER NOT NULL,
    old_messages INTEGER NOT NULL,
    PRIMARY KEY (recipe, old_exchange_id)
);

CREATE TRIGGER exchange_remap_no_update BEFORE UPDATE ON exchange_remap BEGIN
    SELECT RAISE(ABORT, 'the exchange remap is append-only');
END;

CREATE TRIGGER exchange_remap_no_delete BEFORE DELETE ON exchange_remap BEGIN
    SELECT RAISE(ABORT, 'the exchange remap is append-only');
END;

-- A slice rebuilt for recipe N is stored as 'name@N'; the frozen rows are never touched.
CREATE VIEW current_eval_slices AS
WITH versions AS (
    SELECT DISTINCT name AS stored_name,
        CASE WHEN instr(name, '@') > 0 THEN substr(name, 1, instr(name, '@') - 1) ELSE name END
            AS base,
        CASE WHEN instr(name, '@') > 0
            THEN CAST(substr(name, instr(name, '@') + 1) AS INTEGER) ELSE 1 END AS recipe
    FROM eval_slices
)
SELECT v.base AS name, s.exchange_id, s.position, s.population, s.seed, s.frozen_at
FROM eval_slices s
JOIN versions v ON v.stored_name = s.name
WHERE v.recipe = (SELECT MAX(w.recipe) FROM versions w WHERE w.base = v.base);
