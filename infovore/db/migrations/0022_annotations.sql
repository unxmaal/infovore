CREATE TABLE annotations (
    id INTEGER PRIMARY KEY,
    subject_kind TEXT NOT NULL CHECK (subject_kind IN ('exchange', 'message', 'claim')),
    subject_id INTEGER NOT NULL,
    scorer TEXT NOT NULL,
    scorer_version INTEGER NOT NULL,
    reproducibility TEXT NOT NULL CHECK (reproducibility IN ('derived', 'recorded')),
    score REAL,
    label TEXT,
    recipe_json TEXT,
    source_ref TEXT,
    created_at TEXT NOT NULL,
    CHECK (score IS NOT NULL OR label IS NOT NULL),
    -- A derived value without its recipe is the defect this table exists to fix:
    -- four triage models trained, three with no recoverable trace.
    CHECK (reproducibility <> 'derived' OR recipe_json IS NOT NULL)
);

-- A derived score is a cache, so one answer per subject per scorer version.
-- Recorded annotations deliberately have NO such constraint: repeat human
-- judgments on the same item are the self-consistency measurement, not a bug.
CREATE UNIQUE INDEX annotations_one_derived_per_version
    ON annotations (subject_kind, subject_id, scorer, scorer_version)
    WHERE reproducibility = 'derived';

CREATE INDEX annotations_subject ON annotations (subject_kind, subject_id, scorer);
CREATE INDEX annotations_scorer ON annotations (scorer, scorer_version, subject_kind);

-- Append-only: a value that can be overwritten in place is exactly how fifteen
-- message models and three triage models left nothing behind.
CREATE TRIGGER annotations_no_update BEFORE UPDATE ON annotations BEGIN
    SELECT RAISE(ABORT, 'annotations are append-only');
END;

-- Deletion splits on reproducibility, which is the whole point of the column.
-- A derived score is droppable because it can be recomputed from its recipe.
-- A claim, a human label or an LLM verdict cannot be recomputed ever.
CREATE TRIGGER annotations_recorded_no_delete BEFORE DELETE ON annotations
WHEN old.reproducibility = 'recorded' BEGIN
    SELECT RAISE(ABORT, 'a recorded annotation cannot be deleted; it cannot be recomputed');
END;

-- "Which gate was live in September" must be answerable from the database
-- rather than from git history, so activation is a fact with a time, not a
-- column that gets overwritten.
CREATE TABLE scorer_activations (
    id INTEGER PRIMARY KEY,
    scorer TEXT NOT NULL,
    scorer_version INTEGER NOT NULL,
    activated_at TEXT NOT NULL,
    note TEXT
);

CREATE INDEX scorer_activations_scorer ON scorer_activations (scorer, activated_at);

CREATE TRIGGER scorer_activations_no_update BEFORE UPDATE ON scorer_activations BEGIN
    SELECT RAISE(ABORT, 'scorer activations are append-only');
END;

CREATE TRIGGER scorer_activations_no_delete BEFORE DELETE ON scorer_activations BEGIN
    SELECT RAISE(ABORT, 'scorer activations are append-only');
END;

-- The live answer per scorer, without a band column: bands discard ordering
-- within a band, and ranking is what pays (the gate scored 0.917 AUC while
-- delivering no economic advantage on a gain curve).
CREATE VIEW current_annotations AS
SELECT a.*
FROM annotations a
JOIN scorer_activations s
  ON s.scorer = a.scorer AND s.scorer_version = a.scorer_version
WHERE s.activated_at = (
    SELECT MAX(activated_at) FROM scorer_activations WHERE scorer = a.scorer
);

-- Backfill. Only the CURRENTLY stored values are recoverable: p_lore holds one
-- answer per exchange, so triage models 1-3 are not here. Their params_json
-- survives in triage_model, which is what makes them recomputable rather than
-- lost. created_at is the row's own write time, and the recipe says the value
-- predates it.
INSERT INTO annotations (
    subject_kind, subject_id, scorer, scorer_version, reproducibility,
    score, recipe_json, source_ref, created_at
)
SELECT
    'exchange', e.id, 'p_lore', e.p_lore_model, 'derived', e.p_lore,
    json_object(
        'model_table', 'triage_model',
        'model_version', e.p_lore_model,
        'params_in', 'triage_model.params_json',
        'value_predates_row', json('true')
    ),
    'migration:0022',
    strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now')
FROM exchanges e
WHERE e.p_lore IS NOT NULL AND e.p_lore_model IS NOT NULL;

INSERT INTO annotations (
    subject_kind, subject_id, scorer, scorer_version, reproducibility,
    score, recipe_json, source_ref, created_at
)
SELECT
    'message', m.id, 'p_trash', m.p_trash_model, 'derived', m.p_trash,
    json_object(
        'model_table', 'message_combiner',
        'model_version', m.p_trash_model,
        'params_in', 'message_combiner.params_json',
        'value_predates_row', json('true')
    ),
    'migration:0022',
    strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now')
FROM messages m
WHERE m.p_trash IS NOT NULL AND m.p_trash_model IS NOT NULL;

-- The versions those backfilled values were produced under are, by definition,
-- the ones live right now. trained_at is a lower bound on when scoring
-- happened, which is the best the old schema recorded.
INSERT INTO scorer_activations (scorer, scorer_version, activated_at, note)
SELECT 'p_lore', t.version, t.trained_at, 'backfilled by migration 0022'
FROM triage_model t
WHERE t.version = (SELECT MAX(p_lore_model) FROM exchanges WHERE p_lore IS NOT NULL);

INSERT INTO scorer_activations (scorer, scorer_version, activated_at, note)
SELECT 'p_trash', c.version, c.trained_at, 'backfilled by migration 0022'
FROM message_combiner c
WHERE c.version = (SELECT MAX(p_trash_model) FROM messages WHERE p_trash IS NOT NULL);
