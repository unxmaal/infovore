-- Probe cost was unmeasurable: the probe makes two LLM calls per claim and
-- recorded neither. A probe_run is one call pair (recall + judge), covering
-- however many claims shared it, so SUM(cost_usd) over this table is the
-- probe's true cost with no double counting.

CREATE TABLE probe_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    probe_model TEXT,
    judge_model TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    claim_count INTEGER NOT NULL CHECK (claim_count > 0),
    batched INTEGER NOT NULL CHECK (batched IN (0, 1)),
    recall_input_tokens INTEGER,
    recall_output_tokens INTEGER,
    judge_input_tokens INTEGER,
    judge_output_tokens INTEGER,
    cost_usd REAL,
    outcome TEXT NOT NULL CHECK (outcome IN ('ok', 'failed')),
    error TEXT
);

ALTER TABLE claims ADD COLUMN probe_run_id INTEGER REFERENCES probe_runs (id);
CREATE INDEX claims_probe_run ON claims (probe_run_id);

ALTER TABLE extraction_runs ADD COLUMN cost_usd REAL;
