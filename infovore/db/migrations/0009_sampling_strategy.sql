CREATE TABLE extraction_batches (
  batch_id TEXT PRIMARY KEY,
  mode TEXT NOT NULL CHECK (mode IN ('trial', 'live')),
  strategy TEXT CHECK (strategy IN ('stratified', 'random', 'uncertain', 'mixed')),
  seed INTEGER,
  sample INTEGER,
  created_at TEXT NOT NULL
);
ALTER TABLE extraction_runs ADD COLUMN sampled_by TEXT;
