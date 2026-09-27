ALTER TABLE extraction_runs ADD COLUMN batch_id TEXT;
CREATE INDEX extraction_runs_batch_id ON extraction_runs (batch_id);
