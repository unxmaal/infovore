ALTER TABLE exchanges ADD COLUMN triage_score REAL;
ALTER TABLE exchanges ADD COLUMN triage_reasons TEXT;
ALTER TABLE exchanges ADD COLUMN triage_version TEXT;
CREATE INDEX exchanges_triage_score ON exchanges (triage_score);
