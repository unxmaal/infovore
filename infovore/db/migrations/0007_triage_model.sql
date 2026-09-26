CREATE TABLE triage_model (
  version INTEGER PRIMARY KEY AUTOINCREMENT,
  trained_at TEXT NOT NULL,
  labels_used INTEGER NOT NULL,
  holdout_size INTEGER NOT NULL,
  params_json TEXT NOT NULL
);
CREATE TABLE triage_tokens (
  model_version INTEGER NOT NULL REFERENCES triage_model (version),
  token TEXT NOT NULL,
  lore_count INTEGER NOT NULL,
  noise_count INTEGER NOT NULL,
  PRIMARY KEY (model_version, token)
);
ALTER TABLE exchanges ADD COLUMN p_lore REAL;
ALTER TABLE exchanges ADD COLUMN p_lore_model INTEGER REFERENCES triage_model (version);
