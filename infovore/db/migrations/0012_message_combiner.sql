ALTER TABLE message_model ADD COLUMN kind TEXT NOT NULL DEFAULT 'legacy';
CREATE TABLE message_combiner (
  version INTEGER PRIMARY KEY AUTOINCREMENT,
  trained_at TEXT NOT NULL,
  citation_model_version INTEGER NOT NULL REFERENCES message_model (version),
  human_model_version INTEGER REFERENCES message_model (version),
  fallback INTEGER NOT NULL DEFAULT 0,
  params_json TEXT NOT NULL
);
