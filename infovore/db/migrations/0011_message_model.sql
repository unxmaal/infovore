CREATE TABLE message_model (
  version INTEGER PRIMARY KEY AUTOINCREMENT,
  trained_at TEXT NOT NULL,
  labels_used INTEGER NOT NULL,
  holdout_size INTEGER NOT NULL,
  params_json TEXT NOT NULL
);
CREATE TABLE message_tokens (
  model_version INTEGER NOT NULL REFERENCES message_model (version),
  token TEXT NOT NULL,
  trash_count INTEGER NOT NULL,
  keep_count INTEGER NOT NULL,
  PRIMARY KEY (model_version, token)
);
