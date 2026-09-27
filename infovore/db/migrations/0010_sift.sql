CREATE TABLE message_labels (
  message_id INTEGER NOT NULL REFERENCES messages (id),
  label TEXT NOT NULL CHECK (label IN ('trash', 'keep')),
  source TEXT NOT NULL CHECK (source IN ('human', 'citation', 'rule')),
  source_ref TEXT,
  labeled_at TEXT NOT NULL,
  PRIMARY KEY (message_id, source)
);
CREATE INDEX message_labels_label ON message_labels (label);
ALTER TABLE messages ADD COLUMN p_trash REAL;
ALTER TABLE messages ADD COLUMN p_trash_model INTEGER;
