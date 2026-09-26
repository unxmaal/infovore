CREATE TABLE exchange_labels (
  exchange_id INTEGER NOT NULL REFERENCES exchanges (id),
  label TEXT NOT NULL CHECK (label IN ('lore', 'noise')),
  source TEXT NOT NULL CHECK (source IN ('llm', 'human')),
  source_ref TEXT,
  labeled_at TEXT NOT NULL,
  PRIMARY KEY (exchange_id, source)
);
CREATE INDEX exchange_labels_label ON exchange_labels (label);
