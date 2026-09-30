CREATE TABLE label_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES messages (id),
    label TEXT NOT NULL CHECK (label IN ('trash', 'keep')),
    source_ref TEXT,
    labeled_at TEXT NOT NULL
);

CREATE INDEX label_events_message ON label_events (message_id, id);
CREATE INDEX label_events_ref ON label_events (source_ref);

INSERT INTO label_events (message_id, label, source_ref, labeled_at)
SELECT message_id, label, source_ref, labeled_at
FROM message_labels
WHERE source = 'human'
ORDER BY labeled_at, message_id;

CREATE TABLE gate_predictions (
    technique TEXT NOT NULL,
    message_id INTEGER NOT NULL REFERENCES messages (id),
    score REAL,
    predicted TEXT NOT NULL CHECK (predicted IN ('trash', 'keep')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (technique, message_id)
);

CREATE INDEX gate_predictions_technique ON gate_predictions (technique);
