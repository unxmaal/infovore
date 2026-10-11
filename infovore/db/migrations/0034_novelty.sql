CREATE TABLE reference_texts (
    topic TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    text TEXT NOT NULL,
    match REAL NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE claim_novelty (
    claim_id INTEGER PRIMARY KEY REFERENCES claims_v2 (id),
    reference_topic TEXT,
    reference_title TEXT,
    max_cosine REAL,
    nearest TEXT,
    generic_share REAL NOT NULL,
    computed_at TEXT NOT NULL
);
