CREATE TABLE prompt_versions (
    version TEXT PRIMARY KEY,
    text_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    promoted_at TEXT
);

CREATE TABLE extraction_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_id INTEGER NOT NULL REFERENCES exchanges (id),
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL REFERENCES prompt_versions (version),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    mode TEXT NOT NULL CHECK (mode IN ('trial', 'live')),
    outcome TEXT NOT NULL CHECK (outcome IN ('ok', 'failed')),
    error TEXT
);
CREATE INDEX extraction_runs_exchange ON extraction_runs (exchange_id);

CREATE TABLE claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_id INTEGER NOT NULL REFERENCES exchanges (id),
    extraction_run_id INTEGER NOT NULL REFERENCES extraction_runs (id),
    statement TEXT NOT NULL,
    subject TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('fact', 'correction', 'procedure', 'reference')),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    probe_question TEXT NOT NULL,
    permalink TEXT NOT NULL,
    supersedes_claim_id INTEGER REFERENCES claims (id),
    novelty TEXT NOT NULL DEFAULT 'unprobed'
        CHECK (novelty IN ('unprobed', 'unknown', 'partial', 'contradicts', 'known')),
    probe_model TEXT,
    probe_answer TEXT,
    probed_at TEXT,
    probe_error TEXT,
    retracted_at TEXT,
    retraction_reason TEXT
);
CREATE INDEX claims_exchange ON claims (exchange_id);
CREATE INDEX claims_run ON claims (extraction_run_id);
CREATE INDEX claims_novelty ON claims (novelty);
CREATE INDEX claims_supersedes ON claims (supersedes_claim_id);

CREATE TABLE claim_sources (
    claim_id INTEGER NOT NULL REFERENCES claims (id),
    message_id INTEGER NOT NULL REFERENCES messages (id),
    PRIMARY KEY (claim_id, message_id)
);
CREATE INDEX claim_sources_message ON claim_sources (message_id);

CREATE VIRTUAL TABLE claims_fts USING fts5 (
    subject,
    statement,
    content = 'claims',
    content_rowid = 'id',
    tokenize = "unicode61 tokenchars '-./_'"
);

CREATE TRIGGER claims_fts_insert AFTER INSERT ON claims BEGIN
    INSERT INTO claims_fts (rowid, subject, statement)
    VALUES (new.id, new.subject, new.statement);
END;

CREATE TRIGGER claims_fts_delete AFTER DELETE ON claims BEGIN
    INSERT INTO claims_fts (claims_fts, rowid, subject, statement)
    VALUES ('delete', old.id, old.subject, old.statement);
END;

CREATE TRIGGER claims_fts_update AFTER UPDATE OF subject, statement ON claims BEGIN
    INSERT INTO claims_fts (claims_fts, rowid, subject, statement)
    VALUES ('delete', old.id, old.subject, old.statement);
    INSERT INTO claims_fts (rowid, subject, statement)
    VALUES (new.id, new.subject, new.statement);
END;
