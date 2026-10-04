CREATE TABLE claim_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_id_source TEXT NOT NULL CHECK (model_id_source IN ('model_info', 'alias')),
    prompt_hash TEXT NOT NULL,
    selection TEXT NOT NULL,
    recipe_json TEXT NOT NULL
);

CREATE TABLE claim_run_exchanges (
    run_id INTEGER NOT NULL REFERENCES claim_runs (id),
    exchange_id INTEGER NOT NULL REFERENCES exchanges (id),
    outcome TEXT NOT NULL CHECK (outcome IN ('ok', 'failed')),
    error TEXT,
    windows INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    seconds REAL NOT NULL,
    PRIMARY KEY (run_id, exchange_id)
);

CREATE TABLE claims_v2 (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES claim_runs (id),
    exchange_id INTEGER NOT NULL REFERENCES exchanges (id),
    speaker TEXT NOT NULL,
    statement TEXT NOT NULL
);
CREATE INDEX claims_v2_run ON claims_v2 (run_id, id);

CREATE TABLE claims_v2_sources (
    claim_id INTEGER NOT NULL REFERENCES claims_v2 (id),
    message_id INTEGER NOT NULL REFERENCES messages (id),
    PRIMARY KEY (claim_id, message_id)
);

CREATE TABLE claim_rejections (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES claim_runs (id),
    exchange_id INTEGER NOT NULL REFERENCES exchanges (id),
    speaker TEXT NOT NULL,
    statement TEXT NOT NULL,
    cited TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE INDEX claim_rejections_run ON claim_rejections (run_id);

CREATE TABLE claim_reviews (
    id INTEGER PRIMARY KEY,
    claim_id INTEGER NOT NULL REFERENCES claims_v2 (id),
    verdict TEXT NOT NULL CHECK (verdict IN ('good', 'wrong', 'made_up', 'not_useful')),
    reviewed_at TEXT NOT NULL
);
CREATE INDEX claim_reviews_claim ON claim_reviews (claim_id, id);

CREATE VIEW current_claim_reviews AS
SELECT r.claim_id, r.verdict, r.reviewed_at
FROM claim_reviews r
WHERE r.id = (SELECT MAX(id) FROM claim_reviews WHERE claim_id = r.claim_id);

CREATE TRIGGER claim_runs_no_update BEFORE UPDATE ON claim_runs BEGIN
    SELECT RAISE(ABORT, 'claim runs are append-only');
END;
CREATE TRIGGER claim_runs_no_delete BEFORE DELETE ON claim_runs BEGIN
    SELECT RAISE(ABORT, 'claim runs are append-only');
END;
CREATE TRIGGER claim_run_exchanges_no_update BEFORE UPDATE ON claim_run_exchanges BEGIN
    SELECT RAISE(ABORT, 'claim run exchanges are append-only');
END;
CREATE TRIGGER claim_run_exchanges_no_delete BEFORE DELETE ON claim_run_exchanges BEGIN
    SELECT RAISE(ABORT, 'claim run exchanges are append-only');
END;
CREATE TRIGGER claims_v2_no_update BEFORE UPDATE ON claims_v2 BEGIN
    SELECT RAISE(ABORT, 'claims are append-only');
END;
CREATE TRIGGER claims_v2_no_delete BEFORE DELETE ON claims_v2 BEGIN
    SELECT RAISE(ABORT, 'claims are append-only');
END;
CREATE TRIGGER claim_reviews_no_update BEFORE UPDATE ON claim_reviews BEGIN
    SELECT RAISE(ABORT, 'claim reviews are append-only');
END;
CREATE TRIGGER claim_reviews_no_delete BEFORE DELETE ON claim_reviews BEGIN
    SELECT RAISE(ABORT, 'claim reviews are append-only');
END;
