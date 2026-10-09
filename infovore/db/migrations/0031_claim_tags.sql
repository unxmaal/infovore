CREATE TABLE tag_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_hash TEXT NOT NULL
);
CREATE INDEX tag_runs_model_prompt ON tag_runs (model_id, prompt_hash);

CREATE TABLE claim_tags (
    tag_run_id INTEGER NOT NULL REFERENCES tag_runs (id),
    claim_id INTEGER NOT NULL REFERENCES claims_v2 (id),
    tags_json TEXT NOT NULL,
    PRIMARY KEY (tag_run_id, claim_id)
);
CREATE INDEX claim_tags_claim ON claim_tags (claim_id);

CREATE TRIGGER tag_runs_no_update BEFORE UPDATE ON tag_runs BEGIN
    SELECT RAISE(ABORT, 'tag runs are append-only');
END;
CREATE TRIGGER tag_runs_no_delete BEFORE DELETE ON tag_runs BEGIN
    SELECT RAISE(ABORT, 'tag runs are append-only');
END;
CREATE TRIGGER claim_tags_no_update BEFORE UPDATE ON claim_tags BEGIN
    SELECT RAISE(ABORT, 'claim tags are append-only');
END;
CREATE TRIGGER claim_tags_no_delete BEFORE DELETE ON claim_tags BEGIN
    SELECT RAISE(ABORT, 'claim tags are append-only');
END;
