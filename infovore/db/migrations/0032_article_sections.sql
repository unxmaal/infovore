CREATE TABLE article_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    tag_run_id INTEGER NOT NULL REFERENCES tag_runs (id)
);
CREATE INDEX article_runs_model_prompt ON article_runs (model_id, prompt_hash, tag_run_id);

CREATE TABLE article_sections (
    article_run_id INTEGER NOT NULL REFERENCES article_runs (id),
    topic TEXT NOT NULL,
    section TEXT NOT NULL,
    sentences_json TEXT NOT NULL,
    dropped INTEGER NOT NULL,
    PRIMARY KEY (article_run_id, topic, section)
);

CREATE TRIGGER article_runs_no_update BEFORE UPDATE ON article_runs BEGIN
    SELECT RAISE(ABORT, 'article runs are append-only');
END;
CREATE TRIGGER article_runs_no_delete BEFORE DELETE ON article_runs BEGIN
    SELECT RAISE(ABORT, 'article runs are append-only');
END;
CREATE TRIGGER article_sections_no_update BEFORE UPDATE ON article_sections BEGIN
    SELECT RAISE(ABORT, 'article sections are append-only');
END;
CREATE TRIGGER article_sections_no_delete BEFORE DELETE ON article_sections BEGIN
    SELECT RAISE(ABORT, 'article sections are append-only');
END;
