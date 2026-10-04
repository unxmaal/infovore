CREATE TABLE claim_checks (
    id INTEGER PRIMARY KEY,
    claim_id INTEGER NOT NULL REFERENCES claims_v2 (id),
    verdict TEXT NOT NULL CHECK (verdict IN ('supported', 'unsupported_fact', 'low_overlap', 'uncheckable')),
    overlap REAL NOT NULL,
    facts_json TEXT NOT NULL,
    recipe_json TEXT NOT NULL,
    checked_at TEXT NOT NULL
);
CREATE INDEX claim_checks_claim ON claim_checks (claim_id, id);

CREATE VIEW current_claim_checks AS
SELECT c.claim_id, c.verdict, c.overlap, c.facts_json, c.recipe_json, c.checked_at
FROM claim_checks c
WHERE c.id = (SELECT MAX(id) FROM claim_checks WHERE claim_id = c.claim_id);

CREATE TRIGGER claim_checks_no_update BEFORE UPDATE ON claim_checks BEGIN
    SELECT RAISE(ABORT, 'claim checks are append-only');
END;
CREATE TRIGGER claim_checks_no_delete BEFORE DELETE ON claim_checks BEGIN
    SELECT RAISE(ABORT, 'claim checks are append-only');
END;
