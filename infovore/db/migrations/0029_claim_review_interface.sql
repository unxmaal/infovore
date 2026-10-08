ALTER TABLE claim_reviews ADD COLUMN interface TEXT NOT NULL DEFAULT 'cited-only'
    CHECK (interface IN ('cited-only', 'conversation'));

DROP VIEW current_claim_reviews;
CREATE VIEW current_claim_reviews AS
SELECT r.claim_id, r.verdict, r.reviewed_at, r.interface
FROM claim_reviews r
WHERE r.id = (SELECT MAX(id) FROM claim_reviews WHERE claim_id = r.claim_id);
