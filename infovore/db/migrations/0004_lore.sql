CREATE VIEW lore AS
SELECT
    c.id AS claim_id,
    c.subject AS subject,
    c.statement AS statement,
    c.kind AS kind,
    c.confidence AS confidence,
    c.novelty AS novelty,
    c.permalink AS permalink,
    (
        SELECT group_concat(message_id, ',')
        FROM (
            SELECT cs.message_id AS message_id
            FROM claim_sources cs
            WHERE cs.claim_id = c.id
            ORDER BY cs.message_id
        )
    ) AS source_message_ids,
    e.channel_id AS channel_id,
    r.started_at AS extracted_at,
    c.supersedes_claim_id AS supersedes_claim_id
FROM claims c
JOIN extraction_runs r ON r.id = c.extraction_run_id
JOIN exchanges e ON e.id = c.exchange_id
WHERE r.mode = 'live'
  AND c.retracted_at IS NULL
  AND c.novelty IN ('unknown', 'partial', 'contradicts')
  AND NOT EXISTS (
      SELECT 1
      FROM claims newer
      JOIN extraction_runs newer_run ON newer_run.id = newer.extraction_run_id
      WHERE newer.supersedes_claim_id = c.id
        AND newer.retracted_at IS NULL
        AND newer_run.mode = 'live'
  );
