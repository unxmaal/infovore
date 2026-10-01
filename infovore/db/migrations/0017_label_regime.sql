ALTER TABLE message_labels ADD COLUMN regime TEXT;
ALTER TABLE label_events ADD COLUMN regime TEXT;

UPDATE message_labels SET regime = 'isolated'
WHERE source = 'human' AND source_ref IN (
    'sift-serve:batch-002',
    'sift-serve:batch-003',
    'sift:2026-09-27T22:21:27.583979+00:00:a'
);

UPDATE message_labels SET regime = 'context'
WHERE source = 'human' AND regime IS NULL;

-- Each event carries the regime of ITS OWN round. Copying the regime from the
-- message's current label would retag a repeat's earlier judgment as whatever
-- the latest one was, which is the confusion issue #170 is about.
UPDATE label_events SET regime = 'isolated'
WHERE source_ref IN (
    'sift-serve:batch-002',
    'sift-serve:batch-003',
    'sift:2026-09-27T22:21:27.583979+00:00:a'
);

UPDATE label_events SET regime = 'context' WHERE regime IS NULL;

CREATE INDEX message_labels_regime ON message_labels (regime);
