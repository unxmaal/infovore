UPDATE label_events SET regime = 'isolated'
WHERE source_ref IN (
    'sift-serve:batch-002',
    'sift-serve:batch-003',
    'sift:2026-09-27T22:21:27.583979+00:00:a'
) AND regime != 'isolated';

UPDATE label_events SET regime = 'context'
WHERE source_ref NOT IN (
    'sift-serve:batch-002',
    'sift-serve:batch-003',
    'sift:2026-09-27T22:21:27.583979+00:00:a'
) AND regime != 'context';
