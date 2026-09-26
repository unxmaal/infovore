CREATE TABLE exchanges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL,
    thread_id INTEGER,
    first_message_id INTEGER NOT NULL,
    last_message_id INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    grouping_rule TEXT NOT NULL CHECK (grouping_rule IN ('thread', 'reply_chain', 'quiet_gap')),
    content_hash TEXT NOT NULL UNIQUE,
    parent_exchange_id INTEGER REFERENCES exchanges (id),
    extraction_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (extraction_status IN ('pending', 'done', 'skipped', 'failed', 'stale')),
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE INDEX exchanges_status ON exchanges (extraction_status);

CREATE TABLE exchange_messages (
    exchange_id INTEGER NOT NULL REFERENCES exchanges (id),
    message_id INTEGER NOT NULL UNIQUE REFERENCES messages (id),
    position INTEGER NOT NULL,
    PRIMARY KEY (exchange_id, message_id),
    UNIQUE (exchange_id, position)
);
