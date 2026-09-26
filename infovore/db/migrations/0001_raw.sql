CREATE TABLE channels (
    id INTEGER PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    parent_id INTEGER,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('text', 'thread')),
    archived INTEGER NOT NULL DEFAULT 0,
    last_backfilled_message_id INTEGER
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    channel_id INTEGER NOT NULL,
    guild_id INTEGER NOT NULL,
    author_id INTEGER NOT NULL,
    author_name_at_time TEXT NOT NULL,
    author_is_bot INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    edited_at TEXT,
    content TEXT NOT NULL,
    reply_to_id INTEGER,
    thread_id INTEGER,
    deleted_at TEXT,
    ingested_at TEXT NOT NULL,
    raw_json TEXT NOT NULL
);
CREATE INDEX messages_channel_created ON messages (channel_id, created_at, id);
CREATE INDEX messages_thread ON messages (thread_id);
CREATE INDEX messages_reply ON messages (reply_to_id);
CREATE INDEX messages_author ON messages (author_id);

CREATE TABLE message_revisions (
    message_id INTEGER NOT NULL REFERENCES messages (id),
    revision INTEGER NOT NULL,
    content TEXT NOT NULL,
    edited_at TEXT,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (message_id, revision)
);

CREATE TABLE attachments (
    id INTEGER PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages (id),
    filename TEXT NOT NULL,
    content_type TEXT,
    size INTEGER NOT NULL,
    url TEXT NOT NULL,
    sha256 TEXT,
    local_path TEXT
);
CREATE INDEX attachments_message ON attachments (message_id);

CREATE TABLE reactions (
    message_id INTEGER NOT NULL REFERENCES messages (id),
    emoji TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (message_id, emoji)
);

CREATE TABLE opt_outs (
    user_id INTEGER PRIMARY KEY,
    since TEXT NOT NULL
);
