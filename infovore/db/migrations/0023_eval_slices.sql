CREATE TABLE eval_slices (
    name TEXT NOT NULL,
    exchange_id INTEGER NOT NULL REFERENCES exchanges (id),
    position INTEGER NOT NULL,
    population TEXT NOT NULL,
    seed INTEGER NOT NULL,
    frozen_at TEXT NOT NULL,
    PRIMARY KEY (name, exchange_id),
    UNIQUE (name, position)
);

-- Frozen: judgments and measurements are keyed to these exact rows (#190).
CREATE TRIGGER eval_slices_no_update BEFORE UPDATE ON eval_slices BEGIN
    SELECT RAISE(ABORT, 'eval slices are frozen');
END;

CREATE TRIGGER eval_slices_no_delete BEFORE DELETE ON eval_slices BEGIN
    SELECT RAISE(ABORT, 'eval slices are frozen');
END;
