CREATE TABLE author_ratings (
    id INTEGER PRIMARY KEY,
    author_id INTEGER NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 0 AND 3),
    rated_at TEXT NOT NULL
);
CREATE INDEX author_ratings_author ON author_ratings (author_id, id);

CREATE VIEW current_author_ratings AS
SELECT r.author_id, r.rating, r.rated_at
FROM author_ratings r
WHERE r.id = (SELECT MAX(id) FROM author_ratings WHERE author_id = r.author_id);

CREATE TRIGGER author_ratings_no_update BEFORE UPDATE ON author_ratings BEGIN
    SELECT RAISE(ABORT, 'author ratings are append-only');
END;
CREATE TRIGGER author_ratings_no_delete BEFORE DELETE ON author_ratings BEGIN
    SELECT RAISE(ABORT, 'author ratings are append-only');
END;
