CREATE VIEW current_exchanges AS
SELECT * FROM exchanges WHERE superseded_by_recipe IS NULL;

CREATE VIEW current_slice_members AS
SELECT s.* FROM current_eval_slices s JOIN current_exchanges e ON e.id = s.exchange_id;
