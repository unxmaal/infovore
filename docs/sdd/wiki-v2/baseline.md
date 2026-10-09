# Baseline (main 58b2ea6, 2026-10-09)

- `uv run ruff check .` clean; `uv run ruff format --check .` clean.
- `uv run mypy` clean (315 files).
- `uv run pytest -q` 2499 passed, 100% coverage, about 140 s.
- `scripts/red_green.sh` can hang on a leftover test server (#290): always run under `timeout 900`.
- Live wiki from runs 13,14 with --claim-gate: 43 topics (closed topics.toml), 62,722 claims kept, 41,213 unassigned, 44,309 excluded (mostly by claim check), pages are flat claim lists (o2.md: 2,881 lines).
- Claim check vs Eric's conversation-interface reviews on runs 11 and 13: of 45 checked good claims, 17 are low_overlap or unsupported_fact.
