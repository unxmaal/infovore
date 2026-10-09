# Log

- cold read round 1 (sohot-code, thinking off): all READY, rejected as uninformative
- cold read round 2 (strict prompt): answers added to every record as Cold read lines
- T-01: red ok, green ok; 1 local call (review only); tests and impl hand-written (one-line change); review: no issues
- T-02: red ok, green ok; 3 local calls (tests, impl, review); tests rewritten (model_id ints), SQL tables and tags_for hand-fixed; review noise, no action; README row added (not in record)
- T-04: red ok, green ok; 3 local calls; tests patched (typo import, tie expectation), display_names rewritten (consumed generator twice, wrong key); review unusable, no action
- T-05: red ok, green ok; 2 local calls; tests patched (schema assertions), impl used with one cosmetic fix (double space in SYSTEM)
- T-06: red ok, green ok; 2 local calls; tests rewritten (token-overlap expectations), impl patched (empty-token claims were dropped instead of grouped); combined review with T-08: no issues
- T-07: red ok, green ok; 1 shared local call with T-08; tests rewritten compact, impl used nearly as-is
- T-08: red ok, green ok; 1 shared local call with T-07; impl patched (support must use only cited statements), tests rewritten (dedup expectation wrong)
- T-03: red ok, green ok; 1 local call (tests and impl together); tests rewritten (fixture misuse, empty written_sections cases), SQL and module used with line wrapping; README row added (not in record)
- T-09: red ok, green ok; 2 local calls (draft, review); draft unusable (placeholder imports, broken SQL, undefined names), hand-written; review: no actionable issues
- cold-read gap: tasks.md files lists omit README.md data model rows (tests/test_readme.py fails without them) for T-02 and T-03
- T-10: red ok, green ok; 0 local calls (hand-written, small); one test expectation fixed (display-name ties are alphabetical)
- T-11: red ok, green ok; 0 local calls (hand-written; the T-09 local draft was unusable); record defect: Files omits infovore/wiki/build.py (shared sections_of helper) and README
- T-12: red ok, green ok; 0 local calls (hand-written, the local slot was shared and prior drafts needed full rewrites); record defects: Files omits README and infovore/wiki/groups.py needs a lazy import in build.py (circular import via WikiClaim)
