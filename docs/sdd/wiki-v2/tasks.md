# Tasks

Shared context for every task (copied here once; each record names what it uses):

- Repo `/Users/eric/projects/github/unxmaal/infovore`, Python, `uv run`. Checks: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, `uv run pytest -q` (100% coverage required). Line length 100, mypy strict, minimal comments, no em-dashes.
- Tables: `claims_v2(id, run_id, exchange_id, speaker, statement)`; `current_claim_reviews(claim_id, verdict, interface)`; `current_claim_checks(claim_id, verdict)` with verdict in supported|uncheckable|low_overlap|unsupported_fact; `current_exchanges(id, started_at, ...)`.
- Test DB helpers: `tests/claims/seed.py` (`db(tmp_path)`, `environment(tmp_path)`, `conversation(...)`, `NOW`), `tests/wiki/seed.py` (`wiki_db(tmp_path, runs=1)`, `add_run`, `add_exchange(conn, base, day)`, `add_claim(conn, exchange_id, speaker, statement, *, check="supported", review=None, run_id=1)`).
- LLM helpers in `infovore/claims/extract.py`: `Transport = Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], float]]`, `http_post(endpoint) -> Transport`, `http_get(url)`, `fetch_model_id(endpoint, alias, get) -> (model_id, source)`, `ClaimsReplyError`. Stop handling in `infovore/claims/command.py`: `StopFlag`, `_stop_on_signals(flag)`.
- Tokens: `infovore.claims.value.tokens(text) -> frozenset[str]` (lowercase alphanumeric words minus stopwords).
- Migrations: `infovore/db/migrations/NNNN_name.sql`, applied in order on connect; append-only tables get `BEFORE UPDATE`/`BEFORE DELETE` triggers raising `'<table> are append-only'`, as in `0030_speaker_drops.sql`.

### T-01: Publishing ignores the claim check verdict

- **Outcome:** `is_publishable(review, check)` returns True for any check verdict (including None) unless review is wrong, made_up or not_useful.
- **Requirement:** REQ-1
- **Assumptions:** A-08
- **Files:** modify `infovore/wiki/eligibility.py`; modify `tests/wiki/test_build.py` (and any test asserting check-based exclusion; find with `grep -rn "unsupported_fact\|low_overlap\|excluded" tests/wiki`).
- **Context:** current code `return review not in REJECTED_REVIEWS and check in PUBLISHABLE_CHECKS`. Keep the `check` parameter (callers pass it); drop `PUBLISHABLE_CHECKS` if unused.
- **Test first:** unsupported_fact with no review is publishable; low_overlap publishable; None check publishable; made_up review with supported check not publishable.
- **Cold read:** `None` review is publishable. Keep the `check` parameter, unused. Remove `PUBLISHABLE_CHECKS` and any test that asserted check-based exclusion; update counts in tests that relied on it.
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/wiki`
- **Blocked by:** none
- **Parallel group:** A

### T-02: Tag store

- **Outcome:** migration 0031 plus `infovore/db/wiki_tags.py` store and read claim tags per tag run.
- **Requirement:** REQ-2
- **Assumptions:** A-09
- **Files:** create `infovore/db/migrations/0031_claim_tags.sql`, `infovore/db/wiki_tags.py`, `tests/db/test_wiki_tags.py`.
- **Context:** tables `tag_runs(id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, endpoint TEXT NOT NULL, model_alias TEXT NOT NULL, model_id TEXT NOT NULL, prompt_hash TEXT NOT NULL)` and `claim_tags(tag_run_id INTEGER NOT NULL REFERENCES tag_runs(id), claim_id INTEGER NOT NULL REFERENCES claims_v2(id), tags_json TEXT NOT NULL, PRIMARY KEY (tag_run_id, claim_id))`, both append-only (claim_tags: no update/delete). API: `create_tag_run(conn, *, endpoint, model_alias, model_id, prompt_hash, now: datetime) -> int`; `record_tags(conn, tag_run_id, tags: Mapping[int, Sequence[str]]) -> None` (one row per claim, tags_json is a JSON list, empty list allowed); `tagged_claim_ids(conn, model_id, prompt_hash) -> set[int]` (claims tagged by any tag run with that model id and prompt hash); `tags_for(conn, tag_run_ids: Sequence[int]) -> dict[int, list[str]]` (claim id to tags; later runs win).
- **Test first:** record then read back; tagged_claim_ids matches only same model+hash; update raises append-only.
- **Cold read:** Mapping keys are claim ids. Tags stored in the order given. `tags_for`: when several tag runs tag one claim, the highest tag_run_id wins. `create_tag_run` returns `cursor.lastrowid`. Trigger messages: 'tag runs are append-only', 'claim tags are append-only'. `started_at` is `now.isoformat()`.
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/db/test_wiki_tags.py`
- **Blocked by:** none
- **Parallel group:** A

### T-03: Article store

- **Outcome:** migration 0032 plus `infovore/db/wiki_articles.py` store and read written sections per article run.
- **Requirement:** REQ-6
- **Assumptions:** A-09
- **Files:** create `infovore/db/migrations/0032_article_sections.sql`, `infovore/db/wiki_articles.py`, `tests/db/test_wiki_articles.py`.
- **Context:** tables `article_runs(id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, endpoint TEXT NOT NULL, model_alias TEXT NOT NULL, model_id TEXT NOT NULL, prompt_hash TEXT NOT NULL, tag_run_id INTEGER NOT NULL REFERENCES tag_runs(id))` and `article_sections(article_run_id INTEGER NOT NULL REFERENCES article_runs(id), topic TEXT NOT NULL, section TEXT NOT NULL, sentences_json TEXT NOT NULL, dropped INTEGER NOT NULL, PRIMARY KEY (article_run_id, topic, section))`, append-only. `sentences_json` is a JSON list of `[text, [claim_id, ...]]`. API: `create_article_run(conn, *, endpoint, model_alias, model_id, prompt_hash, tag_run_id, now) -> int`; `record_section(conn, article_run_id, topic, section, sentences: Sequence[tuple[str, Sequence[int]]], dropped: int) -> None`; `written_sections(conn, model_id, prompt_hash, tag_run_id) -> set[tuple[str, str]]`; `sections_for(conn, article_run_ids) -> dict[str, dict[str, list[tuple[str, list[int]]]]]` (topic to section to sentences; later runs win).
- **Test first:** record then read back; written_sections filters on model, hash and tag run; delete raises append-only.
- **Cold read:** Outer key of `sections_for` is topic. Highest article_run_id wins per (topic, section). Trigger messages: 'article runs are append-only', 'article sections are append-only'. Claim id lists are never empty (the filter drops such sentences).
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/db/test_wiki_articles.py`
- **Blocked by:** T-02 (foreign key to tag_runs; migration order)
- **Parallel group:** B

### T-04: Canonical topic names

- **Outcome:** `infovore/wiki/canon.py` maps tag strings to canonical keys and picks display names.
- **Requirement:** REQ-3
- **Assumptions:** A-03
- **Files:** create `infovore/wiki/canon.py`, `tests/wiki/test_canon.py`.
- **Context:** `canonical(tag: str) -> str`: casefold, strip, drop one leading vendor word from (sgi, silicon graphics, sun, sun microsystems, ibm, dec, digital, hp, apple, compaq), remove spaces, hyphens, underscores and slashes, remove dots except between two digits; return "" for empty. `display_names(tags: Iterable[str]) -> dict[str, str]`: canonical key to its most frequent original spelling (ties: alphabetical first). `alias_names(topics: Topics) -> dict[str, str]`: canonical key of every alias in `infovore/wiki/topics.toml` to that topic's name (load aliases with `tomllib` from the same file `load_topics` reads; `Topics` is in `infovore/wiki/topics.py`).
- **Test first:** "SGI Indigo 2", "indigo2", "Indigo-2" same key; "IRIX 6.5" key keeps "6.5" dot and differs from "IRIX 65"; display picks most frequent; alias "personal iris" maps to "Indigo".
- **Cold read:** Casefold and strip first, then drop at most one vendor prefix (multi-word prefixes checked first) only when followed by more text. Dots kept only when both neighbours are digits ("6.5.1" keeps both dots; "v1.2" keeps its dot; "Mr." loses it). Ties in `display_names` broken by the original spelling, plain string order. Empty input returns {}. topics.toml entries are `[[topic]]` tables with `name` and optional `aliases` list.
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/wiki/test_canon.py`
- **Blocked by:** none
- **Parallel group:** A

### T-05: Tagging request and reply

- **Outcome:** `infovore/wiki/tagger.py` builds a strict-schema tagging request for a batch of claims and parses the reply into tags per position.
- **Requirement:** REQ-2
- **Assumptions:** A-01
- **Files:** create `infovore/wiki/tagger.py`, `tests/wiki/test_tagger.py`.
- **Context:** SYSTEM prompt: "Each numbered line is a claim about vintage computing. For each, list the specific things it is about: machine models, parts, OS and versions, programs, companies, standards. Use the usual full name ('Indigo2', 'IRIX 6.5', 'R10000', 'Sun Ultra 10'). 0 to 4 per claim; skip generic words like 'computer'. Reply {\"t\":[[n,[name,...]],...]} with n the line number." SCHEMA: object with required "t": array of arrays with prefixItems [integer, array of strings maxLength 60 maxItems 4], minItems 2 maxItems 2; additionalProperties false. `prompt_hash()` = first 12 hex of sha256 of json.dumps([SYSTEM, SCHEMA], sort_keys=True). `BATCH = 20`. `build_request(model, statements: Sequence[str], max_tokens=1200) -> dict` (temperature 0, messages system + user "1. ...\n2. ...", response_format json_schema strict, name "tags"). `parse_reply(reply: Mapping, count: int) -> dict[int, list[str]]`: reads choices[0].message.content as JSON; keeps entries whose n is 1..count (0-based index n-1), strips blanks; positions missing from the reply map to []; raises `ClaimsReplyError` (from `infovore.claims.extract`) on invalid JSON or finish_reason "length".
- **Test first:** request has schema and numbered lines; reply parsed with out-of-range n ignored and missing positions empty; truncated reply raises.
- **Cold read:** `reply` is the parsed JSON dict from the gateway. Names are stripped and empty ones dropped. Only finish_reason "length" raises; other values parse normally. `response_format` is `{"type": "json_schema", "json_schema": {"name": "tags", "schema": SCHEMA, "strict": True}}`. `prompt_hash` is `hashlib.sha256(json.dumps([SYSTEM, SCHEMA], sort_keys=True).encode()).hexdigest()[:12]`. User text is `"\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))`. BATCH is used by T-09, not here.
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/wiki/test_tagger.py`
- **Blocked by:** none
- **Parallel group:** A

### T-06: Near-duplicate claim groups

- **Outcome:** `infovore/wiki/groups.py` groups a topic's claims into near-duplicate groups.
- **Requirement:** REQ-5
- **Files:** create `infovore/wiki/groups.py`, `tests/wiki/test_groups.py`.
- **Context:** input is a sequence of `WikiClaim` (`infovore/wiki/build.py`: fields claim_id, exchange_id, date, speaker, statement, topics) already in date order. `group_claims(claims, threshold=0.5) -> list[ClaimGroup]`, `ClaimGroup(frozen dataclass): lead: WikiClaim, members: tuple[WikiClaim, ...]` (lead included in members). Greedy: walk claims in order; join the first existing group whose lead's `tokens(statement)` has Jaccard >= threshold with this claim's tokens, else start a group. Result sorted by member count descending, then lead date, then lead claim_id. Empty token sets never join (each is its own group).
- **Test first:** two paraphrases join, a distinct claim stays apart, order is size-first.
- **Cold read:** "First" means groups in creation order. Members keep walk order. Empty input returns []. `WikiClaim.date` is a 'YYYY-MM-DD' string.
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/wiki/test_groups.py`
- **Blocked by:** none
- **Parallel group:** A

### T-07: Writing request and reply

- **Outcome:** `infovore/wiki/writer.py` builds a strict-schema section-writing request from claim groups and parses cited sentences.
- **Requirement:** REQ-6
- **Assumptions:** A-02, A-06
- **Files:** create `infovore/wiki/writer.py`, `tests/wiki/test_writer.py`.
- **Context:** SYSTEM: "Write a short wiki section about the topic using only the numbered claims. 2 to 6 sentences, plain and factual, merging claims that say the same thing. Give each sentence the numbers of the claims it rests on. Add nothing the claims do not say." SCHEMA: required "s": array (maxItems 8) of arrays prefixItems [string maxLength 400, array of integers minItems 1], minItems 2 maxItems 2. `MAX_GROUPS = 40`. `prompt_hash()` as in T-05 over [SYSTEM, SCHEMA]. `build_request(model, topic: str, section: str, statements: Sequence[str], max_tokens=900) -> dict`: user text "Topic: {topic}" plus "Section: {section}" when section != "General", then numbered statements. `parse_reply(reply, count) -> list[tuple[str, list[int]]]`: returns sentences with cited 0-based positions (n-1), keeping out-of-range numbers as -1 so the filter can drop them; raises `ClaimsReplyError` on bad JSON or finish_reason "length".
- **Test first:** request includes section line only when not General; reply parsed with positions; out-of-range kept as -1; truncation raises.
- **Cold read:** User text: `f"Topic: {topic}"`, then `f"Section: {section}"` unless General, then a blank line, then numbered statements as in T-05, in input order. `prompt_hash` and response_format as in T-05 with name "section". Positions below 1 or above count become -1. Only finish_reason "length" raises. MAX_GROUPS is used by T-11.
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/wiki/test_writer.py`
- **Blocked by:** none
- **Parallel group:** A

### T-08: Sentence support filter

- **Outcome:** `infovore/wiki/support.py` drops written sentences not traceable to the claims they cite.
- **Requirement:** REQ-7
- **Assumptions:** A-07
- **Files:** create `infovore/wiki/support.py`, `tests/wiki/test_support.py`.
- **Context:** `SUPPORT = 0.6`. `support(sentence: str, cited: Sequence[str]) -> float` = fraction of `tokens(sentence)` found in the union of `tokens(c)` for cited statements; 0.0 when sentence has no tokens. `keep_supported(sentences: Sequence[tuple[str, Sequence[int]]], statements: Sequence[str], threshold=SUPPORT) -> tuple[list[tuple[str, list[int]]], int]`: drops a sentence if it cites nothing, cites any position outside 0..len(statements)-1, or support < threshold; returns kept sentences (positions deduplicated, order kept) and the dropped count.
- **Test first:** paraphrase kept; sentence with an invented spec dropped; bad citation dropped; empty citation dropped.
- **Cold read:** Citations deduplicated within a sentence, first occurrence order. Support equal to the threshold is kept. A sentence whose citations are all out of range is dropped as a bad citation.
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/wiki/test_support.py`
- **Blocked by:** none
- **Parallel group:** A

### T-09: `wiki tag` command

- **Outcome:** `infovore wiki tag` tags the claims of chosen runs in batches through the gateway and records them in a new tag run.
- **Requirement:** REQ-2
- **Assumptions:** A-01, A-04
- **Files:** create `infovore/wiki/tag_command.py`, `tests/wiki/test_tag_command.py`; modify `infovore/wiki/command.py` (add the `tag` subparser and dispatch).
- **Context:** args `--runs` (required, comma ids, unknown -> ConfigError "unknown run N"), `--endpoint`, `--model`, `--concurrency` (default 8, 1..32), `--limit N` (optional, number of claims, default all), `--write` (without it: dry run printing the first request's user text and exit 0), `--resume`, `--progress-every N` (default 500). Claims: `SELECT id, statement FROM claims_v2 WHERE run_id IN (...) ORDER BY id`, minus `tagged_claim_ids(model_id, prompt_hash)` when --resume. Model id from `fetch_model_id`. Batches of `tagger.BATCH`; a thread pool of `concurrency` sends `tagger.build_request`; results consumed on the main thread and stored with `record_tags` + commit per batch; a failed batch (ClaimsReplyError) is counted and skipped. Stop on SIGINT via `StopFlag`/`_stop_on_signals` from `infovore/claims/command.py`. Final line: `claims=N batches=N failed=N tags=N seconds=S` then `tag run T written`. Tests inject a fake transport: make the transport a module-level factory `post_for(endpoint)` defaulting to `http_post` that tests monkeypatch, likewise `get = http_get`.
- **Test first:** writes tags for all claims of the chosen run only; --resume skips already tagged; failed batch counted and others written.
- **Cold read:** --limit counts claims (after resume filtering). Batches are consecutive chunks of BATCH claims in id order. Dry run prints the first batch's user message. Commit after each batch is recorded. Seconds printed with 2 decimals. --concurrency outside 1..32 raises ConfigError('--concurrency must be between 1 and 32').
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/wiki/test_tag_command.py`
- **Blocked by:** T-02, T-05
- **Parallel group:** C

### T-10: Topics from a tag run in `wiki build`

- **Outcome:** `wiki build` and `wiki stats` with `--tag-run T` assign topics from stored tags, canonicalised, instead of topics.toml.
- **Requirement:** REQ-3, REQ-4
- **Assumptions:** A-03, A-05
- **Files:** modify `infovore/wiki/build.py`, `infovore/wiki/command.py`; create `tests/wiki/test_tag_topics.py`.
- **Context:** `load_claims(conn, tech_only, salt, runs, tag_runs: Sequence[int] | None = None)`. When tag_runs is given: read `tags_for(conn, tag_runs)`; build `display_names` over every tag of the loaded claims, overlaid by `alias_names(load_topics())` (a topics.toml name wins as display); each claim's topics = display names of the canonical keys of its tags (empty keys skipped). `--claim-gate` still uses `claim_has_tech(lexicon, topics, statement)` with the topics.toml `Topics`. Without --tag-run behaviour is unchanged.
- **Test first:** "SGI Indigo 2" and "Indigo2" tags land on one page named "Indigo2"; claim with no tags is unassigned; without --tag-run topics.toml is used.
- **Cold read:** A canonical key with no topics.toml alias displays as its most frequent spelling across the loaded claims' tags. A claim whose tags all canonicalise to empty is unassigned. Several tag runs: `tags_for` rule (highest run wins per claim).
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/wiki/test_tag_topics.py`
- **Blocked by:** T-02, T-04
- **Parallel group:** D

### T-11: `wiki write` command

- **Outcome:** `infovore wiki write` writes and stores one filtered section per (topic, section) for topics with enough claims.
- **Requirement:** REQ-6, REQ-7
- **Assumptions:** A-02, A-05, A-06, A-07
- **Files:** create `infovore/wiki/write_command.py`, `tests/wiki/test_write_command.py`; modify `infovore/wiki/command.py`.
- **Context:** args `--tag-run T` (required), `--runs` (default: all), `--claim-gate`, `--min-claims` (default 10), `--endpoint`, `--model`, `--concurrency`, `--limit N` (topics), `--write`, `--resume`, `--progress-every`. Claims via `load_claims(..., runs, tag_runs=[T])` (T-10). Sections per topic exactly as `render_page` groups today: key = strongest co-topic among the claim's other topics or "General"; within a section, `group_claims` (T-06), take the first `writer.MAX_GROUPS` groups and send their lead statements. Positions map back to the group's lead claim id. `keep_supported` (T-08) filters; store with `record_section(..., sentences as (text, claim ids), dropped)`. --resume skips `written_sections(model_id, prompt_hash, T)`. Topics in descending claim count. Final line `sections=N failed=N sentences=N dropped=N seconds=S` then `article run A written`. Same fake-transport hook pattern as T-09.
- **Test first:** writes a General and a "With X" section for a topic; unsupported sentence dropped and counted; --resume skips written sections.
- **Cold read:** Section key rule (same as `render_page`): count co-topics over the topic's claims; a claim's section is its other topic with the highest count, ties by name, or General if none. --min-claims counts claims after the claim gate. Topics ordered by claim count descending, ties by name. --resume skips per (topic, section). Positions map to the lead claim id of the group sent at that position. Seconds printed with 2 decimals.
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/wiki/test_write_command.py`
- **Blocked by:** T-03, T-06, T-07, T-08, T-10
- **Parallel group:** E

### T-12: Render articles in `wiki build`

- **Outcome:** `wiki build --tag-run T --article-run A` writes pages as prose with numbered citations and a source list, falling back to grouped claims.
- **Requirement:** REQ-8, REQ-5
- **Assumptions:** A-08
- **Files:** modify `infovore/wiki/build.py`, `infovore/wiki/command.py`; create `tests/wiki/test_render_articles.py`.
- **Context:** `WikiClaim` gains `check: str | None` (from the existing `k.verdict AS chk` column). Page: `# {name}`, then per section in render_page order a `## {section}` heading (omitted when the only section is General), paragraph of its sentences each followed by citations `[n]`; numbers assigned in first-cited order across the page. Then `## Sources`: `n. {statement} ({speaker}, {date}, exchange {id}, check: {verdict or "unchecked"})`. Sections without stored text render the grouped claim list (T-06 groups: lead statement plus "(+k similar)" when members > 1, with the lead's source). `## See also` as today. `sections_for(conn, [A])` supplies text.
- **Test first:** article page has prose with [1] and a matching source line with check label; topic without article falls back to grouped list with "(+1 similar)".
- **Cold read:** Citation numbers are global per page, in first-cited order; a claim cited again reuses its number. Sections ordered as `render_page`: General last, then by size descending, then name. The `## {section}` heading is omitted only when the page has exactly one section and it is General. Fallback is per section: a section with no stored sentences renders its grouped list under the same heading, each line `- {lead statement} (+k similar)` (k = members - 1, omitted when 0) followed by the lead's source in today's `_entry` form. `## Sources` lists only claims cited in prose, `n. {statement} ({speaker}, {date}, exchange {exchange_id}, check: {verdict})`, verdict raw, "unchecked" when NULL; omitted if nothing is cited. `sections_for(conn, [A])` takes article run ids.
- **Verify:** `timeout 300 uv run pytest -q --no-cov tests/wiki/test_render_articles.py`
- **Blocked by:** T-03, T-06, T-10
- **Parallel group:** F

## Execution order

A: T-01, T-02, T-04, T-05, T-06, T-07, T-08 (disjoint files)
B: T-03
C: T-09
D: T-10
E: T-11
F: T-12

GPU: after C, run `wiki tag --runs 13,14` (A-01: ~5.4k requests). After E, run `wiki write`.

## Coverage

| Requirement | Tasks |
|---|---|
| REQ-1 | T-01 |
| REQ-2 | T-02, T-05, T-09 |
| REQ-3 | T-04, T-10 |
| REQ-4 | T-10 |
| REQ-5 | T-06, T-12 |
| REQ-6 | T-03, T-07, T-11 |
| REQ-7 | T-08, T-11 |
| REQ-8 | T-12 |
