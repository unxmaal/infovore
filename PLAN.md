# Discord Lore Infovore — Implementation Plan

Working name: `infovore`. A passive Discord bot that ingests channel history and live messages, groups them into exchanges, extracts **net-new** factual claims with provenance (net-new = knowledge a frontier LLM does not already have from pretraining: SGI hardware, IRIX internals, period tooling, community procedures), and writes them to a local SQLite database. **The database is the product**: nothing talks to infovore at runtime, and consumers read the file. No MCP server, no API, no user-facing Discord commands. No slash commands. It never posts.

## Operating rules for this build

These apply to every task below and to every subagent.

1. **Red/green TDD, strictly.** For every unit of behavior: write a failing test, run it and confirm it fails for the right reason, write the minimum code to pass, refactor. Commit after each green. A subagent that writes implementation before a failing test has violated the plan.
2. **100% line and branch coverage**, enforced in CI and locally (`pytest --cov --cov-branch --cov-fail-under=100`). No `# pragma: no cover` without a one-line justification in README.md under "Coverage exclusions".
3. **No documentation in code.** No docstrings, no explanatory comment blocks, no module headers. All documentation lives in `README.md`. Names must carry meaning instead. A comment is permitted only to reference an external constraint (e.g. a Discord API quirk) and must be a single line pointing to the README section that explains it.
4. **No network in tests.** Discord is reached through a narrow protocol interface with in-memory fakes. Every LLM call goes through the `LLMBackend` protocol with a deterministic fake. No test opens a socket or spawns a process.
5. **Deterministic, idempotent ingest.** Re-running any stage over data it has already seen produces no duplicates and no changes.
6. **Orchestration.** The lead (Opus) owns the schema, interfaces, and phase ordering. Sonnet subagents take one task each from the task lists below, work in a feature branch, and return with tests passing at 100% coverage. The lead reviews the diff against the acceptance criteria before merging. Subagents do not touch interfaces defined by the lead without returning to ask.

## Stack

- Python 3.12, `uv` for env and lockfile
- `discord.py` (Message Content and Server Members privileged intents enabled on the application; members are needed for opt-out role sync)
- SQLite via `sqlite3` stdlib, schema managed by plain versioned SQL migration files
- `pytest`, `pytest-cov`, `pytest-asyncio`, `hypothesis` for the chunker
- `ruff` for lint and format, `mypy --strict`
- **Pluggable LLM backends** behind one `LLMBackend` protocol, selected per stage (extract, probe, judge) in config. Each backend reports its capabilities (native JSON-schema output, safe concurrency, usage-limit semantics), and callers never branch on backend type:
  - `claude_cli`: headless `claude -p` subprocess. Authenticated by the local Claude Max login, or on another host by a long-lived token from `claude setup-token`. Claude Code can also route to Bedrock/Vertex with that cloud's credentials. Never `--bare` (it ignores OAuth).
  - `openai_compat`: the `openai` SDK against any OpenAI-compatible Chat Completions endpoint (vLLM, llama.cpp, Ollama, LM Studio/MLX on a Mac, hosted providers). No provider features assumed.
  - `fake`: deterministic, shipped, and used by every test above the backend layer.
  - A native Anthropic-SDK backend (API key or Bedrock; prompt caching, batches) is deferred and slots in without touching callers.
- **Deployment-neutral**: one Python package, all configuration from env (optionally loaded from a `.env` file), logs as JSON lines on stdout, no host paths assumed, exit codes meaningful. Runs the same on a laptop, a Mac Studio, an unknown Linux box, or AWS. See Phase 6 "Packaging and deployment".
- SQLite FTS5 over claims, used internally for related-claim lookup

## Repository layout

```
infovore/
  config.py          settings from env; validated at startup
  rows.py            frozen row dataclasses and enums matching the data model
  timing.py          Clock and Sleeper protocols with system and fake implementations
  db/
    connection.py    open, migrate, WAL, foreign keys on
    migrations/      0001_raw.sql, 0002_exchanges.sql, 0003_claims.sql, ...
    raw.py           repository for messages/attachments/reactions
    exchanges.py     repository for exchanges
    claims.py        repository for extracted claims and extraction runs
  source/            (not named `discord` to avoid shadowing discord.py)
    protocol.py      DiscordSource protocol: list_channels, history, events, role_member_ids
    live.py          discord.py client implementing DiscordSource
    fake.py          in-memory DiscordSource for tests (shipped, not test-only)
  ingest/
    backfill.py      walk channel history into raw tables
    live.py          on_message / reaction handlers into raw tables
    normalize.py     discord objects -> row dataclasses
  chunk/
    grouper.py       raw messages -> exchanges
    rules.py         thread, reply-chain, quiet-gap rules
  extract/
    protocol.py      ClaimExtractor protocol
    llm_extractor.py ClaimExtractor over any LLMBackend: prompt in, JSON out, schema validation, one repair retry
    fake.py          deterministic extractor for tests
    prompt.py        prompt assembly from an exchange
    schema.py        pydantic models for extractor output; strict parsing
    runner.py        pending exchanges -> extractor -> validated claims
  llm/
    protocol.py      LLMBackend, LLMRequest, LLMResult (structured output or text, canonical model id, usage, error kind: transient | usage_limit | fatal, retry_after), capabilities
    claude_cli.py    headless `claude -p` backend; the only module that spawns a process
    openai_compat.py OpenAI-compatible Chat Completions backend; the only module importing openai
    fake.py          deterministic backend (shipped, not test-only)
    process.py       ProcessRunner protocol used by claude_cli
    registry.py      config -> backend instance per stage
  privacy/
    optout.py        opt-out role and per-user redaction
  cli.py             entrypoint; subcommands backfill / chunk / extract / probe / review / promote / status / snapshot / run, each added by the task that builds its feature
tests/               mirrors package layout, one test module per source module
README.md
pyproject.toml
```

## Data model

All timestamps UTC ISO-8601. All Discord IDs stored as `INTEGER` (snowflakes fit in 64-bit).

**schema_migrations** — version PK, applied_at.

**channels** — id PK, guild_id, parent_id NULL (set for threads), name, kind (`text` | `thread`), archived, last_backfilled_message_id NULL. Backfill checkpoint per channel and thread; also the channel name the prompt renders.

**messages** — id PK, channel_id, guild_id, author_id, author_name_at_time, created_at, edited_at NULL, content, reply_to_id NULL, thread_id NULL, deleted_at NULL, ingested_at, raw_json.

**message_revisions** — message_id FK, revision, content, edited_at, raw_json, PK(message_id, revision). Prior versions saved on every edit; redacted in place on opt-out.

**attachments** — id PK, message_id FK, filename, content_type, size, url, sha256 NULL (filled when downloaded), local_path NULL.

**reactions** — message_id FK, emoji, count, PK(message_id, emoji). Updated in place; history not kept.

**exchanges** — id PK autoincrement, channel_id, thread_id NULL, first_message_id, last_message_id, started_at, ended_at, message_count, grouping_rule (`thread` | `reply_chain` | `quiet_gap`), content_hash (sha256 of ordered message ids), parent_exchange_id NULL (the closed exchange a late reply continues; rendered as read-only context), extraction_status (`pending` | `done` | `skipped` | `failed` | `stale`), retry_count, last_error NULL, UNIQUE(content_hash). `stale` = a member message was edited after extraction; the runner re-extracts it.

**exchange_messages** — exchange_id FK, message_id FK UNIQUE (a message belongs to at most one exchange), position, PK(exchange_id, message_id), UNIQUE(exchange_id, position).

**prompt_versions** — version PK, text_sha256, created_at, promoted_at NULL. The live version is the most recently promoted one.

**extraction_runs** — id PK, exchange_id FK, model, prompt_version, started_at, finished_at, input_tokens, output_tokens, mode (`trial` | `live`), outcome, error NULL. Only claims from `live` runs appear in the `lore` view; `trial` runs exist for prompt iteration.

**claims** — id PK, exchange_id FK, extraction_run_id FK, statement, subject, kind (`fact` | `correction` | `procedure` | `reference`), confidence (0–1), probe_question, permalink, supersedes_claim_id NULL, novelty (`unprobed` | `unknown` | `partial` | `contradicts` | `known`), probe_model NULL, probe_answer NULL, probed_at NULL, probe_error NULL, retracted_at NULL, retraction_reason NULL.

**claim_sources** — claim_id FK, message_id FK, PK(claim_id, message_id). Replaces a JSON id array so deletion and opt-out retraction are plain queries. A claim is current when it is not retracted and no current claim supersedes it.

**claims_fts** — FTS5 over claims(statement, subject), `unicode61` with domain `tokenchars` (`-./_`), subject weighted above statement in bm25; kept in sync by triggers.

**lore** (view) — the consumer contract: current claims only (not retracted, not superseded by a live non-retracted correction, from `live` runs, novelty `unknown` | `partial` | `contradicts` — i.e. probed and not `known`; `unprobed` claims are excluded until probed), with subject, statement, kind, confidence, novelty, permalink, and source message ids. Documented in README "Consuming the database"; schema version in `PRAGMA user_version`.

**opt_outs** — user_id PK, since.

## Lifecycle rules

Decided in #3. Each rule is documented in README "Grouping rules" or "Privacy and opt-out" and tested one-for-one.

1. **Late reply** to a message in a closed exchange starts a new exchange with `parent_exchange_id` set; the parent's messages are rendered as read-only context and cannot be cited.
2. **Open exchanges**: no rule closes an exchange until its newest message is older than the quiet gap (thread and reply_chain included).
3. **Edit after extraction**: prior content goes to `message_revisions`; the exchange becomes `stale` and is re-extracted. New claims supersede the old ones from that exchange.
4. **Delete**: a claim whose every source message is deleted is retracted (`retraction_reason = 'sources_deleted'`).
5. **Opt-out**: claims whose sources are all from opted-out users are retracted; claims with other sources are kept and the opted-out messages are redacted.
6. **Size cap**: an exchange over 50 messages or the configured token budget is split at its largest internal time gaps, with a 3-message overlap rendered as context.
7. **Deleted, system, and bot messages** are dropped at grouping time.

## Phases

**First milestone is the historic backfill**: backfill, chunk, extract in trial mode, review, iterate the prompt. **The recommended path for the historic backfill is `source/export.py` (Phase 2 task 7) over a DiscordChatExporter JSON export**: fetch from the Discord API exactly once, with DCE's own bot, outside of infovore; every `infovore` command afterwards — `sync-optouts`, `backfill`, `chunk`, `extract`, `probe`, `review`, `promote`, `snapshot` — reads only that export's local files and the SQLite database, with zero further Discord API traffic, and is freely re-runnable. Live event ingest (Phase 2 task 4), the `discord.py` adapter (Phase 2 task 5), and the `run` loop are optional and needed only for live/steady-state ingest, which may trail the historic backfill and prompt settling entirely. Tracked as GitHub milestones **M1: backfill + prompt iteration** and **M2: live + product**. **Gate: no extraction against real Discord data until opt-out redaction (Phase 2 task 6) is merged.**

Phases are sequential. Tasks within a phase may run in parallel across subagents once the lead has merged the phase's interface task.

### Phase 0 — Skeleton (lead)

- Initialize repo, `pyproject.toml`, `uv.lock`, `ruff`, `mypy --strict`, `pytest` config with 100% threshold, pre-commit, CI workflow.
- Write `README.md` skeleton with the section headings used throughout: Purpose, Architecture, Data model, Configuration, Running, Grouping rules, Extraction prompt, Privacy and opt-out, Coverage exclusions, Consuming the database, Deployment, Development workflow.
- Define all `protocol.py` files and the row dataclasses. These are the contracts subagents build against.

Acceptance: `uv run pytest` passes with a single trivial test; coverage gate is active and passes at 100% on the empty package.

### Phase 1 — Storage

Tasks (one subagent each):

1. `db/connection.py` — open with WAL and foreign keys, apply migrations in order, record applied versions. Tests: fresh DB migrates to latest; re-migrate is a no-op; a migration failure rolls back and leaves version untouched.
2. `db/raw.py` — upsert message, upsert attachment, set reaction count, mark deleted, mark edited, upsert channel/thread, get/set backfill checkpoint. Tests: upsert is idempotent; edit updates content and edited_at and writes the prior version to `message_revisions`; delete sets deleted_at and keeps row.
3. `db/exchanges.py` — insert exchange with members, get pending and stale (respecting the retry limit), set status, increment retry, mark stale by member message. Tests: duplicate content_hash is rejected without partial writes; membership positions are contiguous from 0.
4. `db/claims.py` — insert run, insert claims with `claim_sources`, record/promote prompt versions, record probe error, set novelty verdict, retract claim, retract-by-sources (deleted or opted-out), related-claims lookup via `claims_fts`. Tests: claims reference an existing run; supersedes links resolve; FTS stays in sync on insert and retract; retracted claims are excluded from related-claims results.
5. `config.py` — env-driven (optional `.env` file); required: bot token, guild id, channel allowlist, DB path; per stage (extract, probe, judge): backend name plus that backend's settings (`claude_cli`: binary path, model alias, timeout; `openai_compat`: base URL, key, model, `json_schema_supported`), each with a concurrency limit; optional: quiet gap, batch size, max retries, exchange size cap, opt-out role name, bot-messages toggle. Startup fails loudly on any missing required value and runs a health check against each configured backend (one trivial request). Lands in Phase 1 with the core settings and a registry that knows only the `fake` backend; each backend task registers itself and its settings, and later phases add their own settings.
6. `cli.py` skeleton — console entrypoint, config and DB bootstrap, a subcommand registration pattern, and `status` (row counts, pending queues, last run per stage, each stage's backend and model). Every later task that builds a user-facing feature adds its own subcommand: `backfill` (Phase 2), `chunk` (Phase 3), `extract` and `probe` (Phase 4), `review` and `promote` (Phase 4), `snapshot` and `run` (Phase 6). Tests invoke the entrypoint in-process with fakes.

Acceptance: every repository method has a test for the happy path and every failure branch; `uv run infovore status` on a fresh DB reports zeros without error; 100% coverage.

### Phase 2 — Discord ingest

Tasks:

1. `source/fake.py` — in-memory `DiscordSource` seeded from a list of dataclass messages; supports history paging with a configurable page size and an async event stream. This is shipped code and must itself be fully tested.
2. `ingest/normalize.py` — pure functions from a minimal discord-like object to row dataclasses. Tests: replies, threads, attachments, edited, system messages (skipped), bots (skipped, configurable).
3. `ingest/backfill.py` + `infovore backfill` — for each allowlisted channel and its threads (including archived), page history oldest-first, normalize, upsert, checkpoint the last message id per channel so an interrupted run resumes. Tests against the fake: full walk, resume from checkpoint, rate-limit backoff path via an injected sleeper.
4. `ingest/live.py` — handlers for message create, edit, delete, reaction add/remove, thread create; an edit to a message in an extracted exchange marks it `stale`; a delete triggers retract-by-sources. Tests against the fake event stream.
5. `source/live.py` — the only module that imports `discord.py`. Thin. Tests cover the mapping into the protocol using fake discord objects; no client connection is ever made in tests.
6. `privacy/optout.py` — sync the opt-out role from Discord into `opt_outs`; redaction replaces content and author of opted-out users with a placeholder before chunking and extraction; already-stored raw rows for a newly opted-out user are redacted in place and the change is logged. Tests for each path. Retracts claims whose sources are all opted-out users. Placed in Phase 2 because it gates any extraction against real data.
7. `source/export.py` (issue #74) — a third `DiscordSource` reading a directory tree of DiscordChatExporter JSON export files (one file per channel/thread, possibly partitioned): `list_channels` covers text channels and threads with the thread's parent taken from the export; `history(channel_id, after_id, page_size)` pages oldest-first honoring `after_id`, so the same backfill checkpoints and resume behavior work unchanged; `events()` is an empty stream (an export is not live); `role_member_ids(guild_id, role_name)` derives from authors' roles recorded in the export (limitation: only users who posted appear); DCE message types `Default`/`Reply` map to normal messages, everything else to `is_system`; `reference.messageId` maps to `reply_to_id`; malformed or unrecognized files raise a clear `SourceUnavailableError` naming the file, never a raw traceback. `guild_ids()` lets config infer the guild when the export holds exactly one. This is the recommended path for the historic backfill: no Discord token needed at infovore's end, and the export can be fetched once and re-ingested any number of times. Tests: fixture export trees (channel, thread, partitioned channel, system message, reply, attachments/reactions, bot author, nickname vs. name), after-id paging, role inference, a malformed file, guild inference, and a full `backfill()` from the export into a temp DB followed by a re-run that changes nothing.

Acceptance: a full backfill of the fake produces exactly the expected rows; running it twice changes nothing; interrupting mid-page and resuming produces the same result as an uninterrupted run. The same acceptance holds for `source/export.py` against a fixture export tree in place of the fake.

### Phase 3 — Chunking

Tasks:

1. `chunk/rules.py` — three pure grouping rules, each `list[MessageRow] -> list[list[MessageRow]]`:
   - `thread`: all messages sharing a thread_id.
   - `reply_chain`: transitive closure over reply_to_id, including the root.
   - `quiet_gap`: within a channel, split when the gap between consecutive messages exceeds a configurable duration (default 30 minutes).
   Precedence: thread > reply_chain > quiet_gap. A message belongs to exactly one exchange. Size-cap splitting and dropped message types per Lifecycle rules 6–7.
2. `chunk/grouper.py` — apply rules to un-grouped messages, compute content_hash, persist. Tests with `hypothesis`: every message lands in exactly one exchange; ordering within an exchange is by created_at then id; re-grouping is a no-op; an exchange whose last message is younger than the quiet gap is not closed yet (still open, not persisted), for every rule; a late reply creates a new exchange linked by `parent_exchange_id`.

Acceptance: property tests pass; documented rule behavior in README matches the tests one-for-one.

### Phase 4 — Extraction

Tasks:

1. `extract/schema.py` — pydantic models for extractor output; unknown fields rejected; confidence bounded; source_message_ids must be a non-empty subset of the exchange's own ids (not context ids); `supersedes` must be one of the related-claim ids supplied in the prompt; `probe_question` required and must not contain the claim's statement verbatim. Tests for every validation branch.
2. `extract/prompt.py` — assemble the prompt from an exchange: channel name, ordered messages with author and timestamp, reaction counts, attachment filenames, the exchange permalink, parent/overlap context messages (marked uncitable), and up to 10 related existing claims from `claims_fts` with their ids. Prompt text is a versioned constant; `prompt_version` is written to every run. Tests: rendering is stable (snapshot), opt-out users are redacted before rendering, token estimate is computed.
3. `extract/fake.py` — deterministic extractor that returns claims keyed off message content markers (e.g. a message containing `FACT:` yields a fact claim). Fully tested.
4. LLM backends (`llm/`) and `extract/llm_extractor.py`. The protocol and fake come first (lead-owned); each backend is its own task and can land independently.
   - `llm_extractor.py` implements `ClaimExtractor` (and the probe's recall/judge calls) against `LLMBackend` only. It passes the JSON schema generated from `schema.py`; if the backend lacks native schema support, it extracts the first JSON object from the text. It always validates with `schema.py` and makes one repair call on a validation failure. Error kinds map to outcomes: `transient`/`fatal` record a failed run, leave the exchange `pending`, and increment `retry_count`; `usage_limit` pauses the whole runner until `retry_after` and is not counted against the exchange. The canonical model id from the result is what gets recorded.
   - `llm/claude_cli.py`: argv `claude -p --model <alias> --system-prompt <ours> --tools "" --strict-mcp-config --setting-sources "" --no-session-persistence --output-format json --json-schema <schema>`, prompt on stdin, empty scratch cwd, never `--bare`. Reads `structured_output`, `is_error`, `api_error_status`, `usage`, `total_cost_usd` (API-equivalent, informational), and `modelUsage` (canonical id). Spawns through an injected `ProcessRunner`; tests never spawn `claude`.
   - `llm/openai_compat.py`: `openai` SDK with configurable base URL/key/model; sends `response_format` json_schema only when the stage config says the endpoint supports it; maps HTTP 429/5xx/timeouts to `transient` and `finish_reason` of `length`/`content_filter` to `fatal`; records usage when reported. Tests inject `httpx.MockTransport`; no real calls.
   - A shared **backend contract test suite** (parametrized over every backend with its fake transport) asserts that the same request shapes yield the same `LLMResult` semantics, so a new backend proves itself by passing it.
5. `extract/runner.py` + `infovore extract` — pulls pending and stale exchanges in batches, gathers context messages and related claims, calls the extractor with the stage's concurrency limit; a `usage_limit` result pauses the runner until `retry_after` without touching retry counts; `--mode trial` never modifies live claims or exchange status; validates, persists claims and the run record, marks the exchange. Tests: a validation failure records a failed run and no claims; a success records claims with correct provenance; batch size and concurrency are respected; stale re-extraction supersedes the previous claims.
6. `extract/novelty.py` + `infovore probe` — closed-book novelty probe. For each `unprobed` claim: (a) ask the probe model the claim's `probe_question` with no context, telling it to answer "I don't know" if unsure; (b) a judge call compares that answer to the claim and returns `known` | `partial` | `unknown` | `contradicts`. Stores verdict, probe model, and answer; a failed probe leaves the claim `unprobed` with `probe_error` set. Protocol plus deterministic fake; the real implementation runs on whichever `LLMBackend` the probe and judge stages are configured with. The probe defaults to `claude_cli` with `sonnet`, the weakest model expected to consume the lore (the lead is Opus and subagents are Sonnet); what Sonnet already knows, Opus almost certainly does. The judge defaults to `haiku`. Extraction can run on a different backend (e.g. a local model) while the probe stays on Claude, because "net-new" is defined relative to Claude's knowledge. The canonical probe model id is recorded per verdict, so a model upgrade re-probes only the claims that model hasn't seen. Nothing is deleted on `known`; re-probing under a new probe model is idempotent per (claim, model).
7. `extract/review.py` + `infovore review` — prompt iteration loop over backfilled data. `infovore extract --sample N --seed S --mode trial` runs the current prompt version on a reproducible sample; `infovore review --run-ids ...` renders a single self-contained HTML report (exchange, claims, sources, novelty verdicts, tokens, cost), with a side-by-side diff when two prompt versions cover the same exchanges. `infovore promote --prompt-version V` records the promotion in `prompt_versions`, making V the live version.

Extraction prompt intent (write the actual text in README under "Extraction prompt" and mirror it in `prompt.py`): the model is told it is reading an archived exchange from a hobbyist SGI/IRIX community; its job is to capture domain knowledge a general-purpose LLM would not already have: specific part numbers, jumper settings, PROM/firmware versions, IRIX quirks and workarounds, repair procedures, compatibility facts, sources for software and manuals. Generic computing knowledge is not wanted. It extracts generously; the novelty probe is the filter, not the extractor. Each claim must be specific and supported by the messages; each carries a `probe_question` that asks for the fact without revealing it; corrections of a supplied related claim cite its id; community corrections within the exchange yield the corrected version only; chatter, opinions, and questions without an answer yield zero claims; every claim cites message ids. Reaction counts are provided as a weak signal of community agreement. Output is a JSON object matching `schema.py`, nothing else.

Why a closed-book probe rather than asking "did you already know this?": shown the answer, models reliably overclaim familiarity. Answering the question cold tests recall, not recognition. `contradicts` (the model confidently believes something else) is the most valuable verdict, since that is where an agent would otherwise be confidently wrong.

Acceptance: end-to-end test with fake source, real chunker, fake extractor, fake novelty probe produces the expected claims table with verdicts from a seeded conversation fixture.

### Phase 5 — Product schema

The SQLite file is the deliverable. Consumers (other tools, agents' own tooling) read it directly; infovore exposes nothing at runtime.

Tasks:

1. `lore` view and schema contract (lead) — the view defined in the data model, `PRAGMA user_version` set by migrations, and README "Consuming the database" with the view's columns, their meaning, example queries (including FTS via `claims_fts`), and the stability promise (columns are only added, never renamed or removed, without a version bump). Tests: seeded DB where the view returns exactly the current, live, probed-and-not-known, non-retracted claims; superseded and trial claims are excluded; a read-only connection (`mode=ro`) works while a writer holds the DB in WAL mode.

Acceptance: a consumer can answer "what do we know about X" with one documented SQL query against `lore`.

### Phase 6 — Operations

Tasks:

1. `infovore run` — live ingest plus a periodic chunk → extract → probe loop, sharing the runner pause semantics; graceful shutdown on SIGTERM. Tests with the fake source and fake backend.
2. Packaging and deployment + `infovore snapshot` — nothing host-specific in code. A multi-arch (amd64 + arm64) OCI image built in CI, with a build arg to include the `claude` CLI; `uv tool install` works as the non-container path. The DB path and scratch dir are volumes/config. `infovore snapshot <dest>` writes a consistent copy of the product DB with the SQLite backup API, safe while a writer is running, for shipping the product off the host. README "Deployment" gives short recipes, not code: laptop, macOS launchd (Mac Studio), Linux systemd, a container on unknown hardware, and AWS (container plus a persistent block volume; SQLite must not live on network filesystems such as EFS). It also covers how each LLM backend authenticates on a headless host (`claude setup-token` or cloud credentials; endpoint URL and key).
3. README completion: every section filled; a "Runbook" subsection with the exact commands for first backfill, enabling the intent, and posting the server notice.

Acceptance: `infovore run` against the fake source and fake backend ingests, groups, extracts, and probes a scripted event sequence, then shuts down cleanly on SIGTERM; the image runs `infovore status` on both arches; README sections match the code as built.

### M3 — deterministic triage

Most Discord chatter carries no lore. **M3** builds a triage pipeline, tracked as GitHub milestone "M3: deterministic triage", so most exchanges never reach an LLM. Each step depends on the one before it and is its own issue:

1. **Rules (#82)** — `infovore/triage/score.py` scores every exchange deterministically and for free from programmatic signals (part numbers, IRIX versions, code, tiny-message ratio, and the rest; see README "Triage"), writing `exchanges.triage_score` / `triage_reasons` / `triage_version`.
2. **Labels (#87)** — `exchange_labels`, `infovore/db/labels.py`, and `infovore label` turn rule scores and trial extraction into ground truth: `--from-runs` derives an `llm`-sourced `lore`/`noise` label per trial run from its claims' novelty verdicts, and `--exchange-id --lore|--noise` records a `human`-sourced hand correction that always wins over an LLM label for the same exchange. Labels accumulate over time and are never thrown away; they are the training data for the next step.
3. **Bayes classifier (#84)** — `infovore/triage/bayes.py` trains a naive Bayes classifier (Robinson/Fisher, standard library only) on the accumulated labels via `effective_labels`, evaluates precision/recall/F1 on a deterministic held-out split, and recommends an extraction threshold from measured recall at that split, not a guess. `infovore extract --mode trial --strategy uncertain|stratified|random` samples cheaply to grow the label set faster.
4. **Gating (#83)** — `extract` (and `run`'s periodic cycle) skip exchanges scoring below the calibrated threshold — `p_lore` once a model exists, the rule score before that (cold start) — so the LLM only ever sees the exchanges worth its cost.

## Definition of done

- All phases merged to main; CI green; coverage 100% line and branch.
- `README.md` is the only documentation and is complete.
- A single fixture-driven end-to-end test exercises backfill → chunk → extract → probe → `lore` view using only fakes.
- Import boundaries enforced by ruff `banned-api`: `discord` only in `source/live.py`; `subprocess` / `asyncio.create_subprocess_exec` only in `llm/claude_cli.py`; `openai` / `httpx` only in `llm/openai_compat.py`. Nothing above `llm/` knows which backend is in use.

## Deferred (do not build yet)

- Attachment download and image description.
- Attachment OCR for manual PDFs.
- Multi-guild support.
- Native Anthropic-SDK backend (`llm/anthropic_api.py`) for an API key or Bedrock: prompt caching, batches, strict structured outputs. Must pass the backend contract suite.
