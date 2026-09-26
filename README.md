# infovore

## Purpose

## Architecture

## Data model

## Configuration

Settings are read from the process environment by `infovore.config.load_settings`. `infovore.config.settings_from_environment(environ, dotenv_path)` first loads an optional `.env` file (`infovore.config.read_dotenv`: `KEY=VALUE` lines, blank lines and lines starting with `#` are ignored, surrounding single or double quotes on the value are stripped, a missing file yields no values) and then overlays the real environment on top of it, so real environment variables always win over the `.env` file. Startup fails loudly: every problem (missing required value, a value that fails to parse, an empty channel list, a non-positive number) is collected and raised together in one `ConfigError`, so all problems are visible at once instead of one at a time. The discord token is never included in any error message or in `repr()`/`str()` of the settings object.

| Variable | Default | Meaning |
| --- | --- | --- |
| `INFOVORE_DISCORD_TOKEN` | *(required)* | Discord bot token. Never logged or included in error messages. |
| `INFOVORE_GUILD_ID` | *(required)* | Discord guild (server) id to operate in. Must be a positive integer. |
| `INFOVORE_CHANNEL_IDS` | *(required)* | Comma-separated list of allowlisted channel ids. Must be non-empty; every entry must be a positive integer. |
| `INFOVORE_DB_PATH` | *(required)* | Filesystem path to the SQLite database file. |
| `INFOVORE_SCRATCH_DIR` | `scratch` | Working directory for backend scratch files (e.g. an empty cwd for `claude_cli` subprocesses). |
| `INFOVORE_QUIET_GAP_MINUTES` | `30` | Minutes of silence in a channel before the quiet-gap grouping rule closes an exchange. Must be a positive integer. |
| `INFOVORE_BATCH_SIZE` | `10` | Number of pending exchanges processed per extraction/probe batch. Must be a positive integer. |
| `INFOVORE_MAX_RETRIES` | `3` | Maximum retry count for a failed exchange before it stops being retried automatically. Must be a positive integer. |
| `INFOVORE_EXCHANGE_MAX_MESSAGES` | `50` | Maximum messages in one exchange before it is split (Lifecycle rule 6). Must be a positive integer. |
| `INFOVORE_OPT_OUT_ROLE` | `no-archive` | Name of the Discord role that opts a member's messages out of extraction. |
| `INFOVORE_INCLUDE_BOT_MESSAGES` | `false` | Whether bot-authored messages are ingested. Accepts `1`/`true`/`yes`/`on` and `0`/`false`/`no`/`off` (case-insensitive). |

Each stage — `extract`, `probe`, `judge` — has its own backend selection, all under an `INFOVORE_<STAGE>_*` prefix (`<STAGE>` is `EXTRACT`, `PROBE`, or `JUDGE`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `INFOVORE_<STAGE>_BACKEND` | `claude_cli` for every stage | Backend name for that stage, looked up in the `llm.registry.Registry` (e.g. `claude_cli`, `openai_compat`, `fake`). |
| `INFOVORE_<STAGE>_MODEL` | `sonnet` (`extract`, `probe`), `haiku` (`judge`) | Model alias or id passed to the stage's backend. |
| `INFOVORE_<STAGE>_CONCURRENCY` | `2` | Concurrent in-flight requests allowed for that stage. Must be a positive integer. |
| `INFOVORE_<STAGE>_TIMEOUT` | `60` | Per-request timeout in seconds for that stage. Must be a positive number. |
| `INFOVORE_<STAGE>_<KEY>` | *(none)* | Any other `INFOVORE_<STAGE>_*` variable is passed through to that stage's `StageSettings.options` under its lowercased key (e.g. `INFOVORE_EXTRACT_BINARY_PATH` becomes `options["binary_path"]`), for backend-specific settings such as `claude_cli`'s binary path or `openai_compat`'s base URL and key. |

`infovore.llm.registry.Registry` maps each stage's configured backend name to a `BackendFactory` (`name`, `validate(StageSettings) -> list[str]`, `build(StageSettings) -> LLMBackend`); `default_registry()` registers the `fake`, `claude_cli`, and `openai_compat` factories. `Registry.validate(settings)` reports an unknown backend name per stage plus anything the matching factory's own `validate` rejects; `Registry.build_backends(settings)` raises `ConfigError` if validation fails, otherwise returns one backend per stage; `Registry.health_check(backends)` sends one trivial request per backend and reports `None` on success or the error message on failure, without ever raising.

### `claude_cli` backend

`infovore.llm.claude_cli.ClaudeCliBackend` runs the `claude` CLI headless, once per `complete()` call, through an injected `infovore.llm.process.ProcessRunner`; it is the only module allowed to spawn a process (`subprocess`/`asyncio.create_subprocess_exec` banned everywhere else by ruff `banned-api`). `ClaudeCliFactory` (registry name `claude_cli`) validates that `StageSettings.model` is non-empty and builds a backend with `SubprocessRunner`, the stage's `model`, `concurrency`, and `timeout_seconds`, `options["binary"]` (default `claude`), and `options["scratch_dir"]` (default `scratch`, the same default as the top-level `INFOVORE_SCRATCH_DIR`).

Argv, in exact order: `<binary> -p --model <model> --system-prompt <request.system> --tools "" --strict-mcp-config --setting-sources "" --no-session-persistence --output-format json`, plus `--json-schema <json.dumps(request.json_schema, sort_keys=True)>` when the request carries a schema. `--bare` is never passed — it makes the CLI read only `ANTHROPIC_API_KEY` and ignore the Max-plan OAuth login or a `claude setup-token` token. The prompt goes on stdin. `cwd` is a fresh, empty directory created with `tempfile.TemporaryDirectory` under `scratch_dir` (created with `parents=True` if missing) for every call, so no `CLAUDE.md` is ever discovered there; the directory is removed again once the call finishes, whether it succeeds, returns an error result, or the runner raises, so a long-running process never accumulates empty scratch directories.

`claude -p --output-format json` emits one JSON object on stdout. Mapping to `LLMResult`:

- `structured_output` (schema requests) or `result` (plain text); a schema request whose response has no `structured_output` object is `fatal`.
- The canonical model id is the `modelUsage` key with the highest `outputTokens` (ties broken arbitrarily by iteration order); if `modelUsage` is absent or empty, the requested model alias is used instead.
- `usage.input_tokens` / `usage.output_tokens` and `total_cost_usd` (an API-equivalent cost, informational) populate `Usage`.
- A timed-out run is `transient`. A non-zero exit with stdout that isn't a JSON object is `transient` when stderr looks like a network error (connection refused/reset, unreachable network, DNS/`getaddrinfo`, a socket error) and `fatal` otherwise.
- Within a parsed `is_error` payload: `api_error_status == 429`, or `result`/`stderr` mentioning "usage limit", "rate limit", or "limit reached" (case-insensitive), is `usage_limit`, with `retry_after` parsed from an ISO-8601 timestamp or a 10-digit Unix epoch seconds value found in that text (rounded to the nearest second, floored at zero), defaulting to `300.0` seconds when no such timestamp is present. `api_error_status == 529` or in `500..599` is `transient`. `api_error_status` in `401`/`403`, or "not logged in"/"authentication" in the text, is `fatal`. Anything else `is_error` is `fatal`.

Capabilities: `native_json_schema=True`, `max_concurrency` is the stage's configured concurrency.

### `openai_compat` backend

`infovore.llm.openai_compat.OpenAICompatFactory` (registered as `openai_compat`) drives any OpenAI-compatible Chat Completions endpoint through the `openai` SDK: vLLM, llama.cpp, Ollama, LM Studio/MLX, or a hosted provider. It is the only module allowed to import `openai` or `httpx2` (enforced by ruff `banned-api`; classic `httpx` stays banned everywhere too, in case a future dependency bump reintroduces it).

A stage picks this backend with `INFOVORE_<STAGE>_BACKEND=openai_compat` and configures it with these `INFOVORE_<STAGE>_*` options (passed through `StageSettings.options`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `INFOVORE_<STAGE>_BASE_URL` | *(required)* | Base URL of the OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1` for Ollama. |
| `INFOVORE_<STAGE>_API_KEY` | *(required)* | API key sent to the endpoint. Never logged or included in error messages; local servers that don't check it still need a placeholder value such as `ollama`. |
| `INFOVORE_<STAGE>_JSON_SCHEMA_SUPPORTED` | `false` | `true` or `false` (case-insensitive). Whether the endpoint supports native `response_format` json_schema output. When `false`, the backend asks for plain text and the extractor layer extracts the first JSON object from it instead. |

Examples:

```
INFOVORE_EXTRACT_BACKEND=openai_compat
INFOVORE_EXTRACT_BASE_URL=https://vllm.internal.example/v1
INFOVORE_EXTRACT_API_KEY=sk-...
INFOVORE_EXTRACT_JSON_SCHEMA_SUPPORTED=true

INFOVORE_PROBE_BACKEND=openai_compat
INFOVORE_PROBE_BASE_URL=http://localhost:11434/v1
INFOVORE_PROBE_API_KEY=ollama
INFOVORE_PROBE_MODEL=llama3.1
INFOVORE_PROBE_JSON_SCHEMA_SUPPORTED=false

INFOVORE_JUDGE_BACKEND=openai_compat
INFOVORE_JUDGE_BASE_URL=http://localhost:1234/v1
INFOVORE_JUDGE_API_KEY=lm-studio
```

Error mapping: HTTP 429 maps to `transient` and honors a `retry-after` header, except when the response body reports the OpenAI error code `insufficient_quota`, which maps to `usage_limit` instead (a hard billing/plan cap rather than a retryable rate limit) — also honoring `retry-after` when present. HTTP 5xx and connection/timeout errors map to `transient`. HTTP 401/403/400/404 map to `fatal`. A `finish_reason` of `length` or `content_filter`, an empty completion, or (when a JSON schema was requested and natively supported) a response that fails to parse as JSON all map to `fatal`. Usage is recorded from the response when the endpoint reports it, `None` otherwise; the model id is whatever the server echoes back, falling back to the configured model if blank.

## Running

```
uv run infovore status
uv run infovore backfill [--page-size N]
uv run infovore chunk [--now 2026-01-01T00:00:00+00:00]
uv run infovore extract [--mode trial|live] [--sample N] [--seed S] [--exchange-id ID ...]
uv run infovore snapshot <dest> [--force]
```

`chunk` groups ingested messages into exchanges and persists the closed ones; `--now` overrides the clock, which is useful when iterating over an old backfill.

`snapshot` writes a consistent copy of the product database to `<dest>` using the SQLite backup API (`infovore.db.snapshot.snapshot`), safe to run at any time, including while `backfill`/`chunk`/`extract`/`run` is mid-write against the same file: the backup only ever sees committed data, never a writer's in-flight transaction. It refuses to overwrite an existing `<dest>` unless `--force` is given, creates `<dest>`'s parent directories as needed, and writes through a temporary file in the same directory that it atomically renames into place, so a reader never observes a partially written snapshot. It reports the destination path, its size in bytes, and its `PRAGMA user_version` (the schema version). This is how the product database leaves a host — see "Deployment" below.

Every subcommand loads configuration (environment, then `.env` in the working directory for anything not set), opens and migrates the database, and runs. `status` prints row counts, the exchange queue by status, claims by novelty, run outcomes, the last extraction and probe times, the live prompt version, and each stage's backend and model.

`backfill` walks every channel in `INFOVORE_CHANNEL_IDS`, plus their threads (including archived ones), oldest message first, from `infovore.ingest.backfill.backfill`. Each channel's checkpoint (`channels.last_backfilled_message_id`) is stored after every page of messages, in the same transaction as that page's rows, so an interrupted run resumes exactly where it left off and never re-walks or loses data. A page that fails with a rate limit sleeps for the retry-after duration and retries from the checkpoint; a page that fails because the source is unavailable backs off exponentially (1s, 2s, 4s, ...); a channel that fails five consecutive times (configurable via the `backfill()` function's `max_attempts`) is recorded as failed and the walk continues with the remaining channels. `backfill` prints one line per channel (pages walked, messages inserted/updated/unchanged, messages skipped as system or bot) and one line per failed channel, then exits `0` if every channel completed or `1` if any channel failed. `--page-size` (default `100`) controls how many messages are requested per history page. `backfill` and `sync-optouts` log in to Discord with `INFOVORE_DISCORD_TOKEN`, wait up to 60 seconds for the gateway to be ready, and always close the connection when they finish. A bad token, missing privileged intents, or a timeout exits `3` without printing the token.

`extract` (`infovore.extract.runner.run_extraction`, `infovore.extract.command.ExtractCommand`) turns pending and stale exchanges into claims with the extract stage's configured backend (`LLMClaimExtractor`), model, and concurrency. `--mode` defaults to `live`: it registers the current prompt version (`infovore.extract.prompt.PROMPT_VERSION`/`PROMPT_SHA256`) and refuses to run unless that exact version is the live, promoted one (exit `2`, "run `infovore promote --prompt-version vN`"), then repeatedly pulls up to `INFOVORE_BATCH_SIZE` claimable exchanges (`pending` or `stale`, under `INFOVORE_MAX_RETRIES`) until none remain. In either mode, an exchange whose every message is from an opted-out author is skipped without calling the extractor (it would only ever see `[redacted]`) and counted in `report.skipped`; only in live mode is the exchange's status also written to `skipped` — trial mode leaves status and retry count untouched, same as any other trial-mode exchange. On success, claims are recorded under the live run and the exchange is marked `done` (zero claims is still `done`); a `stale` exchange's previous live claims are retracted (`reextracted`) in the same step. A `transient`/`fatal`/`invalid_output` failure records a failed run and increments the exchange's retry count, marking it `failed` once `INFOVORE_MAX_RETRIES` is reached; a `usage_limit` result pauses on the configured `Sleeper` for its `retry_after` (default 300s) without touching retry counts, then retries the same exchange. `--mode trial` runs the extractor over an explicit set of exchanges (any status) picked with `--sample N` (a reproducible sample via `infovore.extract.runner.select_trial_sample`, stratified by channel and exchange size bucket, using `--seed`, default `0`) and/or repeatable `--exchange-id`; trial mode never changes exchange status, retry counts, or existing claims, and its claims are recorded under a `trial` run for prompt iteration. Extractor calls are bounded by the extract stage's configured concurrency (`asyncio.Semaphore`). `extract` prints processed/succeeded/failed/skipped/claims_recorded/pauses counts and the recorded run ids, and exits `1` if any exchange failed.

Exit codes: `0` ok, `1` unexpected failure, `2` configuration or usage error, `3` Discord or an LLM backend is unavailable.

### First run against a real server

1. In the Discord Developer Portal, create an application and add a bot. On the Bot tab, enable the two privileged intents **Message Content** (message bodies) and **Server Members** (role membership, for opt-out sync). Without both, the gateway rejects the connection.
2. Invite the bot to the server with the `bot` scope and the **View Channels** and **Read Message History** permissions. It never needs to send messages.
3. Create the opt-out role (default name `no-archive`, `INFOVORE_OPT_OUT_ROLE`) and post the server notice explaining what is archived and how to opt out.
4. Set `INFOVORE_DISCORD_TOKEN`, `INFOVORE_GUILD_ID`, `INFOVORE_CHANNEL_IDS` and `INFOVORE_DB_PATH` (see Configuration).
5. `infovore sync-optouts`, then `infovore backfill`, then `infovore chunk`. Run `sync-optouts` before any extraction, and again whenever the role changes.

## Grouping rules

`infovore/chunk/rules.py` groups the `MessageRow`s of one channel (plus its threads, which carry a `thread_id`) into `Group`s. Each `Group` has a `rule` (`GroupingRule.THREAD` | `REPLY_CHAIN` | `QUIET_GAP`), an ordered `messages` tuple, and a `context` tuple of uncitable overlap messages, non-empty only for parts produced by the size-cap split below. Every message given to `group_messages` ends up in exactly one group's `messages`, or is dropped first. Messages within a group, and groups within the returned list, are ordered by `(created_at, id)` (a group's position is its first message's key).

**Drops** (lifecycle rule 7): `drop_ungroupable` removes deleted messages (`deleted_at` set) unconditionally, and bot-authored messages (`author_is_bot`) unless `include_bots=True`. System messages never reach the chunker; they are filtered at ingest/normalize time and have no representation in `MessageRow`.

**Precedence** is thread > reply_chain > quiet_gap. Each rule claims messages from what the previous rule left behind; nothing is grouped by more than one rule.

- **thread**: every kept message with a non-null `thread_id` is grouped with every other kept message sharing that `thread_id`, one group per distinct `thread_id`. This runs first, so a message inside a thread is never pulled into a reply chain or quiet-gap group with anything outside its thread, even if it replies to a message outside the thread — that parent is simply not part of the thread's group, and is handled by a later rule on its own. This is the "reply inside a thread whose parent is outside the thread" case: the reply stays in its thread group; the outside parent, if it has no other links, becomes its own quiet_gap group of one.
- **reply_chain**: among messages not claimed by `thread`, this is the transitive closure over `reply_to_id` — the connected components of the reply graph, treating each message as a node and each `reply_to_id` that points at another message in the same input as an edge. A component is a `reply_chain` group only if it has more than one message; a message with no reply links, or whose only reply link points outside this input (already grouped by `thread`, in a different channel, or missing entirely), forms a component of size one and falls through to `quiet_gap` instead. A cycle in the reply graph (possible only from malformed data) is tolerated: union-find just treats it as an already-merged edge and the messages still end up in one group.
- **quiet_gap**: the messages left after `thread` and `reply_chain` are sorted by `(created_at, id)` and split into a new group whenever the gap between consecutive messages strictly exceeds `quiet_gap` (a `timedelta`, default 30 minutes — exactly 30 minutes does not split). A single leftover message becomes a group of one.

**Size cap** (lifecycle rule 6): after the three rules produce their groups, `split_oversized` splits any group whose `messages` exceeds `max_messages` (default 50) into consecutive, non-overlapping parts, each within the cap, by repeatedly cutting the longest prefix that still fits. Concretely: while more than `max_messages` messages remain unsplit, look at the first `max_messages + 1` of them, find the internal gap (the time delta between two consecutive messages) that is largest — ties broken toward the later position, so a run of equal gaps fills each part up to the cap rather than making many tiny parts — and cut the prefix there; that prefix (at most `max_messages` messages, at least one) becomes the next part, and the scan continues on the rest. The final remainder, now `<= max_messages`, becomes the last part. Each part after the first carries the previous part's last `overlap` messages (default 3) as its `context`, capped at however many messages that previous part actually has; `context` never contains any message that is also in the same part's `messages`, and it is never itself split or citable.

**Closing** (lifecycle rule 2): `is_closed(group, now, quiet_gap)` reports a group as closed only once its newest message's `created_at` is strictly older than `quiet_gap` relative to `now`; at exactly `quiet_gap` it is still open. This applies uniformly to `thread`, `reply_chain`, and `quiet_gap` groups alike — the grouper (#16) uses it to decide when a group is done accumulating messages and can be persisted as an exchange.

**Persisting** (`infovore/chunk/grouper.py`): `group_pending(conn, clock, quiet_gap, max_messages, include_bots)` loads, for every distinct `channel_id` with at least one message not yet in `exchange_messages` (a thread's messages carry their thread's id as `channel_id`, so a thread is naturally its own batch), every such message ordered by `(created_at, id)`, and runs `group_messages` over that batch. Only groups for which `is_closed(group, clock.now(), quiet_gap)` is true are persisted; the rest are left ungrouped for a later run (lifecycle rule 2) and counted as deferred. Each persisted group becomes one `exchanges` row via `insert_exchange`, in its own transaction: `content_hash` is the hex SHA-256 of the group's message ids, in the same `(created_at, id)` order as `messages`, joined by `,`; `started_at`/`ended_at` come from the first/last message; `grouping_rule` and `channel_id`/`thread_id` come from the group and the batch's channel. `group_pending` returns a `GroupingReport(exchanges_created, messages_grouped, groups_deferred)`. Re-running with nothing new to group creates nothing, since a message already in `exchange_messages` is never loaded again; if `insert_exchange` reports `DuplicateExchangeError` (its `content_hash` already exists) the group is treated as already persisted and grouping continues with the next one.

`parent_exchange_id` is resolved per group, in this precedence, highest first:

1. **Size-cap split part** (lifecycle rule 6): if the group is part `N > 0` of an oversized group's split (its `context` is non-empty), the parent is whichever exchange already owns `context`'s messages — that is always part `N - 1`, since a later part can only be closed once every earlier part of the same split is too. The renderer takes the parent's last messages (its `context` at persist time) as uncitable background.
2. **Late reply** (lifecycle rule 1): otherwise, if the group's first message (by `(created_at, id)`) has a non-null `reply_to_id` and that target message already belongs to an exchange, the parent is that exchange.
3. **Thread revival**: otherwise, if the group is a `thread` group and that `thread_id` already has at least one exchange, the parent is the most recently started one.

A group matching none of these is a new, unparented exchange. Precedence matters because a group can match more than one case at once — a split part whose first message is also a late reply still links to the previous part, not to the reply target, and a revived thread whose first message is a late reply links to the reply target, not to the thread's own prior exchange.

## Extraction prompt

`infovore.extract.prompt` assembles a versioned prompt from an `infovore.extract.protocol.ExtractionRequest` (built by `infovore.extract.request.build_request` from an exchange). `PROMPT_VERSION` (currently `v1`) identifies the exact system prompt below, whose sha256 is `PROMPT_SHA256`; `prompt_version` is written to every `extraction_runs` row so a change to the wording is a new version, never a silent edit. `render_prompt(request)` returns a `RenderedPrompt(system, prompt, version, token_estimate)`; `token_estimate` is `ceil(len(system + prompt) / 4)`.

The system prompt, verbatim:

```
You are reading an archived exchange from a hobbyist SGI/IRIX community.

Your job is to capture domain knowledge a general-purpose LLM would not already have: specific part numbers, jumper settings, PROM/firmware versions, IRIX quirks and workarounds, repair procedures, compatibility facts, and sources for software and manuals. Generic computing knowledge is not wanted.

Extract generously. A later closed-book novelty probe is the filter, not you: your job is to notice everything specific and supported by the messages, not to decide whether it is already widely known.

Each claim must be specific and supported by the messages. Each claim carries a probe_question that asks for the fact without revealing it, so the fact can be tested for later without leaking the answer.

If a claim corrects one of the supplied related existing claims, cite that claim's id in supersedes. If the community corrects itself within this exchange, extract only the corrected version, never the original mistake.

Chatter, opinions, and questions that are never answered yield zero claims.

Every claim must cite the ids of the messages in this exchange that support it. Never cite a message from the CONTEXT section: those messages are read-only background from a prior exchange and cannot be cited.

Reactions are provided as a weak signal of community agreement, not proof.

Output ONLY a JSON object matching the given schema. No other text.
```

The user prompt (`RenderedPrompt.prompt`) lays out, in order:

- `CHANNEL:` the channel name (falls back to the numeric channel id when the channel is unknown) and `PERMALINK:` the exchange's permalink, `https://discord.com/channels/{guild_id}/{channel_id}/{first_message_id}` (built by `infovore.extract.prompt.permalink`).
- `CONTEXT (do not cite):`, present only when the exchange has a `parent_exchange_id` — the last `context_size` messages of the parent exchange, read-only and never citable.
- `EXCHANGE:` — every message of the exchange itself, each rendered as `[id] author @ ISO-8601 timestamp:` followed by its content, then an optional `Reactions: emoji×count, ...` line and an optional `Attachments: filename, ...` line.
- `RELATED EXISTING CLAIMS:` — up to `related_limit` claims from `claims_fts` matching the exchange's own message contents (excluding any message authored by an opted-out user, so their words never influence what is sent to the model), each as `[claim:<id>] (<kind>) <subject>: <statement>`, or the literal `none` when there are no matches.

Before rendering, any message whose author has opted out (`infovore.db.raw.opted_out_user_ids`) has its author and content replaced with `[redacted]`; the message id is kept so citations and ordering stay consistent. Rendering is otherwise pure and deterministic: the same `ExtractionRequest` always renders to the same `RenderedPrompt`, and no wall-clock time is read.

### Backend-neutral extraction

`infovore.extract.llm_extractor.LLMClaimExtractor` implements `ClaimExtractor` (#5) against any `LLMBackend` (#5), never branching on which backend is configured. `LLMClaimExtractor(backend, max_output_tokens=8000)` renders the prompt (above), requests `json_schema_for(ExtractionOut)`, and reads the result: when `backend.capabilities().native_json_schema` is true and `result.structured` is present, that structured payload is used directly; otherwise the first JSON object is extracted from `result.text` (`schema.first_json_object`). The payload is always validated with `schema.parse_extraction`, given the exchange's own message ids as the citable set and the ids of the supplied related claims as the valid `supersedes` targets.

On `InvalidExtractionError`, exactly one repair call is made: same system prompt, and a user prompt that is the original prompt plus a clearly delimited section (`--- PREVIOUS OUTPUT (invalid) ---` / `--- VALIDATION ERROR ---`) quoting the previous output verbatim and the validation error, asking for a corrected JSON object only. If the repair call itself returns an `LLMResult` error, that error is mapped normally (below); if the repair call succeeds but its payload still fails `parse_extraction`, the outcome is `Failure(FailureKind.INVALID_OUTPUT, <validation error>)` with `model=None`. There is no second repair attempt.

`LLMResult` errors map to `Failure` kinds one-for-one: `ErrorKind.TRANSIENT` -> `FailureKind.TRANSIENT`, `ErrorKind.FATAL` -> `FailureKind.FATAL`, `ErrorKind.USAGE_LIMIT` -> `FailureKind.USAGE_LIMIT` (carrying `retry_after`). This applies to an error from either the initial call or the repair call. `ExtractionOutcome.model` is always the canonical model id from `LLMResult.model` (the backend's own resolved id, never a configured alias) on the call that ultimately produced the claims — the initial call's model normally, or the repair call's model when a repair was needed. `input_tokens`/`output_tokens` are summed across the initial and repair calls (a missing count from one call is treated as zero once the other call reports a count); if neither call reports token counts, both fields are `None`.

### Novelty probe

`infovore.extract.llm_extractor.LLMNoveltyProbe` implements `NoveltyProbe` (#5) as two backend-neutral calls, `LLMNoveltyProbe(probe_backend, judge_backend)`:

1. **Recall** — `probe_backend` is asked the claim's `probe_question`, and nothing else: no exchange text, no statement, no claim id. The system prompt (`RECALL_SYSTEM_PROMPT`, verbatim below) tells it to answer from its own knowledge only, briefly, and to say exactly "I don't know" if unsure. The schema is `RecallOut`; native structured output is used when the backend reports it, otherwise `first_json_object` on the text, same as extraction.
2. **Judge** — `judge_backend` is given the recall answer alongside the claim's subject and statement, and asked to return a `JudgeOut` verdict: `unknown` (the answer says it doesn't know, or is unrelated), `partial` (some but not all of the specifics), `contradicts` (the answer confidently asserts something incompatible with the claim), or `known` (substantively the same fact). The system prompt is `JUDGE_SYSTEM_PROMPT`, verbatim below.

`ProbeOutcome.model` is always the **recall** call's canonical model id — that is the model whose knowledge was actually probed; the judge model is not recorded, since "net-new" is defined relative to the probe model's knowledge, not the judge's. `ProbeOutcome.answer` is the recall answer once the recall call has produced one, even if a later step (the judge call) fails. An error from either call maps to a `Failure` exactly as in extraction (`TRANSIENT`/`FATAL`/`USAGE_LIMIT`); invalid JSON from either call is `FailureKind.INVALID_OUTPUT`. Unlike extraction, the probe makes **no repair call** on invalid output from either the recall or the judge call — keeping the probe simple was chosen over matching the extractor's one-repair-attempt behavior, since a probe/judge failure just leaves the claim `unprobed` for a later run (Phase 4 task 6), rather than losing a batch of extracted claims.

`RECALL_SYSTEM_PROMPT`, verbatim:

```
Answer the following question using only your own knowledge, with no other context. Be brief. If you are not sure of the answer, respond with exactly "I don't know".
```

`JUDGE_SYSTEM_PROMPT`, verbatim:

```
You are comparing a closed-book recall answer against a claim, to judge how much the answering model already knew.

Return unknown if the answer says it does not know, or is unrelated to the claim.
Return partial if the answer gives some but not all of the claim's specifics.
Return contradicts if the answer confidently asserts something incompatible with the claim.
Return known if the answer is substantively the same fact as the claim.

Output ONLY a JSON object matching the given schema. No other text.
```

## Privacy and opt-out

A Discord role (`INFOVORE_OPT_OUT_ROLE`, default `no-archive`) lets a guild member opt their messages out of extraction. `infovore sync-optouts` fetches the role's current members via `DiscordSource.role_member_ids` and reconciles them against the `opt_outs` table (`infovore.privacy.optout.sync_opt_outs`): a member holding the role who is not yet in `opt_outs` is added with `since` set to now; a member in `opt_outs` who no longer holds the role is removed. Removing a user from `opt_outs` only stops future redaction — **opting back in never restores previously redacted history**, since redaction is destructive (the original content is overwritten, not merely hidden).

For each newly added user, every message they already authored is redacted in place: `content` and `author_name_at_time` become `[redacted]`, `raw_json` becomes `{}`, every `message_revisions` row for their messages is redacted the same way, and their `attachments` rows are deleted. `reactions` are left untouched (no author-identifying content). This redaction is a set of plain `UPDATE`/`DELETE` statements in one transaction; it never goes through `upsert_message`, so it never writes a new revision.

Going forward, every message from an opted-out author is redacted *before* it reaches storage: `infovore.privacy.optout.redact_normalized` replaces `content`, `author_name_at_time`, and `raw_json` (attachments dropped, reactions kept) on the `NormalizedMessage` produced by `ingest.normalize`, and this must run before `db.raw.upsert_message` is called. This ordering matters for backfill re-runs: `upsert_message` treats any difference in `content` from the stored row as an edit and writes a revision. If an already-opted-out author's original, unredacted content arrived from Discord again and were upserted directly, it would look like an edit and overwrite the redacted row (with the unredacted text saved as a "prior" revision). Redacting first means the incoming row always matches the already-redacted stored row, so the upsert is a no-op and no revision is ever written — a backfill can be re-run any number of times over an opted-out author's history without ever un-redacting it.

Per lifecycle rule 5: when `sync_opt_outs` adds users, it also calls `db.claims.retract_claims_with_all_sources_opted_out`, which retracts (`retraction_reason = 'sources_opted_out'`) every claim whose *every* source message is authored by an opted-out user. A claim with at least one source from a non-opted-out author is kept; only its opted-out source messages are redacted.

`sync_opt_outs` logs one `logging` info line per added or removed user id (never message content), so the change is auditable without exposing what was said.

**M1 gate**: no extraction may run against real Discord data until this module is merged (Phase 2 task 6). Prompt-time redaction (`extract/prompt.py`, `ExtractionRequest.opted_out_user_ids`) depends on `opted_out_user_ids` from this module.

## Coverage exclusions

- `...` bodies: Protocol method stubs have no executable behavior; they define shapes that implementations are tested against.
- `if TYPE_CHECKING:` blocks: imports needed only by the type checker never run at runtime.
- `infovore.llm.claude_cli.SubprocessRunner.run`: this is the one place allowed to spawn a real process, and exercising it would mean either spawning the real `claude` binary (never done in tests: no network, no dependency on being logged in) or spawning some other process as a stand-in, which still violates "no test spawns a process." Every other `claude_cli` behavior (argv, stdin, cwd, result mapping, every `ErrorKind`) is tested against `ClaudeCliBackend` with a fake `ProcessRunner`; `SubprocessRunner` itself is a thin, direct translation of `asyncio.create_subprocess_exec` plus `asyncio.wait_for` with no branching of its own to verify beyond what the standard library already guarantees.
- `infovore.source.live.connect` and `infovore.source.live._real_start`: perform the real Discord login/gateway handshake over the network; PLAN operating rule 4 forbids tests from opening a network connection, so these one-line wrappers around `discord.Client.start` cannot be exercised in the test suite. The lifecycle around them (`open_discord_source`) is fully tested with a fake client and starter.

## Consuming the database

The SQLite file is the product. Read it directly; open it read-only (`file:infovore.db?mode=ro`) so readers never block the writer (the database runs in WAL mode).

### The `lore` view

`lore` is the contract. It contains only **current, net-new** claims:

- from `live` extraction runs (trial runs are for prompt iteration and never appear);
- not retracted (source messages deleted, or every source author opted out);
- not superseded by a newer live, non-retracted correction;
- novelty `unknown`, `partial`, or `contradicts` — claims the closed-book probe found the model did **not** already know. `known` and not-yet-probed (`unprobed`) claims are excluded.

| column | meaning |
| --- | --- |
| `claim_id` | stable id of the claim |
| `subject` | what the claim is about (e.g. `Octane2`, `IP35`) |
| `statement` | the fact itself |
| `kind` | `fact`, `correction`, `procedure`, or `reference` |
| `confidence` | extractor confidence, 0–1 |
| `novelty` | `unknown` (model had no idea), `partial`, or `contradicts` (model confidently believed something else — the most valuable) |
| `permalink` | Discord link to the exchange the claim came from |
| `source_message_ids` | comma-separated Discord message ids the claim cites, ascending |
| `channel_id` | channel or thread the exchange belongs to |
| `extracted_at` | when the extraction run started (ISO-8601 UTC) |
| `supersedes_claim_id` | the claim this one corrects, if any |

### Example queries

What do we know about a subject:

```sql
SELECT subject, statement, novelty, permalink
FROM lore
WHERE subject LIKE '%Octane%'
ORDER BY novelty = 'contradicts' DESC, confidence DESC;
```

Full-text search (the `claims_fts` index keeps part numbers, versions and paths such as `030-1234-001`, `6.5.22`, `/usr/sbin/inst` as single tokens; quote each term):

```sql
SELECT lore.subject, lore.statement, lore.permalink
FROM claims_fts
JOIN lore ON lore.claim_id = claims_fts.rowid
WHERE claims_fts MATCH '"030-1234-001" OR "Octane2"'
ORDER BY bm25(claims_fts, 2.0, 1.0);
```

Where the model is confidently wrong:

```sql
SELECT subject, statement, permalink FROM lore WHERE novelty = 'contradicts';
```

### Stability

`PRAGMA user_version` holds the schema version (the latest applied migration). Columns of `lore` are only ever added; renaming or removing one bumps the version and is called out here.

## Deployment

Nothing host-specific lives in code. Every setting comes from the environment (or a `.env` file next to the working directory, see "Configuration"); `INFOVORE_DB_PATH` and `INFOVORE_SCRATCH_DIR` are the only paths involved and both must sit on durable, local (non-network) storage. `infovore` is one console-script package (`uv tool install .`) plus one container image built from the repo's `Dockerfile`; the recipes below are the same few commands on every host.

### Laptop

```
uv tool install .
infovore backfill && infovore chunk
```

or, from a checkout without installing anything system-wide, `uv run infovore <command>`.

### macOS launchd (Mac Studio)

A `launchd` user agent runs a periodic `backfill` + `chunk` pair. Save as `~/Library/LaunchAgents/com.example.infovore.plist` and load with `launchctl load ~/Library/LaunchAgents/com.example.infovore.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.example.infovore</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string>
    <string>-c</string>
    <string>infovore backfill && infovore chunk</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>WorkingDirectory</key><string>/Users/you/infovore</string>
  <key>StartInterval</key><integer>1800</integer>
  <key>StandardOutPath</key><string>/Users/you/infovore/infovore.log</string>
  <key>StandardErrorPath</key><string>/Users/you/infovore/infovore.log</string>
</dict>
</plist>
```

`WorkingDirectory` is where `infovore` looks for `.env`; put every `INFOVORE_*` and backend credential variable there instead of in the plist so secrets never end up in `launchctl list` output.

### Linux systemd

A oneshot service plus a timer, run as the unprivileged user that owns the database:

```ini
# /etc/systemd/system/infovore.service
[Unit]
Description=infovore backfill + chunk

[Service]
Type=oneshot
User=infovore
EnvironmentFile=/etc/infovore/infovore.env
WorkingDirectory=/var/lib/infovore
ExecStart=/usr/local/bin/infovore backfill
ExecStart=/usr/local/bin/infovore chunk
```

```ini
# /etc/systemd/system/infovore.timer
[Unit]
Description=Run infovore periodically

[Timer]
OnBootSec=5min
OnUnitActiveSec=30min

[Install]
WantedBy=timers.target
```

`sudo systemctl enable --now infovore.timer`.

### Container on unknown hardware

Build with `docker build -t infovore .` (add `--build-arg WITH_CLAUDE_CLI=1` to bundle the `claude` CLI for the `claude_cli` backend; it adds Node.js and `@anthropic-ai/claude-code`, needed only for that backend). The image runs as a non-root user (uid/gid 1000) and declares `/data` as the volume holding both `INFOVORE_DB_PATH` (`/data/infovore.db` by default) and `INFOVORE_SCRATCH_DIR` (`/data/scratch` by default). A bind-mounted host directory must be owned by uid 1000 (or `chown 1000:1000` it first); a named Docker volume is populated with the image's own ownership automatically and needs no extra step:

```
mkdir -p data && sudo chown 1000:1000 data
docker run --rm -v "$PWD/data:/data" --env-file .env infovore backfill
docker run --rm -v "$PWD/data:/data" --env-file .env infovore chunk
```

or, once the long-running `run` loop (PLAN.md Phase 6 task 1) lands, `docker run -d --name infovore --restart unless-stopped -v "$PWD/data:/data" --env-file .env infovore run` keeps the same container alive as the live ingest + periodic pipeline. `--env-file .env` carries every `INFOVORE_*` setting plus whichever backend credentials apply (see "Headless auth" below); none of it needs to be baked into the image.

### AWS

Run the same image on ECS (Fargate or EC2 launch type) or a plain EC2 instance, with `/data` backed by a **persistent block volume** — an EBS volume, attached to the task (Fargate's EBS volume attachment support) or mounted on the instance and bind-mounted into the container (EC2 launch type or plain `docker run`). **Never put `INFOVORE_DB_PATH` on EFS, FSx, or any other network filesystem**: SQLite's locking (and WAL mode especially) depends on POSIX byte-range advisory locks that network filesystems emulate poorly or not at all, leading to silent corruption or "database is locked" errors that never clear. Credentials and endpoint URLs come from the task definition's environment/secrets (Secrets Manager or SSM Parameter Store), the same `INFOVORE_*` and backend variables as everywhere else.

### Headless auth per backend

- **`claude_cli`**: run `claude setup-token` once, interactively, on any machine with a browser and a Claude Pro/Max/Team/Enterprise subscription; it prints a one-year OAuth token and does not store it anywhere. Set that token as `CLAUDE_CODE_OAUTH_TOKEN` in the environment (or `.env`/secrets store) of the host that runs `infovore`. Verified 2026-09-26 against Claude Code's current documentation at `https://code.claude.com/docs/en/authentication` ("Generate a long-lived token"): *"The command opens the same browser authorization flow as `/login`, and the token prints to the terminal after you approve access in the browser. It does not save the token anywhere; copy it and set it as the `CLAUDE_CODE_OAUTH_TOKEN` environment variable wherever you want to authenticate."* This is also confirmed by running `claude setup-token --help` locally (a Claude Code v2.1.283 install), though the CLI's own `--help` text does not name the variable — only the docs page does. `infovore`'s `claude_cli` backend never passes `--bare`, and the same docs page states bare mode does not read `CLAUDE_CODE_OAUTH_TOKEN`, so this token works with it.

  Claude Code's own cloud-credential modes work the same way, as an alternative to a subscription token, also verified against the current docs on 2026-09-26:
  - Amazon Bedrock (`https://code.claude.com/docs/en/amazon-bedrock`): set `CLAUDE_CODE_USE_BEDROCK=1`, AWS credentials by any of the AWS SDK's normal means (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN`, `AWS_PROFILE`, or `AWS_BEARER_TOKEN_BEDROCK`), and `AWS_REGION` (falls back to `AWS_DEFAULT_REGION`, then the active AWS profile's region, then `us-east-1`).
  - Google Cloud's Agent Platform / Vertex AI (`https://code.claude.com/docs/en/google-vertex-ai`): set `CLAUDE_CODE_USE_VERTEX=1`, `ANTHROPIC_VERTEX_PROJECT_ID`, `CLOUD_ML_REGION`, and `GOOGLE_APPLICATION_CREDENTIALS` pointing at a service account key (or other Application Default Credentials).

  These are Claude Code's own environment variables, not infovore's; set them alongside `INFOVORE_*` in the same environment/`.env` file/secrets store, since the `claude_cli` backend spawns `claude` inheriting its process environment untouched.
- **`openai_compat`**: no interactive login — set `INFOVORE_<STAGE>_BASE_URL` and `INFOVORE_<STAGE>_API_KEY` (see "`openai_compat` backend" above for the full option list). Any OpenAI-compatible endpoint works: a hosted provider, or a local server (vLLM, llama.cpp, Ollama, LM Studio/MLX) reachable from the host running `infovore`.

## Development workflow

```
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Every change is red/green TDD on a feature branch named `<issue>-<slug>`, opened as a PR that references its issue. Coverage is enforced at 100% line and branch.

Import boundaries are enforced by ruff `banned-api`: `discord` only in `infovore/source/live.py`, process spawning only in `infovore/llm/claude_cli.py`, `openai`/`httpx`/`httpx2` only in `infovore/llm/openai_compat.py`.
