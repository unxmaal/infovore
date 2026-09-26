# infovore

## Purpose

infovore is a passive archivist for a hobbyist SGI/IRIX Discord community. It reads channel history and live messages, groups them into conversations ("exchanges"), and uses a language model to pull out specific factual claims — part numbers, jumper settings, PROM and firmware versions, IRIX quirks and workarounds, repair procedures, compatibility facts, where to find software and manuals — each tied back to the Discord messages it came from.

It keeps only **net-new** knowledge: things a frontier LLM does not already know from pretraining. Every claim is checked with a closed-book probe (the model is asked the question cold, and a judge compares its answer to the claim); only claims the model didn't know, only partly knew, or got wrong reach the product.

The product is a SQLite file (see "Consuming the database"). Nothing talks to infovore at runtime: it has no API, no MCP server, and no Discord commands, and it never posts.

## Architecture

```
 Discord ──► DiscordSource ──► ingest ─────────► raw tables
 (history,   source/live.py     backfill.py        channels, messages,
  events)    source/export.py   live.py            revisions, attachments,
             source/fake.py     normalize.py       reactions
                                privacy/optout.py  opt_outs (redaction before storage)
                                        │
                                        ▼
                               chunk/rules.py + grouper.py ──► exchanges (+ members)
                                        │
                                        ▼
            extract/request.py + prompt.py ──► extract/llm_extractor.py ──► claims + sources
                  extract/runner.py (trial | live)      │
                                        │               │  llm/protocol.py LLMBackend
                                        ▼               │   ├ llm/claude_cli.py   (claude -p)
                             extract/novelty.py ────────┤   ├ llm/openai_compat.py
                             recall + judge per claim   │   └ llm/fake.py
                                        │               │  llm/registry.py picks one per stage
                                        ▼
                                 lore view (the product) ──► infovore snapshot ──► consumers
```

- **Ingest.** `infovore backfill` walks every allowlisted channel and its threads oldest-first with a per-channel checkpoint, one transaction per page, against whichever `DiscordSource` is configured: `source/live.py` (`discord.py`, live Discord) or `source/export.py` (a DiscordChatExporter JSON export, no Discord API access at all — see "Configuration" and "Running" → "Runbook"). `infovore run` consumes live events, which only `source/live.py` produces. Every message is normalized, then redacted if its author has opted out, then upserted — so a re-run never un-redacts anything.
- **Chunking.** Messages are grouped by thread, then reply chain, then quiet gap, split when too large, and persisted only once closed (see "Grouping rules"). Late replies and thread revivals link to the earlier exchange as read-only context.
- **Extraction.** For each pending or stale exchange the runner builds a request (messages, uncitable context, related existing claims found through `claims_fts`, opted-out authors), renders the versioned prompt, and asks the extract-stage backend for JSON matching a strict schema, with one repair attempt. `trial` runs are for prompt iteration and never touch exchange status or the product; `live` runs require the prompt version to be promoted.
- **Novelty probe.** Each claim's `probe_question` is asked closed-book of the probe-stage model, and the judge-stage model classifies the answer as `unknown`, `partial`, `contradicts`, or `known`.
- **Backends.** Every LLM call goes through `LLMBackend`; each stage (extract, probe, judge) picks its backend and model in configuration, so extraction can run on a local model while the probe stays on Claude. Nothing above `llm/` knows which backend is in use.
- **Import boundaries** (enforced by ruff `banned-api`): `discord` only in `source/live.py`; process spawning only in `llm/claude_cli.py`; `openai`/`httpx2` only in `llm/openai_compat.py`.
- **One writer.** SQLite runs in WAL mode; the whole pipeline runs on the host that holds the file, and readers open it read-only. LLM backends may be remote.

## Data model

All timestamps are ISO-8601 UTC text; Discord ids are 64-bit integers. Migrations in `infovore/db/migrations/` are applied in order and recorded in `schema_migrations`; `PRAGMA user_version` is the latest applied migration.

| object | holds |
| --- | --- |
| `schema_migrations` | applied migration versions and when they ran |
| `channels` | channels and threads (`parent_id` set for threads), their names, and each one's backfill checkpoint `last_backfilled_message_id` |
| `messages` | every ingested message: author and display name at the time, content, reply and thread links, `edited_at`, `deleted_at` (rows are never deleted), raw JSON |
| `message_revisions` | the prior content of every edited message; redacted in place on opt-out |
| `attachments` | attachment metadata per message (files are not downloaded) |
| `reactions` | current reaction count per message and emoji |
| `opt_outs` | users holding the opt-out role, and since when |
| `exchanges` | grouped conversations: channel, thread, first/last message, grouping rule, `content_hash` (unique), `parent_exchange_id` for context, `extraction_status` (`pending`, `done`, `skipped`, `failed`, `stale`), retry count and last error, the deterministic `triage_score` / `triage_reasons` / `triage_version`, and the trained classifier's `p_lore` / `p_lore_model` (the `triage_model.version` that produced it; `NULL` until a model has scored the exchange) (see "Triage") |
| `exchange_messages` | ordered membership; a message belongs to at most one exchange |
| `prompt_versions` | every extraction prompt version with its text hash; the most recently promoted one is live |
| `extraction_runs` | one row per extraction attempt: exchange, model, prompt version, mode (`trial` or `live`), outcome, tokens, error |
| `claims` | extracted claims: statement, subject, kind, confidence, `probe_question`, permalink, `supersedes_claim_id`, novelty verdict with probe model/answer/error, retraction |
| `claim_sources` | which messages each claim cites |
| `claims_fts` | full-text index over claim subject and statement (`unicode61`, keeping `-./_` inside tokens) |
| `lore` | the product view: current, live, probed, net-new claims (see "Consuming the database") |
| `exchange_labels` | ground-truth `lore`/`noise` labels per exchange, one row per `(exchange_id, source)`: `llm` (derived from trial runs) or `human` (hand correction), with `source_ref` and `labeled_at`; a human label always wins over an LLM one (see "Triage") |
| `triage_model` | one row per trained Bayes classifier: `version` (autoincrementing primary key), `trained_at`, `labels_used`, `holdout_size`, and `params_json` (the Robinson/Fisher hyperparameters plus the trained `lore_documents`/`noise_documents` totals needed to reconstruct the model) (see "Triage") |
| `triage_tokens` | that model version's per-token counts: `(model_version, token)` primary key, `lore_count`, `noise_count` (see "Triage") |

## Configuration

Settings are read from the process environment by `infovore.config.load_settings`. `infovore.config.settings_from_environment(environ, dotenv_path)` first loads an optional `.env` file (`infovore.config.read_dotenv`: `KEY=VALUE` lines, blank lines and lines starting with `#` are ignored, surrounding single or double quotes on the value are stripped, a missing file yields no values) and then overlays the real environment on top of it, so real environment variables always win over the `.env` file. Startup fails loudly: every problem (missing required value, a value that fails to parse, an empty channel list, a non-positive number) is collected and raised together in one `ConfigError`, so all problems are visible at once instead of one at a time. The discord token is never included in any error message or in `repr()`/`str()` of the settings object.

`INFOVORE_SOURCE` (`discord` default, or `export`) picks which `DiscordSource` implementation backs every command, and shifts which of the settings below are required, per the "Default" column. **`export` is the recommended source for the historic backfill**: fetch from Discord's API exactly once, with DiscordChatExporter (see "Running" → "Runbook", step 1), and every `infovore` command afterwards reads only that export's local JSON files and the SQLite database — re-runnable as often as needed with zero further Discord API traffic. `discord` (talking to Discord live via `discord.py`) is only needed for `infovore run`, which is optional.

| Variable | Default | Meaning |
| --- | --- | --- |
| `INFOVORE_SOURCE` | `discord` | `discord` or `export`. `discord` talks to Discord live (`infovore backfill`, `sync-optouts`, `infovore run`); `export` reads a DiscordChatExporter JSON export from `INFOVORE_EXPORT_DIR` and never touches Discord's API. |
| `INFOVORE_EXPORT_DIR` | *(required when `INFOVORE_SOURCE=export`)* | Root directory of a DiscordChatExporter JSON export, searched recursively for `*.json` files (`infovore.source.export.ExportDiscordSource`). |
| `INFOVORE_DISCORD_TOKEN` | *(required when `INFOVORE_SOURCE=discord`)* | Discord bot token. Never logged or included in error messages. Not needed with `INFOVORE_SOURCE=export` — DiscordChatExporter uses its own token, once, outside of infovore. |
| `INFOVORE_GUILD_ID` | *(required when `INFOVORE_SOURCE=discord`; optional with `export`)* | Discord guild (server) id to operate in. Must be a positive integer. With `INFOVORE_SOURCE=export` and unset, it is inferred from the export at source-open time (`infovore.config.resolve_guild_id`); a `ConfigError` if the export holds more than one guild and none is configured. |
| `INFOVORE_CHANNEL_IDS` | *(optional)* | Comma-separated list of allowlisted channel ids; every entry must be a positive integer. Unset or blank means every text channel and thread the source can see — every channel in the export with `INFOVORE_SOURCE=export`, or every channel the bot can read with `INFOVORE_SOURCE=discord`. `infovore.ingest.allowlist.is_channel_allowed` is the one predicate both `backfill` and live ingest (`infovore run`) apply: a channel is allowed if the list is empty, its own id is listed, or it is a thread whose parent channel id is listed (the parent comes from the `channels` table, filled in by `backfill` or a live `ThreadCreated` event; a message arriving in a thread infovore hasn't seen yet has its parent resolved through the open `DiscordSource` and cached, or is otherwise ignored and counted rather than stored). |
| `INFOVORE_DB_PATH` | *(required)* | Filesystem path to the SQLite database file. |
| `INFOVORE_SCRATCH_DIR` | `scratch` | Working directory for backend scratch files (e.g. an empty cwd for `claude_cli` subprocesses). |
| `INFOVORE_QUIET_GAP_MINUTES` | `30` | Minutes of silence in a channel before the quiet-gap grouping rule closes an exchange. Must be a positive integer. |
| `INFOVORE_BATCH_SIZE` | `10` | Number of pending exchanges processed per extraction/probe batch. Must be a positive integer. |
| `INFOVORE_MAX_RETRIES` | `3` | Maximum retry count for a failed exchange before it stops being retried automatically. Must be a positive integer. |
| `INFOVORE_EXCHANGE_MAX_MESSAGES` | `50` | Maximum messages in one exchange before it is split (Lifecycle rule 6). Must be a positive integer. |
| `INFOVORE_OPT_OUT_ROLE` | `no-archive` | Name of the Discord role that opts a member's messages out of extraction. |
| `INFOVORE_INCLUDE_BOT_MESSAGES` | `false` | Whether bot-authored messages are ingested. Accepts `1`/`true`/`yes`/`on` and `0`/`false`/`no`/`off` (case-insensitive). |
| `INFOVORE_TRIAGE_MIN_SCORE` | `0.3` | Threshold (0..1) the rule `triage_score` must meet or exceed for live `extract` to claim an exchange — applied only when that exchange has no `p_lore` yet (cold start, before a classifier is trained); also the threshold `infovore triage --report` and `status` compare against. Must be a number between 0 and 1 inclusive. |
| `INFOVORE_TRIAGE_MIN_P_LORE` | `0.5` | Threshold (0..1) the trained classifier's `p_lore` must meet or exceed for live `extract` to claim an exchange, once that exchange has been scored by a trained model — see "Triage". Must be a number between 0 and 1 inclusive. |

Each stage — `extract`, `probe`, `judge` — has its own backend selection, all under an `INFOVORE_<STAGE>_*` prefix (`<STAGE>` is `EXTRACT`, `PROBE`, or `JUDGE`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `INFOVORE_<STAGE>_BACKEND` | `claude_cli` for every stage | Backend name for that stage, looked up in the `llm.registry.Registry` (e.g. `claude_cli`, `openai_compat`, `fake`). |
| `INFOVORE_<STAGE>_MODEL` | `sonnet` (`extract`, `probe`), `haiku` (`judge`) | Model alias or id passed to the stage's backend. |
| `INFOVORE_<STAGE>_CONCURRENCY` | `2` | Concurrent in-flight requests allowed for that stage. Must be a positive integer. |
| `INFOVORE_<STAGE>_TIMEOUT` | `60` | Per-request timeout in seconds for that stage. Must be a positive number. |
| `INFOVORE_<STAGE>_<KEY>` | *(none)* | Any other `INFOVORE_<STAGE>_*` variable is passed through to that stage's `StageSettings.options` under its lowercased key (e.g. `INFOVORE_EXTRACT_BINARY_PATH` becomes `options["binary_path"]`), for backend-specific settings such as `claude_cli`'s binary path or `openai_compat`'s base URL and key. |

`infovore.llm.registry.Registry` maps each stage's configured backend name to a `BackendFactory` (`name`, `validate(StageSettings) -> list[str]`, `build(StageSettings) -> LLMBackend`); `default_registry()` registers the `fake`, `claude_cli`, and `openai_compat` factories. `Registry.validate(settings)` reports an unknown backend name per stage plus anything the matching factory's own `validate` rejects; `Registry.build_backends(settings)` raises `ConfigError` if validation fails, otherwise returns one backend per stage; `Registry.health_check(backends)` sends one trivial request per backend and reports `None` on success or the error message on failure, without ever raising. `extract`, `probe`, and `run` build and health-check their stage backends through `infovore.cli.stage_backend`, which prints a flushed `checking <stage> backend (<backend> / <model>)...` line before each stage's health check — visible feedback while a slow backend (e.g. `claude_cli` spawning `claude -p`) is still starting up, before any of that command's own streamed progress lines.

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
uv run infovore triage [--report]
uv run infovore triage --train
uv run infovore triage --recommend-threshold [--min-recall R]
uv run infovore extract [--mode trial|live] [--sample N] [--seed S] [--exchange-id ID ...]
                        [--min-score F] [--max-score F]
uv run infovore probe [--run-id ID ...] [--limit N] [--probe-model CANONICAL_ID] [--retry-failed]
uv run infovore review --run-ids ID [ID ...] [--out PATH]
uv run infovore promote --prompt-version V
uv run infovore label --from-runs RUN_ID [RUN_ID ...]
uv run infovore label --exchange-id ID --lore|--noise
uv run infovore snapshot <dest> [--force]
uv run infovore run [--interval SECONDS] [--once]
```

`chunk` groups ingested messages into exchanges and persists the closed ones; `--now` overrides the clock, which is useful when iterating over an old backfill. While it runs, `chunk` streams progress, one flushed line at a time: `grouping: N channels with ungrouped messages`, then one `channel <id>: +E exchanges, D deferred` line per channel that has ungrouped messages, before the existing summary.

`triage` (`infovore.triage.runner.triage_pending`, `infovore.triage.command.TriageCommand`) scores every exchange whose `triage_version` differs from the current `infovore.triage.score.TRIAGE_VERSION` — idempotent, so a rerun with no rule change scores nothing new, and bumping `TRIAGE_VERSION` re-scores the whole table. After scoring, it computes each channel's mean *raw* rule score (the sum of that exchange's signal weights, clamped to 0..1, before any channel adjustment) and the database-wide raw mean, then applies a bounded prior: `delta = clamp(WEIGHT * (channel_mean - global_mean), -CAP, CAP)` and `adjusted = clamp(raw + delta, 0, 1)`, with `WEIGHT = 0.3` and `CAP = 0.1` (`infovore.triage.runner.CHANNEL_PRIOR_WEIGHT`/`CHANNEL_PRIOR_CAP`) — a chatty channel whose average is below the database mean gets pulled down by at most 0.1, and one running consistently hot lore gets pulled up by at most 0.1, without hand-tuning per channel. `triage_score` is overwritten with `adjusted`; the delta itself is appended to `triage_reasons` as a `channel_prior` entry, and the per-channel deltas are returned in the report. While it runs, `triage` streams progress, one flushed line at a time: a start line with the candidate count (`triage: N exchanges to score`), one `exchange <id>: score=<raw> (k/N)` line per exchange scored, and a closing `channel priors applied: C channels` line, before the `scored=`/`channels_adjusted=` summary. `--report` additionally prints a score histogram in 0.1-wide buckets, each channel's adjusted-score mean and exchange count, counts of exchanges at/above and below `INFOVORE_TRIAGE_MIN_SCORE`, and the 10 most frequent triage reasons database-wide. After rule scoring and channel priors, plain `infovore triage` also loads the latest trained classifier, if any (`infovore.triage.train.load_latest_model`), and rescopes `p_lore`/`p_lore_model` (`infovore.triage.train.score_stale`) only for exchanges whose `p_lore_model` isn't that model's version yet — idempotent the same way rule scoring is, and cheap on a rerun with no new model.

`infovore triage --train` (`infovore.triage.train.train_and_store`) builds features for every exchange with an effective label (`infovore.db.labels.effective_labels`), splits them by the same deterministic holdout as `infovore.triage.bayes.in_holdout` (a hash of the exchange id, about a fifth of labels), trains the Robinson/Fisher naive Bayes classifier on the non-holdout examples, and refuses with a clear message (exit `2`) if either class has fewer than 10 labels. It prints the trained model's version, labels used and holdout size; a precision/recall/F1 table evaluated on the holdout at thresholds `0.1` through `0.9`; the confusion matrix at the configured `INFOVORE_TRIAGE_MIN_P_LORE`; and the 15 most informative tokens (largest `|p - 0.5|`) with their lore/noise counts. The trained model (`triage_model`/`triage_tokens`) is stored before any of this prints, and every exchange is then scored (`infovore.triage.train.score_all`), writing `p_lore`/`p_lore_model` database-wide — training is a full, idempotent rebuild, never incremental.

`infovore triage --recommend-threshold [--min-recall R]` (default `R = 0.9`) re-evaluates the latest stored model over its own holdout split and reports the highest `p_lore` threshold that keeps recall at or above `R` (`infovore.triage.bayes.recommend_threshold`), printed as a ready-to-set `INFOVORE_TRIAGE_MIN_P_LORE=<threshold>` line together with that threshold's recall and precision, and the expected share of all `p_lore`-scored exchanges that would be sent to the LLM at that threshold. No threshold reaching the target recall prints that plainly instead of a number. Without a trained model, it exits `2` ("run `infovore triage --train` first").

`probe` runs the closed-book novelty probe (`infovore.extract.novelty.run_probe`) over claims that still need it, built from the `probe` and `judge` stage backends (`infovore.extract.llm_extractor.LLMNoveltyProbe`), with the `probe` stage's configured concurrency. `--run-id` (repeatable) scopes the run to the non-retracted claims of those extraction runs only, instead of the whole database. `--limit` (default `INFOVORE_BATCH_SIZE`) is the number of candidates fetched per batch; the command loops fetching batches until a fetch turns up nothing left to probe. `--probe-model` names the canonical model id (e.g. `claude-sonnet-5`) the operator expects the configured probe backend to resolve to; passing it after a probe model upgrade re-probes exactly the claims that model hasn't seen (`db.claims.claims_needing_probe` / `claims_for_runs_needing_probe`), since probing is idempotent per `(claim, probe_model)`. Without `--probe-model`, only claims with novelty `unprobed` are considered, because the canonical model id is only known from a call's own result, not in advance. A claim whose `probe_error` is set (a `transient`/`fatal`/`invalid_output` failure from a previous probe attempt) is parked: it is skipped by every future `probe` run, on this or any other invocation, until `--retry-failed` is passed, which includes parked claims in the candidate set again for this run only. While it runs, `probe` streams progress, one flushed line at a time: `probe: N candidates` for the first batch fetched, then one line per claim — `claim <id>: <verdict>`, `claim <id>: failed`, or `claim <id>: paused` — before the existing summary. `probe` prints `probed`, `by verdict` (counts per `Novelty`), `failed`, and `pauses`, and exits `1` if any claim failed (`0` otherwise, even if some claims paused on a usage limit and later succeeded, or were parked and skipped).

`review` (`infovore.extract.review.build_review`/`render_review_html`, `ReviewCommand`) renders a prompt-iteration report over one or two prompt-version "run sets": the run ids in `--run-ids` (repeatable, at least one; trial mode creates one run per exchange, so this is typically every run id `extract --mode trial` printed) are grouped by their `prompt_version`, and every exchange covered by any of them is rendered with its messages, and, per version present for that exchange, that version's claims (kind, subject, statement, confidence, cited message ids, novelty and probe answer), run outcome/error, and tokens. With exactly two versions covering the same exchange, a per-exchange diff shows claims added, dropped, and changed (claims are matched by `(subject, statement)` normalized; a "changed" match is the same subject with a different statement or kind, found among claims left over after the exact match pass) plus verdict shifts on matched claims. A summary header gives, per version: exchanges, claims per exchange, verdict distribution, share of `known`, failed runs, and tokens per 100 exchanges (runs don't record a dollar cost, so token counts are the number to watch). The report is one self-contained HTML file — inline CSS only, no external requests, readable in light and dark, collapsible per-exchange `<details>` sections — written to `--out` (default `review.html` under `INFOVORE_SCRATCH_DIR`, parent directories created as needed) and the path is printed. An unknown run id exits `2`.

`promote` (`infovore.db.claims.promote_prompt_version`, `PromoteCommand`) records `--prompt-version V` as the live prompt version, so the next `extract` without `--mode trial` uses it. If `V` equals the current `infovore.extract.prompt.PROMPT_VERSION`, it is registered first (same as `extract` would); any other version must already be registered (by a prior `extract --mode trial` run, which registers the prompt version it ran with) or the command exits `2` with a clear message. It prints the resulting live prompt version.

`label` (`infovore.triage.label.LabelCommand`, `infovore.db.labels`) records ground-truth `lore`/`noise` labels on `exchange_labels` (see "Data model" and "Triage"). `--from-runs RUN_ID [RUN_ID ...]` derives an `llm`-sourced label per run (`infovore.triage.label.derive_labels_from_runs`), one `run {id}: {outcome}` progress line per run id (flushed immediately) followed by a `labeled: lore=N noise=N skipped=N` summary line (with the skipped run ids and their reasons in parentheses when any were skipped): a run whose outcome is `failed` is skipped ("failed run"); otherwise, a run with at least one claim probed `unknown`, `partial`, or `contradicts` is labeled `lore`; a run whose claims are all `known` (including a run with zero claims) is labeled `noise`; a run with neither — i.e. it still has an `unprobed` claim and no `unknown`/`partial`/`contradicts` claim — is skipped ("unprobed claims"). An unknown run id exits `2`. `--exchange-id ID --lore` or `--exchange-id ID --noise` instead records a `human`-sourced label directly, for hand corrections; `--lore` and `--noise` are mutually exclusive, `--exchange-id` requires exactly one of them, and `--from-runs`/`--exchange-id` are themselves mutually exclusive — every missing or conflicting combination exits `2`. Since `exchange_labels` has one row per `(exchange_id, source)`, re-labelling the same exchange from the same source (a later trial run, or a corrected hand label) replaces the earlier row rather than accumulating duplicates, and a human label always wins over an LLM one for the same exchange regardless of insert order.

`snapshot` writes a consistent copy of the product database to `<dest>` using the SQLite backup API (`infovore.db.snapshot.snapshot`), safe to run at any time, including while `backfill`/`chunk`/`extract`/`run` is mid-write against the same file: the backup only ever sees committed data, never a writer's in-flight transaction. It refuses to overwrite an existing `<dest>` unless `--force` is given, creates `<dest>`'s parent directories as needed, and writes through a temporary file in the same directory that it atomically renames into place, so a reader never observes a partially written snapshot. It reports the destination path, its size in bytes, and its `PRAGMA user_version` (the schema version). This is how the product database leaves a host — see "Deployment" below.

Every subcommand loads configuration (environment, then `.env` in the working directory for anything not set), opens and migrates the database, and runs. `status` prints row counts, the exchange queue by status, claims by novelty, run outcomes, the last extraction and probe times, the live prompt version, the count of exchanges triaged at the current `TRIAGE_VERSION` and how many of those are at or above `INFOVORE_TRIAGE_MIN_SCORE` (`triaged: N (above threshold M)`), and each stage's backend and model.

`backfill` walks every channel in `INFOVORE_CHANNEL_IDS` (or, if it's unset, every channel the source has), plus their threads (including archived ones), oldest message first, from `infovore.ingest.backfill.backfill`. Each channel's checkpoint (`channels.last_backfilled_message_id`) is stored after every page of messages, in the same transaction as that page's rows, so an interrupted run resumes exactly where it left off and never re-walks or loses data. A page that fails with a rate limit sleeps for the retry-after duration and retries from the checkpoint; a page that fails because the source is unavailable backs off exponentially (1s, 2s, 4s, ...); a channel that fails five consecutive times (configurable via the `backfill()` function's `max_attempts`) is recorded as failed and the walk continues with the remaining channels. While it runs, `backfill` streams progress, one flushed line at a time: `opening discord source...` before connecting, `found N channels (M selected)`, `channel <id> <name>: start` (with `resuming after message <id>` when a checkpoint exists), one `page +new, updated, unchanged (total so far)` line per saved page, and `done` or `failed: <reason>` per channel. At the end it prints one summary line per channel (pages walked, messages inserted/updated/unchanged, messages skipped as system or bot) and one line per failed channel, then exits `0` if every channel completed or `1` if any channel failed. `--page-size` (default `100`) controls how many messages are requested per history page. `backfill` and `sync-optouts` log in to Discord with `INFOVORE_DISCORD_TOKEN`, wait up to 60 seconds for the gateway to be ready, and always close the connection when they finish. A bad token, missing privileged intents, or a timeout exits `3` without printing the token. `sync-optouts` prints the same immediate `opening <source> source...` line as `backfill` before connecting, then its usual `added=`/`removed=`/`redacted_messages=`/`retracted_claims=` summary.

`extract` (`infovore.extract.runner.run_extraction`, `infovore.extract.command.ExtractCommand`) turns pending and stale exchanges into claims with the extract stage's configured backend (`LLMClaimExtractor`), model, and concurrency. `--mode` defaults to `live`: it registers the current prompt version (`infovore.extract.prompt.PROMPT_VERSION`/`PROMPT_SHA256`) and refuses to run unless that exact version is the live, promoted one (exit `2`, "run `infovore promote --prompt-version vN`"). Live mode is gated on triage: before claiming anything, it refuses (exit `2`, "run `infovore triage` first") if any exchange still queued for extraction (`pending` or `stale`, under `INFOVORE_MAX_RETRIES`) has a `triage_version` that is `NULL` or not the current one; once every queued exchange is triaged, it repeatedly pulls up to `INFOVORE_BATCH_SIZE` claimable exchanges that pass the gate (below that, an exchange simply stays `pending` — it is lore-negative, not an error) until none remain. The gate is one function, `infovore.triage.gate.passes_gate(exchange, min_score, min_p_lore)` (mirrored in SQL by `gate_sql(min_score, min_p_lore)`, which `claimable_exchanges` uses so the live queue and a trial `--strategy uncertain` sample see exactly the same exchanges a live run would): an exchange with a `p_lore` (scored by a trained classifier) passes when `p_lore >= INFOVORE_TRIAGE_MIN_P_LORE`; one with no `p_lore` yet (cold start, or a classifier hasn't scored it) falls back to the rule `triage_score >= INFOVORE_TRIAGE_MIN_SCORE`. See "Triage" for how `p_lore` gets populated. In either mode, an exchange whose every message is from an opted-out author is skipped without calling the extractor (it would only ever see `[redacted]`) and counted in `report.skipped`; only in live mode is the exchange's status also written to `skipped` — trial mode leaves status and retry count untouched, same as any other trial-mode exchange. On success, claims are recorded under the live run and the exchange is marked `done` (zero claims is still `done`); a `stale` exchange's previous live claims are retracted (`reextracted`) in the same step. A `transient`/`fatal`/`invalid_output` failure records a failed run and increments the exchange's retry count, marking it `failed` once `INFOVORE_MAX_RETRIES` is reached; a `usage_limit` result pauses on the configured `Sleeper` for its `retry_after` (default 300s) without touching retry counts, then retries the same exchange. `--mode trial` runs the extractor over an explicit set of exchanges (any status, untriaged or not — trial mode is never gated) picked with `--sample N` (a reproducible sample via `infovore.extract.runner.select_trial_sample`, stratified by channel and exchange size bucket, using `--seed`, default `0`) and/or repeatable `--exchange-id`; `--sample` alone draws from every exchange regardless of triage score, and `--min-score`/`--max-score` (either or both, inclusive) narrow that pool to a score band for calibration — running a sample from just below and just above a candidate `INFOVORE_TRIAGE_MIN_SCORE` and comparing extraction results is how the threshold gets picked. Trial mode never changes exchange status, retry counts, or existing claims, and its claims are recorded under a `trial` run for prompt iteration. Extractor calls are bounded by the extract stage's configured concurrency (`asyncio.Semaphore`). While it runs, `extract` streams progress, one flushed line at a time: a start line naming the mode and how many exchanges are queued (`extract: trial mode, N exchanges queued`) or that it is draining the live queue (`extract: live mode, draining queued exchanges`), then one line per exchange outcome — `exchange <id>: <n> claims`, `skipped`, `failed: <kind>`, or `paused <s>s (usage limit)` — each suffixed with a running counter, `(k/N)` in trial mode or `(k done)` in live mode, before the existing summary. `extract` prints processed/succeeded/failed/skipped/claims_recorded/pauses counts and the recorded run ids, and exits `1` if any exchange failed.

`run` (`infovore.run.run_forever`/`run_once`, `infovore.run.RunCommand`) is the long-running mode: live ingest concurrently with a periodic chunk → triage → extract → probe cycle, sharing the extraction runner's pause semantics. It opens the configured `DiscordSource` once (`context.source_factory`, closed on shutdown) and builds the `extract`/`probe`/`judge` stage backends up front, same as `extract` and `probe`. One task consumes `source.events()` via `infovore.ingest.live.handle_event`, applying the same `INFOVORE_CHANNEL_IDS` allowlist as `backfill` (see "Configuration"): an event for a channel or thread outside the allowlist is ignored and counted in `events_ignored` rather than stored. A failing event is logged and counted in `events_failed`, and the loop moves on to the next event. Concurrently, a cycle runs immediately and then every `--interval` seconds (default `600`): `sync_opt_outs` → `chunk.grouper.group_pending` → `infovore.triage.runner.triage_pending` → `extract` in `live` mode (gated on `INFOVORE_TRIAGE_MIN_SCORE`, same as a standalone `extract`) → `probe`. Running `triage_pending` every cycle keeps it caught up automatically, so `run`'s live extraction never hits the "untriaged" refusal. If the live prompt version is not the promoted one, extraction is skipped for that cycle only (a warning is logged) while opt-out sync, grouping, triage, and probing still run. An unexpected exception from any cycle step is logged and counted in `cycles_failed`; the loop waits out the interval and tries again. `--once` runs exactly one cycle with no live consumption — for cron, launchd, or systemd timers instead of a long-running process. On `SIGTERM` or `SIGINT`, the running cycle finishes its current step (SQLite writes are synchronous, so there is no partial write to abandon), the live-event task is cancelled at its next await point, the source's context manager exits, and `run` prints `events_handled`/`events_failed`/`events_ignored`/`cycles_completed`/`cycles_failed` and exits `0`. `run` streams progress too: `opening <source> source...` before connecting, then one flushed `cycle: <step>` line (`sync-optouts`, `chunk`, `triage`, `extract`, `probe`, in that order) at the start of each step of every cycle.

Exit codes: `0` ok, `1` unexpected failure, `2` configuration or usage error, `3` Discord or an LLM backend is unavailable.

### Runbook

The guiding principle: **fetch from the Discord API once, then do everything else as a secondary step.** Step 1 below (exporting with DiscordChatExporter) is the only step that touches Discord's API, and it is done by DCE, not infovore. With `INFOVORE_SOURCE=export`, every infovore command — `infovore sync-optouts`, `infovore backfill`, `infovore chunk`, `infovore extract --mode trial`, `infovore probe`, `infovore review`, `infovore promote`, a real `infovore extract`/`infovore probe`, and `infovore snapshot` — reads only the export's local JSON files and the SQLite database. None of them make Discord API calls, so the whole pipeline is re-runnable as often as needed with zero further Discord API traffic; re-export and re-run `infovore backfill` whenever the channel history has moved on. `infovore run` (live ingest, step 9) is the only thing that talks to Discord at all, and it is optional.

**1. Export the server with DiscordChatExporter.** Download [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) (Tyrrrz), create a Discord bot application for **DCE's own use** (not infovore's — infovore's `discord.py` client, set up in steps 2–3, is only needed for the optional `infovore run`), and on that bot's application enable the same two privileged intents DCE needs to read message bodies and resolve member/role info: **Message Content** and **Server Members**. Invite it read-only (**View Channels**, **Read Message History**), then export:

```
DiscordChatExporter.Cli exportguild -t <bot-token> -g <guild-id> -f Json --include-threads all -o export/
```

`-f Json` is the JSON export format `infovore.source.export.ExportDiscordSource` reads; `--include-threads all` is required — a lot of the lore lives in threads. Set `INFOVORE_SOURCE=export` and `INFOVORE_EXPORT_DIR=export/` (see "Configuration"); `INFOVORE_DISCORD_TOKEN` is not needed by infovore in this mode.

**2. Create infovore's own bot** (only if you plan to use `infovore run` for live ingest — step 9; skip to step 4 otherwise). In the Discord Developer Portal, create an application and add a bot. On the Bot tab, enable the two privileged intents **Message Content** (message bodies) and **Server Members** (role membership, for opt-out sync); without both, the gateway rejects the connection. Copy the bot token.

**3. Invite it read-only.** Invite the bot with the `bot` scope and only the **View Channels** and **Read Message History** permissions on the channels you want archived. It never sends messages.

**4. Create the opt-out role and post the notice.** Create a role named `no-archive` (or set `INFOVORE_OPT_OUT_ROLE`), make it self-assignable (e.g. through your roles bot or onboarding), and post this Server notice in an announcements channel before the first backfill:

> **Server notice — channel archiving.** A read-only bot is archiving the technical history of #channel-a and #channel-b so that hard-won SGI/IRIX knowledge (part numbers, jumper settings, PROM versions, fixes, procedures) isn't lost. It reads messages, never posts, and keeps specific technical facts with a link back to the original message. If you don't want your messages archived, give yourself the `no-archive` role: your past and future messages will be redacted in the archive (content and name replaced with `[redacted]`), and facts that came only from you will be removed. Removing the role later only affects future messages; redacted history stays redacted.

With `INFOVORE_SOURCE=export`, `infovore sync-optouts` derives role membership from `ExportDiscordSource.role_member_ids`, i.e. from the roles recorded on authors in the export — so only members who have posted somewhere in the exported history can be recognized as opted out; a member who never posted has nothing to redact regardless.

**5. Configure.** With the export (recommended): set at least `INFOVORE_SOURCE=export`, `INFOVORE_EXPORT_DIR` and `INFOVORE_DB_PATH` (see Configuration) — `INFOVORE_GUILD_ID` and `INFOVORE_CHANNEL_IDS` are optional and default to "every guild/channel the export holds" (one guild only). With live Discord: set at least `INFOVORE_DISCORD_TOKEN`, `INFOVORE_GUILD_ID` and `INFOVORE_DB_PATH` — `INFOVORE_CHANNEL_IDS` is optional here too and defaults to every channel and thread the bot can see. Either way, make sure the extract/probe/judge backends work: `infovore status` shows each stage's backend and model, and every LLM command health-checks its backend first (exit `3` if it's unavailable).

**6. First backfill.** Always sync opt-outs first so nothing from an opted-out user is ever stored unredacted:

```
infovore sync-optouts
infovore backfill
infovore chunk
infovore triage
infovore status
```

`backfill` is resumable; if it's interrupted, run it again. With the export source, re-running it after re-exporting only ingests what's new (per-channel checkpoints), and a re-run over unchanged data changes nothing. The workflow order is always **backfill → chunk → triage → extract**: `triage` must run (or already be current) over every queued exchange before `extract` in live mode will claim any of them.

**7. Settle the prompt** with the trial loop (next section): `infovore extract --mode trial`, `infovore probe`, `infovore review`, then `infovore promote` once a prompt version looks right. Trial mode is never gated on triage, so this loop works even before the first `infovore triage` run.

**8. Extract for real:**

```
infovore triage
infovore extract
infovore probe
```

**9. Steady state (optional).** Everything above is re-runnable by hand with the export source and needs no live connection. If you also want live ingest, either keep one process running, which follows live messages and runs a sync → chunk → triage → extract → probe cycle every `--interval` seconds (this requires `INFOVORE_SOURCE=discord`):

```
infovore run --interval 600
```

or schedule `infovore run --once` from cron, launchd, or a systemd timer (see Deployment). Stop `run` with SIGTERM or Ctrl-C; it finishes the current step and closes the Discord connection.

**10. Ship the product.** `infovore snapshot /path/to/lore-2026-10-01.db` writes a consistent copy of the database, safe while `run` is writing; consumers read its `lore` view (see "Consuming the database").

**When things go wrong:** exit `2` is configuration (the message names every problem), exit `3` is Discord or an LLM backend being unavailable (bad token, missing intents, `claude` not logged in), exit `1` means some exchanges or claims failed and were counted — `infovore status` shows the queues, failed exchanges retry until `INFOVORE_MAX_RETRIES`, and parked probe failures come back with `infovore probe --retry-failed`.

### Iterating the prompt

This is the prompt-iteration loop: run the current prompt over a reproducible sample of the backfill, look at what it extracted, and either promote it or change the prompt and compare. Every step is `--mode trial`, so it never touches `live` claims or exchange status.

1. `infovore backfill` then `infovore chunk`, once, to populate exchanges from the real history.
2. `infovore extract --mode trial --sample 50 --seed 1` — runs `infovore.extract.prompt.PROMPT_VERSION` over 50 exchanges, stratified by channel and size, chosen deterministically by `--seed`. Note the run ids it prints.
3. `infovore probe --run-id <the run ids from step 2>` — closed-book novelty probe over the trial's claims.
4. `infovore review --run-ids <the same run ids>` — writes `review.html`; open it and read the claims, verdicts, and summary numbers (claims per exchange, share of `known`, tokens per 100 exchanges).
5. Edit the prompt in `infovore/extract/prompt.py` and bump `PROMPT_VERSION` (this changes `PROMPT_SHA256` too, so the new version is distinguishable from the old one in the database).
6. Repeat step 2 with the same `--seed` (`infovore extract --mode trial --sample 50 --seed 1`) so the new version runs over the same exchanges, then step 3 against the new run ids.
7. `infovore review --run-ids <old run ids from step 2> <new run ids from step 6>` — with two prompt versions covering the same exchanges, the report adds a side-by-side diff per exchange (added/dropped/changed claims, verdict shifts) so a wording change's effect is visible exchange by exchange, not just in the summary.
8. Once a version looks right, `infovore promote --prompt-version vN` makes it live.
9. `infovore extract` (no `--mode`, so it defaults to `live`) now runs the promoted version over the real pending/stale queue.

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

## Triage

Most Discord chatter carries no lore, so exchanges are scored with deterministic, programmatic signals before any LLM sees them; only exchanges that score high enough are sent to extraction. Scoring is free, re-runs in seconds over the whole database, and is versioned (`TRIAGE_VERSION` = `t1`), so changing the rules simply re-scores everything. Scores live on `exchanges.triage_score` (0–1, clamped sum of the signals below), with the contributing signals in `triage_reasons` (JSON) and the rule version in `triage_version`.

| signal | weight | fires when |
| --- | --- | --- |
| `domain_terms` | +0.15 per distinct term, max +0.45 | SGI/IRIX vocabulary: model and board names (Indy, Indigo2, O2, Octane, Fuel, Tezro, Onyx, Origin, IPxx), CPUs (R10000, R12k…), tools and subsystems (hinv, inst, swmgr, nvram, PROM, XFS, XLV, MIPSpro, sash, GIO/XIO, VPro, Odyssey, Impact…) |
| `irix_version` | +0.2 | IRIX-style versions such as `6.5.22`, `6.5.30m`, `IRIX 5.3` |
| `part_number` | +0.3 | SGI part numbers such as `030-1234-001` |
| `unix_path` | +0.15 | paths under `/usr`, `/var`, `/etc`, `/opt`, `/dev`, `/stand`, `/hw`… |
| `code` | +0.15 | code blocks or inline backticks |
| `archive_link` | +0.15 | links to FTP, archive.org, bitsavers, techpubs, or SGI/IRIX sites |
| `pdf_attachment` | +0.15 | a PDF attachment (manuals, datasheets) |
| `answered_question` | +0.2 | a message with a `?` followed by a reply of 40+ characters from a different author |
| `agreed_answer` | +0.05 | a ✅/👍/☑️/✔️/💯 reaction on a message after the first |
| `thread` | +0.05 | the exchange is in a thread |
| `substantial` | +0.1 | 400+ characters of text in total |
| `mostly_tiny_messages` | −0.2 | more than 70% of messages are under 20 characters |
| `gif_links` | −0.1 | tenor, giphy, or `.gif` links |
| `laughter` | −0.1 | more than 30% of messages are just "lol", "lmao", "haha"… |

Channel priors and the extraction threshold are applied by `infovore triage` and `extract` (see the triage issues); the threshold is chosen by calibrating against LLM extraction on samples above and below it.

The rule score above is a cold-start heuristic. Ground truth accumulates in `exchange_labels` (see "Data model") and never shrinks: `infovore label --from-runs` derives `lore`/`noise` labels from trial extraction runs, and `infovore label --exchange-id ID --lore|--noise` records hand corrections, which always win over a derived label for the same exchange (see "Running"). Those labels are the training data for the Bayesian classifier that replaces the rule score once enough of them exist.

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

`infovore.extract.novelty.run_probe(conn, probe, clock, sleeper, *, probe_model, limit, concurrency, run_ids=None, retry_failed=False)` drives a `NoveltyProbe` (`LLMNoveltyProbe` or the fake `MarkerProbe`) over the database. Candidates come from one of three `db.claims` queries, in this precedence, each taking an `include_failed` flag set from `retry_failed`:

1. `run_ids` given: `db.claims.claims_for_runs_needing_probe(conn, run_ids, probe_model, limit, include_failed)` — the non-retracted claims of exactly those extraction runs, that are `unprobed` or (when `probe_model` is given) whose recorded `probe_model` differs from it.
2. `run_ids` is `None` and `probe_model` is given: `db.claims.claims_needing_probe(conn, probe_model, limit, include_failed)` — every non-retracted claim database-wide that is `unprobed` or whose `probe_model` differs.
3. Neither: `db.claims.unprobed_claims(conn, limit, include_failed=include_failed)` — every non-retracted claim with novelty `unprobed`, since without a target model there is nothing to compare a claim's existing `probe_model` against.

Each of these three queries additionally requires `probe_error IS NULL` unless `include_failed` is true. This means a claim that previously failed with a non-`USAGE_LIMIT` error is **parked**: once `db.claims.set_probe_error` has recorded a `probe_error` for it, no candidate query returns it again — in this run or any future one — until something clears `probe_error` (a successful `set_novelty` call, which always clears it) or the caller explicitly asks for parked claims back with `include_failed=True` (`retry_failed=True` from `run_probe`, `--retry-failed` from the CLI). This keeps a handful of permanently failing claims (e.g. persistent model refusals) from occupying the front of the `id`-ordered candidate window and starving every claim behind them.

Each candidate is probed at most once per fetched batch, with an `asyncio.Semaphore(concurrency)` bounding how many `probe.probe()` calls are in flight at once. A successful outcome calls `db.claims.set_novelty(conn, claim.id, outcome.verdict, outcome.model, outcome.answer, clock.now())`, which also clears `probe_error`. A `TRANSIENT`, `FATAL`, or `INVALID_OUTPUT` failure calls `db.claims.set_probe_error(conn, claim.id, failure.message)` (parking the claim as above), counts it in `failed`, and — for this run only — is not retried again even if `retry_failed=True` keeps returning it as a candidate; an in-memory "gave up this run" set filters it out of later batches so a claim that fails on every attempt cannot loop forever within a single `--retry-failed` run. A `USAGE_LIMIT` failure counts in `pauses`, calls `sleeper.sleep(failure.retry_after or 300)`, and retries the same claim — indefinitely, if the backend keeps returning `USAGE_LIMIT` — rather than pausing other claims already in flight; it never sets `probe_error` and so never parks the claim.

`run_probe` fetches a batch of up to `limit` candidates, processes it fully (concurrently, bounded as above), then fetches again; it stops once a fetch returns nothing left to process (after filtering out claims already given up on this run). Because parking is a database-level exclusion rather than an in-memory one, this loop reaches every real (non-parked) candidate in one run regardless of how many claims are parked, without needing a paged/offset query.

`run_probe` returns a `ProbeReport(probed, by_verdict, failed, pauses)`: `probed` is the count of claims that got a verdict this run; `by_verdict` maps `Novelty` (`unknown`/`partial`/`contradicts`/`known`) to how many of those probed claims got that verdict; `failed` is the count newly parked (or re-parked, under `--retry-failed`) this run; `pauses` is the number of `USAGE_LIMIT` sleeps taken (a single claim retried three times before succeeding counts three pauses).

## Privacy and opt-out

A Discord role (`INFOVORE_OPT_OUT_ROLE`, default `no-archive`) lets a guild member opt their messages out of extraction. `infovore sync-optouts` fetches the role's current members via `DiscordSource.role_member_ids` and reconciles them against the `opt_outs` table (`infovore.privacy.optout.sync_opt_outs`): a member holding the role who is not yet in `opt_outs` is added with `since` set to now; a member in `opt_outs` who no longer holds the role is removed. Removing a user from `opt_outs` only stops future redaction — **opting back in never restores previously redacted history**, since redaction is destructive (the original content is overwritten, not merely hidden).

For each newly added user, every message they already authored is redacted in place: `content` and `author_name_at_time` become `[redacted]`, `raw_json` becomes `{}`, every `message_revisions` row for their messages is redacted the same way, and their `attachments` rows are deleted. `reactions` are left untouched (no author-identifying content). This redaction is a set of plain `UPDATE`/`DELETE` statements in one transaction; it never goes through `upsert_message`, so it never writes a new revision.

Going forward, every message from an opted-out author is redacted *before* it reaches storage: `infovore.privacy.optout.redact_normalized` replaces `content`, `author_name_at_time`, and `raw_json` (attachments dropped, reactions kept) on the `NormalizedMessage` produced by `ingest.normalize`, and this must run before `db.raw.upsert_message` is called. This ordering matters for backfill re-runs: `upsert_message` treats any difference in `content` from the stored row as an edit and writes a revision. If an already-opted-out author's original, unredacted content arrived from Discord again and were upserted directly, it would look like an edit and overwrite the redacted row (with the unredacted text saved as a "prior" revision). Redacting first means the incoming row always matches the already-redacted stored row, so the upsert is a no-op and no revision is ever written — a backfill can be re-run any number of times over an opted-out author's history without ever un-redacting it.

Per lifecycle rule 5: when `sync_opt_outs` adds users, it also calls `db.claims.retract_claims_with_all_sources_opted_out`, which retracts (`retraction_reason = 'sources_opted_out'`) every claim whose *every* source message is authored by an opted-out user. A claim with at least one source from a non-opted-out author is kept; only its opted-out source messages are redacted.

`sync_opt_outs` logs one `logging` info line per added or removed user id (never message content), so the change is auditable without exposing what was said.

**Before any extraction** against real Discord data, run `infovore sync-optouts`; `backfill`, live ingest and `run` redact opted-out authors before storing anything, and prompt rendering (`extract/prompt.py`, `ExtractionRequest.opted_out_user_ids`) redacts them again.

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

or `docker run -d --name infovore --restart unless-stopped -v "$PWD/data:/data" --env-file .env infovore run` keeps the same container alive as the live ingest + periodic pipeline (see "Running" above). `--env-file .env` carries every `INFOVORE_*` setting plus whichever backend credentials apply (see "Headless auth" below); none of it needs to be baked into the image.

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
