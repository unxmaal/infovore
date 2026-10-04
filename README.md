# infovore

## Purpose

infovore is a passive archivist for an SGI, IRIX and retrocomputing Discord community. It ingests channel history and live messages, groups them into conversations called exchanges, and decides which exchanges are worth keeping. Exchanges are the base object: every label, score and archive decision attaches to an exchange, not to a single message.

The decision is made by a zero-token relevance cascade. It runs locally with no LLM calls. The product is a SQLite file of archived exchanges. infovore has no API, no MCP server and no Discord commands, and it never posts.

## Architecture

Pipeline:

1. Ingest. `backfill` walks allowlisted channels and threads oldest-first with a per-channel checkpoint; `run` consumes live events. Messages are normalized, redacted if the author opted out, then upserted. The source is either live Discord or a DiscordChatExporter JSON export (`INFOVORE_SOURCE`).
2. Chunk, recipe v2. Messages are grouped by thread, reply chain and quiet gap, then persisted once closed. Gap, folding of singletons and adaptive gaps are recipe parameters recorded in `chunk_recipes`. Late replies link to the earlier exchange as read-only context.
3. Relevance cascade. Each current exchange passes through, in order:
   1. excluded channels (`INFOVORE_EXCLUDE_CHANNELS`): never archived;
   2. `no_text`: nothing to judge;
   3. lexicon: terms that settle clearly relevant or clearly irrelevant exchanges;
   4. middle stage: bge-small embedding plus logistic head trained on human labels (`relevance_embed`, needs `uv run --extra embed`);
   5. residue: what the earlier stages abstain on.
4. Archive. Exchanges the cascade accepts are archived and exported with `export-archive`.

An optional local-model residue scorer (`relevance llm-score --residue`) scores the residue with an OpenAI-compatible server. Claim extraction (extract, probe, review, promote) still ships but is paused and is not part of the archive path.

Import boundaries are enforced by ruff `banned-api`: `discord` only in `infovore/source/live.py`, process spawning only in `infovore/llm/claude_cli.py`, `openai` and `httpx` only in `infovore/llm/openai_compat.py`.

## Data model

All tables live in one SQLite file (`INFOVORE_DB_PATH`). Migrations run on open.

| Object | Role |
|---|---|
| `channels`, `messages`, `message_revisions`, `attachments`, `reactions` | raw ingest; `messages_fts` and its shadow tables index message text |
| `opt_outs` | opted-out users; their history is redacted |
| `exchanges`, `exchange_messages`, `superseded_exchange_messages`, `exchange_remap` | closed exchanges, membership, and the mapping across rechunks |
| `chunk_recipes`, `channel_chunk_gaps` | chunking recipes and per-channel gap overrides |
| `current_exchanges` | view: exchanges under the active recipe; the base object for everything below |
| `all_exchange_messages` | view: members of every exchange |
| `reviewed_words` | append-only tech or not decisions on words from undecided conversations |
| `annotations`, `current_annotations` | derived and human annotations on exchanges; the view shows the latest per exchange |
| `eval_slices`, `current_eval_slices`, `current_slice_members` | frozen evaluation slices and their current membership |
| `exchange_labels`, `label_events` | human lore/noise labels and their event history |
| `scorer_activations`, `gate_predictions` | which scorer is active and its stored predictions |
| `triage_model`, `triage_tokens` | triage classifier state |
| `message_labels`, `message_model`, `message_tokens`, `message_combiner` | message-level sifting labels and models |
| `claim_runs`, `claim_run_exchanges`, `claims_v2`, `claims_v2_sources`, `claim_rejections` | claim trial: runs, per-conversation outcomes, redacted claims with their cited messages, and claims rejected with the reason; append-only |
| `claim_reviews`, `current_claim_reviews` | append-only human verdicts on trial claims; the view shows the latest per claim |
| `claim_checks`, `current_claim_checks` | append-only deterministic grounding checks of trial claims against their cited messages; the view shows the latest per claim |
| `claims`, `claim_sources`, `claims_fts` | extracted claims and their source messages (paused) |
| `extraction_runs`, `extraction_batches`, `prompt_versions`, `probe_runs` | extraction and probe bookkeeping (paused) |
| `lore` | view: claims judged net-new (paused) |
| `schema_migrations` | applied migrations |

## Configuration

Settings come from the environment, or a `.env` file in the working directory. Missing or malformed required values fail at startup with all errors listed.

| Variable | Default | Meaning |
|---|---|---|
| `INFOVORE_SOURCE` | `discord` | `discord` or `export` (`ExportDiscordSource`, a DiscordChatExporter JSON export) |
| `INFOVORE_DB_PATH` | required | SQLite file; keep on local durable storage, never a network filesystem |
| `INFOVORE_SCRATCH_DIR` | `scratch` | scratch files |
| `INFOVORE_DISCORD_TOKEN` | required for `discord` | bot token |
| `INFOVORE_GUILD_ID` | required for `discord`; optional for `export` | guild id, inferred from the export when unset |
| `INFOVORE_EXPORT_DIR` | required for `export` | directory of exported JSON |
| `INFOVORE_CHANNEL_IDS` | unset | optional comma-separated channel id allowlist; unset means all channels |
| `INFOVORE_EXCLUDE_CHANNELS` | unset | comma-separated channel names that are never archived or sampled; always beats `--channels` |
| `INFOVORE_QUIET_GAP_MINUTES` | `30` | quiet gap that closes an exchange |
| `INFOVORE_EXCHANGE_MAX_MESSAGES` | `50` | split size for long exchanges |
| `INFOVORE_BATCH_SIZE` | `10` | extraction batch size |
| `INFOVORE_MAX_RETRIES` | `3` | retries for transient failures |
| `INFOVORE_OPT_OUT_ROLE` | `no-archive` | role name that opts a member out |
| `INFOVORE_INCLUDE_BOT_MESSAGES` | false | ingest bot messages |
| `INFOVORE_TRIAGE_MIN_SCORE` | `0.3` | minimum rule-based triage score |
| `INFOVORE_TRIAGE_RULES` | built-in | triage rule overrides |
| `INFOVORE_PSEUDONYM_SALT` | unset | secret salt for the per-user pseudonyms `claims extract` shows the model; keep it stable and never commit it |
| `INFOVORE_WORKERS` | CPU count | worker processes |
| `INFOVORE_<STAGE>_BACKEND`, `_MODEL`, `_CONCURRENCY`, `_TIMEOUT` | `claude_cli`, per stage, `2`, `60` | LLM backend per stage (`EXTRACT`, `PROBE`, `JUDGE`); other `INFOVORE_<STAGE>_*` keys, such as `BASE_URL` and `API_KEY` for `openai_compat`, pass through as backend options |

`INFOVORE_TRIAGE_MIN_P_LORE` was removed and is rejected at startup.

## Commands

Run as `uv run infovore <command>` from a checkout, or `infovore <command>` once installed. `--help` on any command lists all flags.

- `infovore status`: row counts, queues, last runs and configured backends.
- `infovore search TERMS [--limit N] [--context N] [--exchanges] [--all]`: full-text search of the message corpus.
- `infovore backfill [--page-size N]`: walk allowlisted channels into the raw tables, resuming from checkpoint.
- `infovore chunk [--rechunk] [--recipe N] [--dry-run] [--measure] [--gap M] [--adaptive] [--fold F] [--channels LIST]`: group messages into closed exchanges; `--measure` reports without writing.
- `infovore run [--interval S] [--once]`: live ingest plus a periodic chunk, triage and cascade loop, with no LLM, until SIGTERM or SIGINT.
- `infovore sync-optouts`: sync the opt-out role and redact newly opted-out users' history.
- `infovore snapshot DEST [--force]`: consistent copy of the database through the SQLite backup API.
- `infovore export-archive DEST [--force]`: shareable SQLite file of archived exchanges.
- `infovore slice freeze | show`: freeze the evaluation slices once; show exchanges and messages per size bucket.
- `infovore judge serve | report`: human judging page and its report (see below).
- `infovore words serve [--host H] [--port P] [--top-n N]`: dense page to mark undecided-conversation words as tech or not; checked words join the lexicon as `reviewed`.
- `infovore labels export --jsonl PATH [--include-excluded-channels]`: one JSON object per labelled current exchange (text as `llm-score` renders it, label, held_out, slice, channel, cascade_stage, label_source) plus counts.
- `infovore words report [--top-n N] [--show K]`: reviewed and approved counts, candidates left, and undecided conversations with an approved word.
- `infovore relevance cascade [--slices LIST] [--all] [--write] [--explain ID]`: run the cascade and report; `--write` records derived annotations.
- `infovore relevance compare [--cv K] [--model M] [--pool first|mean|max] [--cache FILE] [--residue]`: stratified cross-validation of human Bayes against embedding plus logistic head.
- `infovore relevance embed-score [--cv K] [--model M] [--cache FILE]`: write `p_relevant_embed` for labelled exchanges.
- `infovore relevance llm-score --endpoint URL --model NAME [--residue] [--slices LIST] [--limit N] [--write] [--dry-run]`: score exchanges with a served OpenAI-compatible model.
- `infovore relevance mine [--tech CH] [--off CH] [--min-count N]`: candidate lexicon terms by channel log-odds.
- `infovore relevance collisions [--tech CH] [--off CH] [--terms LIST]`: lexicon terms common in off-topic channels.
- `infovore triage [--report] [--human-report] [--train-human] [--scorer NAME] [--scorer-version V] [--include-training] [--human-limit N] [--all-exchanges]`: rule-based triage scores, reports and the human-label classifier.
- `infovore label [--from-runs IDS] [--exchange-id ID] [--lore | --noise]`: record lore or noise labels.
- `infovore sift export | import | citations | train | serve`: message-level trash sifting. `export` writes an lnav batch, `import` records its labels, `serve` is a browser UI; both take `--size`, `--strategy random|uncertain|mixed`, `--seed` and `--channels`.
- `infovore claims extract --endpoint URL --model ALIAS (--slices LIST | --ids LIST | --channels LIST) --limit N [--write] [--dry-run] [--resume] [--concurrency N] [--timeout S] [--progress-every N]`: author-redacted claim extraction with a served local model; `--limit` is required, `--dry-run` sends nothing, `--write` stores the run.
  - `--channels a,b`: current archived conversations (cascade relevant or undecided) in those channels, by exchange id; excluded channels never.
  - `--resume`: with `--write`, skip conversations already done with the same model id and prompt hash; failed ones retry. Ctrl-C or SIGTERM finishes in-flight requests, commits them, and exits 130.
  - `--concurrency N`: requests in flight, default and maximum 4; each conversation is committed as it completes.
  - `--timeout S`: per-request seconds, default 300; a timeout is a failure and retries on `--resume`.
  - `--progress-every N`: stderr line (done/total, claims, conversations per hour, ETA) every N conversations, default 10.
- `infovore claims serve --run RUN [--host H] [--port P]`: dense review page (g good, w wrong, m made up, n not useful); verdicts append and resume.
- `infovore claims report [--run RUN]`: per run conversations, claims, zero-claim share, verdict counts, made-up rate, tokens and seconds.
- `infovore claims check --run RUN [--run RUN] [--write] [--threshold T]`: no model calls; verifies each claim's numbers, versions, part numbers, models, quoted strings and file names against the cited messages and scores word overlap; prints a confusion table against your reviews and a tuned threshold. `claims show --run RUN --check` lists stored verdicts and the failing fact.
- `infovore extract [--mode trial|live] [--sample N] [--seed N] [--exchange-id ID] [--min-score X] [--max-score X] [--strategy stratified|random] [--compare-prompt V]`: claim extraction (paused).
- `infovore probe [--run-id IDS] [--limit N] [--probe-model M] [--retry-failed] [--compare]`: closed-book novelty probe over claims (paused).
- `infovore review [--run-ids IDS] [--out FILE]`: HTML report for prompt-version run sets (paused).
- `infovore promote --prompt-version V`: promote a prompt version to live (paused).

## Human judging workflow

1. `infovore slice freeze` once, to fix the evaluation slices.
2. `infovore judge serve [--host H] [--port P] --queue frozen|uncertain|c1|likely-irrelevant` serves the judging page until Ctrl-C. `--queue uncertain` orders by scorer uncertainty and takes `--scorer`.
3. Judge exchanges in the browser; labels land in `exchange_labels` and `label_events`.
4. `infovore judge report [--by-channel] [--min-labels N]` prints labels, slice progress, self-agreement and labels still needed. `--by-channel` breaks labels down per channel.

## Measuring

- `infovore relevance cascade --slices s1,s2,...` evaluates the cascade on the named frozen slices without writing; add `--write` to record annotations and `--explain ID` to show which stage decided an exchange.
- `infovore triage --human-report --scorer NAME` reports a scorer against the human labels.
- `infovore relevance compare` compares the human Bayes stage with the embedding head under cross-validation; `--residue` restricts it to residue exchanges.

Measurement history and results are kept in the mcm-engine knowledge base, not in this repository.

## Privacy and opt-out

Members opt out by holding the role named by `INFOVORE_OPT_OUT_ROLE` (default `no-archive`). The bot needs the Message Content and Server Members intents. `infovore sync-optouts` syncs the role and redacts the history of newly opted-out users. Redaction happens before storage in ingest, so a re-run never un-redacts anything. `INFOVORE_EXCLUDE_CHANNELS` keeps whole channels out of the archive and out of every sampler. Share archives only through `export-archive`.

## Deployment

Nothing host-specific lives in code. `infovore` is one console-script package (`uv tool install .`) and one container image from the `Dockerfile`; `/data` holds `INFOVORE_DB_PATH` and `INFOVORE_SCRATCH_DIR`. Never place the database on EFS, FSx or any network filesystem.

- Initial load: fetch from the Discord API once with DiscordChatExporter (`exportguild --include-threads all`), set `INFOVORE_SOURCE=export` and `INFOVORE_EXPORT_DIR`, then `infovore backfill` and `infovore chunk`.
- Steady state: `infovore run` keeps ingest and the chunk, triage and cascade loop alive. Run it under launchd, systemd or `docker run -d --restart unless-stopped -v "$PWD/data:/data" --env-file .env infovore run`.
- Periodic alternative: a timer that runs `infovore backfill && infovore chunk`.
- `deploy/run-unattended.sh start --i-approved` is the tmux and caffeinate launcher for `infovore run` (no extraction). It refuses to start without `--i-approved`.
- The `claude_cli` backend needs `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) on headless hosts; `openai_compat` needs `INFOVORE_<STAGE>_BASE_URL` and `INFOVORE_<STAGE>_API_KEY`.

## Development

```
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Work is red/green TDD on a branch opened as a PR that references its issue. Coverage is enforced at 100% line and branch. `scripts/red_green.sh BASE HEAD` checks that changed tests fail on BASE and pass on HEAD.
