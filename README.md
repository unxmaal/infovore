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

`infovore.llm.registry.Registry` maps each stage's configured backend name to a `BackendFactory` (`name`, `validate(StageSettings) -> list[str]`, `build(StageSettings) -> LLMBackend`); `default_registry()` currently registers only the `fake` factory. `Registry.validate(settings)` reports an unknown backend name per stage plus anything the matching factory's own `validate` rejects; `Registry.build_backends(settings)` raises `ConfigError` if validation fails, otherwise returns one backend per stage; `Registry.health_check(backends)` sends one trivial request per backend and reports `None` on success or the error message on failure, without ever raising.

## Running

```
uv run infovore status
uv run infovore chunk [--now 2026-01-01T00:00:00+00:00]
```

`chunk` groups ingested messages into exchanges and persists the closed ones; `--now` overrides the clock, which is useful when iterating over an old backfill.

Every subcommand loads configuration (environment, then `.env` in the working directory for anything not set), opens and migrates the database, and runs. `status` prints row counts, the exchange queue by status, claims by novelty, run outcomes, the last extraction and probe times, the live prompt version, and each stage's backend and model.

Exit codes: `0` ok, `1` unexpected failure, `2` configuration or usage error, `3` an LLM backend is unavailable.

`infovore.source.live.build_client` requires two privileged intents to be enabled for the bot application in the Discord Developer Portal (Bot tab): **Message Content** (message bodies) and **Server Members** (role membership sync for opt-out). Without both enabled, the gateway connection is rejected. The full first-run runbook lands with the `backfill`/`run` CLI subcommands.

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
- `infovore.source.live.connect`: performs the real Discord login/gateway handshake over the network; PLAN operating rule 4 forbids tests from opening a network connection, so this one-line wrapper around `discord.Client.start` cannot be exercised in the test suite.

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

## Development workflow

```
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Every change is red/green TDD on a feature branch named `<issue>-<slug>`, opened as a PR that references its issue. Coverage is enforced at 100% line and branch.

Import boundaries are enforced by ruff `banned-api`: `discord` only in `infovore/source/live.py`, process spawning only in `infovore/llm/claude_cli.py`, `openai`/`httpx` only in `infovore/llm/openai_compat.py`.
