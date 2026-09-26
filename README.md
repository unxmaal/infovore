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

## Grouping rules

`infovore/chunk/rules.py` groups the `MessageRow`s of one channel (plus its threads, which carry a `thread_id`) into `Group`s. Each `Group` has a `rule` (`GroupingRule.THREAD` | `REPLY_CHAIN` | `QUIET_GAP`), an ordered `messages` tuple, and a `context` tuple of uncitable overlap messages, non-empty only for parts produced by the size-cap split below. Every message given to `group_messages` ends up in exactly one group's `messages`, or is dropped first. Messages within a group, and groups within the returned list, are ordered by `(created_at, id)` (a group's position is its first message's key).

**Drops** (lifecycle rule 7): `drop_ungroupable` removes deleted messages (`deleted_at` set) unconditionally, and bot-authored messages (`author_is_bot`) unless `include_bots=True`. System messages never reach the chunker; they are filtered at ingest/normalize time and have no representation in `MessageRow`.

**Precedence** is thread > reply_chain > quiet_gap. Each rule claims messages from what the previous rule left behind; nothing is grouped by more than one rule.

- **thread**: every kept message with a non-null `thread_id` is grouped with every other kept message sharing that `thread_id`, one group per distinct `thread_id`. This runs first, so a message inside a thread is never pulled into a reply chain or quiet-gap group with anything outside its thread, even if it replies to a message outside the thread — that parent is simply not part of the thread's group, and is handled by a later rule on its own. This is the "reply inside a thread whose parent is outside the thread" case: the reply stays in its thread group; the outside parent, if it has no other links, becomes its own quiet_gap group of one.
- **reply_chain**: among messages not claimed by `thread`, this is the transitive closure over `reply_to_id` — the connected components of the reply graph, treating each message as a node and each `reply_to_id` that points at another message in the same input as an edge. A component is a `reply_chain` group only if it has more than one message; a message with no reply links, or whose only reply link points outside this input (already grouped by `thread`, in a different channel, or missing entirely), forms a component of size one and falls through to `quiet_gap` instead. A cycle in the reply graph (possible only from malformed data) is tolerated: union-find just treats it as an already-merged edge and the messages still end up in one group.
- **quiet_gap**: the messages left after `thread` and `reply_chain` are sorted by `(created_at, id)` and split into a new group whenever the gap between consecutive messages strictly exceeds `quiet_gap` (a `timedelta`, default 30 minutes — exactly 30 minutes does not split). A single leftover message becomes a group of one.

**Size cap** (lifecycle rule 6): after the three rules produce their groups, `split_oversized` splits any group whose `messages` exceeds `max_messages` (default 50) into consecutive, non-overlapping parts, each within the cap, by repeatedly cutting the longest prefix that still fits. Concretely: while more than `max_messages` messages remain unsplit, look at the first `max_messages + 1` of them, find the internal gap (the time delta between two consecutive messages) that is largest — ties broken toward the later position, so a run of equal gaps fills each part up to the cap rather than making many tiny parts — and cut the prefix there; that prefix (at most `max_messages` messages, at least one) becomes the next part, and the scan continues on the rest. The final remainder, now `<= max_messages`, becomes the last part. Each part after the first carries the previous part's last `overlap` messages (default 3) as its `context`, capped at however many messages that previous part actually has; `context` never contains any message that is also in the same part's `messages`, and it is never itself split or citable.

**Closing** (lifecycle rule 2): `is_closed(group, now, quiet_gap)` reports a group as closed only once its newest message's `created_at` is strictly older than `quiet_gap` relative to `now`; at exactly `quiet_gap` it is still open. This applies uniformly to `thread`, `reply_chain`, and `quiet_gap` groups alike — the grouper (#16) uses it to decide when a group is done accumulating messages and can be persisted as an exchange.

## Extraction prompt

## Privacy and opt-out

## Coverage exclusions

- `...` bodies: Protocol method stubs have no executable behavior; they define shapes that implementations are tested against.
- `if TYPE_CHECKING:` blocks: imports needed only by the type checker never run at runtime.

## Consuming the database

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
