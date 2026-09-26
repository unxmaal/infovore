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
