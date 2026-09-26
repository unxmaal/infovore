FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev
COPY infovore ./infovore
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.12-slim AS runtime
ARG WITH_CLAUDE_CLI=0
RUN set -eux; \
    if [ "$WITH_CLAUDE_CLI" = "1" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends ca-certificates curl gnupg; \
        mkdir -p /etc/apt/keyrings; \
        curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
            | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg; \
        echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
            > /etc/apt/sources.list.d/nodesource.list; \
        apt-get update; \
        apt-get install -y --no-install-recommends nodejs; \
        npm install -g @anthropic-ai/claude-code; \
        apt-get purge -y curl gnupg; \
        apt-get autoremove -y; \
        rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/nodesource.list /etc/apt/keyrings/nodesource.gpg; \
    fi

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin infovore

WORKDIR /app
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INFOVORE_DB_PATH=/data/infovore.db \
    INFOVORE_SCRATCH_DIR=/data/scratch

RUN mkdir -p /data/scratch && chown -R infovore:infovore /data /app

VOLUME ["/data"]
USER infovore

ENTRYPOINT ["infovore"]
