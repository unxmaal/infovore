import asyncio

import pytest

from infovore.source.live import DiscordPySource, open_discord_source
from infovore.source.protocol import SourceUnavailableError


class SessionClient:
    def __init__(self, becomes_ready: bool = True) -> None:
        self.becomes_ready = becomes_ready
        self.ready = asyncio.Event()
        self.closed = False
        self.started_with: str | None = None

    def get_guild(self, guild_id: int) -> None:
        return None

    def get_channel(self, channel_id: int) -> None:
        return None

    async def wait_until_ready(self) -> None:
        await self.ready.wait()

    async def close(self) -> None:
        self.closed = True


def starter_that_connects(client: SessionClient) -> "Starter":
    async def start(_: object, token: str) -> None:
        client.started_with = token
        if client.becomes_ready:
            client.ready.set()
        await asyncio.Event().wait()

    return start


Starter = object


async def test_yields_a_discord_source_once_ready_and_closes_afterwards() -> None:
    client = SessionClient()
    async with open_discord_source(
        "tok", 1.0, client_factory=lambda: client, starter=starter_that_connects(client)
    ) as source:
        assert isinstance(source, DiscordPySource)
        assert client.started_with == "tok"
        assert not client.closed
    assert client.closed


async def test_not_ready_in_time_is_unavailable_and_closes() -> None:
    client = SessionClient(becomes_ready=False)
    with pytest.raises(SourceUnavailableError, match="not ready within 0.05s"):
        async with open_discord_source(
            "tok", 0.05, client_factory=lambda: client, starter=starter_that_connects(client)
        ):
            raise AssertionError("must not yield")
    assert client.closed


class LoginFailure(Exception):
    pass


async def test_login_failure_is_unavailable_without_leaking_the_token() -> None:
    client = SessionClient(becomes_ready=False)

    async def failing(_: object, token: str) -> None:
        raise LoginFailure(f"bad token {token}")

    with pytest.raises(SourceUnavailableError) as raised:
        async with open_discord_source(
            "SECRET-TOKEN", 1.0, client_factory=lambda: client, starter=failing
        ):
            raise AssertionError("must not yield")
    assert "LoginFailure" in str(raised.value)
    assert "SECRET-TOKEN" not in str(raised.value)
    assert client.closed


async def test_client_stopping_before_ready_is_unavailable() -> None:
    client = SessionClient(becomes_ready=False)

    async def returns_early(_: object, token: str) -> None:
        return None

    with pytest.raises(SourceUnavailableError, match="stopped before it was ready"):
        async with open_discord_source(
            "tok", 1.0, client_factory=lambda: client, starter=returns_early
        ):
            raise AssertionError("must not yield")
    assert client.closed


async def test_errors_in_the_body_still_close_the_client() -> None:
    client = SessionClient()
    with pytest.raises(RuntimeError, match="boom"):
        async with open_discord_source(
            "tok", 1.0, client_factory=lambda: client, starter=starter_that_connects(client)
        ):
            raise RuntimeError("boom")
    assert client.closed


async def test_a_starter_that_fails_after_ready_is_not_reraised_on_close() -> None:
    client = SessionClient()

    async def ready_then_fail(_: object, token: str) -> None:
        client.ready.set()
        await asyncio.sleep(0)
        raise ConnectionResetError("gateway dropped")

    async with open_discord_source(
        "tok", 1.0, client_factory=lambda: client, starter=ready_then_fail
    ):
        await asyncio.sleep(0.01)
    assert client.closed
