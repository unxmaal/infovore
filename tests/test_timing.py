from datetime import UTC, datetime, timedelta

from infovore.timing import AsyncioSleeper, FixedClock, RecordingSleeper, SystemClock


def test_system_clock_is_timezone_aware_utc() -> None:
    assert SystemClock().now().tzinfo is UTC


def test_fixed_clock_returns_its_time_and_advances() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    clock = FixedClock(start)
    assert clock.now() == start
    clock.advance(timedelta(minutes=5))
    assert clock.now() == start + timedelta(minutes=5)


async def test_asyncio_sleeper_sleeps() -> None:
    await AsyncioSleeper().sleep(0)


async def test_recording_sleeper_records_and_advances_its_clock() -> None:
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    sleeper = RecordingSleeper(clock)
    await sleeper.sleep(2.5)
    assert sleeper.slept == [2.5]
    assert clock.now() == datetime(2026, 1, 1, 0, 0, 2, 500000, tzinfo=UTC)


async def test_recording_sleeper_without_clock_only_records() -> None:
    sleeper = RecordingSleeper()
    await sleeper.sleep(1)
    assert sleeper.slept == [1]
