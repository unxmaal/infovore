"""Tests for the shared process-pool plumbing (issue #113)."""

from infovore.triage.parallel import ChunkPool, chunk_for_workers

# --- chunk_for_workers -------------------------------------------------------


def test_chunk_for_workers_empty_returns_no_chunks() -> None:
    assert chunk_for_workers([], 4) == []


def test_chunk_for_workers_one_worker_returns_a_single_chunk() -> None:
    assert chunk_for_workers([1, 2, 3, 4, 5], 1) == [[1, 2, 3, 4, 5]]


def test_chunk_for_workers_splits_into_at_most_workers_contiguous_chunks() -> None:
    chunks = chunk_for_workers(list(range(10)), 3)
    assert len(chunks) <= 3
    assert [item for chunk in chunks for item in chunk] == list(range(10))


def test_chunk_for_workers_never_produces_more_chunks_than_items() -> None:
    chunks = chunk_for_workers([1, 2], 8)
    assert len(chunks) == 2
    assert chunks == [[1], [2]]


def test_chunk_for_workers_treats_non_positive_workers_as_one() -> None:
    assert chunk_for_workers([1, 2, 3], 0) == [[1, 2, 3]]


# --- ChunkPool: module-level, spawn-safe worker/initializer functions --------

_offset = 0


def _init_offset(value: int) -> None:
    global _offset
    _offset = value


def _add_offset_chunk(items: list[int]) -> list[int]:
    return [item + _offset for item in items]


def _sum_chunk(items: list[int]) -> int:
    return sum(items)


# --- ChunkPool: workers=1 (in-process, no pool) -----------------------------


def test_chunk_pool_workers_one_never_creates_an_executor() -> None:
    with ChunkPool(_add_offset_chunk, workers=1) as pool:
        assert pool._executor is None


def test_chunk_pool_workers_one_runs_the_worker_directly_in_process() -> None:
    with ChunkPool(_add_offset_chunk, workers=1) as pool:
        results = pool.map_chunks([1, 2, 3])
    assert results == [[1, 2, 3]]


def test_chunk_pool_workers_one_calls_initializer_once_in_process() -> None:
    with ChunkPool(_add_offset_chunk, workers=1, initializer=_init_offset, initargs=(10,)) as pool:
        results = pool.map_chunks([1, 2, 3])
    assert results == [[11, 12, 13]]


def test_chunk_pool_workers_one_with_no_items_returns_empty() -> None:
    with ChunkPool(_sum_chunk, workers=1) as pool:
        assert pool.map_chunks([]) == []


# --- ChunkPool: real multi-worker pool --------------------------------------


def test_chunk_pool_workers_two_creates_a_real_process_pool() -> None:
    with ChunkPool(_sum_chunk, workers=2) as pool:
        assert pool._executor is not None


def test_chunk_pool_workers_two_matches_workers_one_results() -> None:
    items = list(range(37))

    with ChunkPool(_add_offset_chunk, workers=1, initializer=_init_offset, initargs=(5,)) as serial:
        serial_result = [value for chunk in serial.map_chunks(items) for value in chunk]

    with ChunkPool(_add_offset_chunk, workers=2, initializer=_init_offset, initargs=(5,)) as parallel:
        parallel_result = [value for chunk in parallel.map_chunks(items) for value in chunk]

    assert parallel_result == serial_result == [item + 5 for item in items]


def test_chunk_pool_shutdown_is_safe_to_call_when_no_executor_exists() -> None:
    pool = ChunkPool(_sum_chunk, workers=1)
    pool.shutdown()  # no executor was created; must not raise
