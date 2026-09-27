"""Shared process-pool plumbing for triage's CPU-bound scoring (issue #113).

After #104 removed the I/O bottleneck (batched loads, batched commits),
`triage --train`'s `p_lore` scoring, a full rule rescore, and the
`--suggest-terms` corpus document-frequency scan are all CPU-bound pure
Python on one core: tokenizing, `score.score_exchange`'s regexes, and
`bayes.p_lore` (clue sort + chi2) are per-exchange and embarrassingly
parallel. SQLite access stays in the main process (single-writer): a batch
is loaded in the main process (`infovore.db.batch.exchange_inputs_for_ids`),
handed to workers as plain, already-picklable data (`MessageRow` /
`ReactionRow` / `AttachmentRow` lists -- never a `sqlite3.Connection`), and
the scored results come back to the main process, which does the batched
write.

`ChunkPool` is the one place that knows how to run a picklable `worker`
function over chunks of items, either in-process (`workers <= 1`: no pool
at all) or spread across a persistent `ProcessPoolExecutor` (`workers > 1`)
whose worker processes are initialized once via `initializer`/`initargs` --
so a model or a `TriageRules` is sent to each worker exactly once, not once
per task.

Every worker/initializer function this module or its callers use is a
plain, top-level (module-level) function -- never a lambda or closure --
which is what makes them safe to hand to a `ProcessPoolExecutor` regardless
of whether the platform's multiprocessing start method is `'fork'` or
`'spawn'` (`'spawn'` re-imports the target module in the child and looks the
function up by name; a closure has no such name).

`workers=1` (the default, and what the test suite uses unless a test asks
for a real pool) never creates an executor: `ChunkPool.map_chunks` calls the
same worker function directly, in the *calling* process. That is also how
these functions end up covered by an ordinary (non-multiprocess) test run --
a `ProcessPoolExecutor` worker's lines run in a child process, and this repo
doesn't carry the extra `coverage`/`pytest-cov` configuration needed to
collect coverage across that process boundary, so the `workers=1` path is
the one place a worker function is guaranteed to run in-process where
`pytest-cov` can see it.
"""

import functools
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from types import TracebackType
from typing import Self


def chunk_for_workers[T](items: Sequence[T], workers: int) -> list[list[T]]:
    """Split `items` into at most `max(1, workers)` contiguous, order-
    preserving chunks of roughly equal size (never more chunks than items,
    and never zero chunks unless `items` is empty)."""
    if not items:
        return []
    workers = max(1, workers)
    if workers == 1:
        return [list(items)]
    size = -(-len(items) // workers)  # ceil division, no floats
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


class ChunkPool[T, R]:
    """Runs `worker` over `chunk_for_workers`-sized chunks of items, either
    directly in-process (`workers <= 1`) or across a `ProcessPoolExecutor`
    (`workers > 1`) whose workers are set up once via `initializer(*initargs)`
    -- called once per worker *process* by the pool, or once, directly, right
    here, for the `workers <= 1` in-process path (so that path sees exactly
    the same initialized state a real worker process would).

    `worker` and `initializer` must be plain module-level functions (spawn-
    safe: picklable by reference, not by value) -- see the module docstring.
    """

    def __init__(
        self,
        worker: Callable[[list[T]], R],
        workers: int,
        initializer: Callable[..., None] | None = None,
        initargs: tuple[object, ...] = (),
    ) -> None:
        self._worker = worker
        self._workers = max(1, workers)
        self._executor: ProcessPoolExecutor | None = None
        if self._workers > 1:
            # `functools.partial` bakes `initargs` into a zero-argument
            # callable: typeshed's `ProcessPoolExecutor.__init__` overloads
            # tie `initargs`'s type to a `TypeVarTuple` capture of
            # `initializer`'s own parameters, which a `Callable[..., None]`
            # (arbitrary args, needed since every caller's initializer takes
            # different arguments) can't satisfy -- passing a bound,
            # zero-argument callable sidesteps that instead of an `ignore`.
            bound_initializer = (
                None if initializer is None else functools.partial(initializer, *initargs)
            )
            self._executor = ProcessPoolExecutor(
                max_workers=self._workers, initializer=bound_initializer
            )
        elif initializer is not None:
            initializer(*initargs)

    def map_chunks(self, items: Sequence[T]) -> list[R]:
        """One `R` per chunk of `items` (`chunk_for_workers(items, workers)`),
        in chunk order -- callers combine per-chunk results themselves
        (`list.extend` for a worker that returns a list of per-item results,
        `Counter.update` for one that returns a partial `Counter`, etc.),
        since that combination differs per caller."""
        chunks = chunk_for_workers(items, self._workers)
        if self._executor is None:
            return [self._worker(chunk) for chunk in chunks]
        futures = [self._executor.submit(self._worker, chunk) for chunk in chunks]
        return [future.result() for future in futures]

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.shutdown()
