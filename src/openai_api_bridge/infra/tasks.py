"""Background work that outlives the request which started it.

Two shapes, both here because they share the same hazard: a task nobody holds a
reference to can be garbage-collected mid-flight, and a task nobody drains is
still pending when the loop is torn down.

* :class:`TaskScheduler` — a bounded pool for video jobs. We avoid FastAPI's
  BackgroundTasks because those are tied to a single request's response
  lifecycle and can be cancelled if the client disconnects mid-response.
  Video jobs need to outlive the HTTP request that kicks them off. A secondary
  index by ``name`` enables external cancellation (used by
  ``DELETE /v1/videos/{id}``).
* :class:`SingleFlight` — at most one live task per key, for work a request
  starts, stops waiting on, and wants the *next* request to join rather than
  duplicate.

Both remove entries with an identity check rather than by key. That is not
incidental: a key can be reused while the previous task is still running, and
an unconditional delete lets a finishing task evict its live successor —
leaving that successor untracked, undrained, and holding the very resource the
map exists to protect.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from types import TracebackType
from typing import Any

log = logging.getLogger(__name__)

# Defensive upper bound on how long any one background task can hold its
# semaphore permit, for a task submitted without its own. A backstop, not a
# deadline: a cap applied from out here reaches the task as a cancellation,
# so work that wants its timeout reported as one (video jobs do) enforces it
# itself and submits with a ``timeout_seconds`` a little past it.
DEFAULT_TASK_TIMEOUT_SECONDS = 1800.0


class SchedulerFull(Exception):
    """Raised by :meth:`TaskScheduler.reservation` when no slot is free."""

    def __init__(self, in_flight: int, capacity: int) -> None:
        super().__init__(f"{in_flight} tasks in flight (capacity {capacity})")
        self.in_flight = in_flight
        self.capacity = capacity


class Reservation:
    """A slot claimed by :meth:`TaskScheduler.reserve`, released on exit."""

    def __init__(self, scheduler: TaskScheduler) -> None:
        self._scheduler: TaskScheduler | None = scheduler

    def __enter__(self) -> Reservation:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._scheduler is not None:
            self._scheduler._release_reservation()
            self._scheduler = None


class TaskScheduler:
    def __init__(
        self,
        max_concurrent: int,
        *,
        default_task_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._sem = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        # Slots claimed by reservation() whose task is not yet submitted.
        self._reserved = 0
        self._tasks: set[asyncio.Task[Any]] = set()
        self._tasks_by_name: dict[str, asyncio.Task[Any]] = {}
        self._default_timeout = default_task_timeout_seconds

    def submit(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        timeout_seconds: float | None = None,
    ) -> asyncio.Task[Any]:
        """Schedule a coroutine to run with concurrency capped by the semaphore.

        ``timeout_seconds`` overrides ``default_task_timeout_seconds`` for this
        one task. The timeout is enforced via ``asyncio.wait_for``, so the
        coroutine is cancelled (and the semaphore permit released) if it
        exceeds the budget.
        """
        timeout = self._default_timeout if timeout_seconds is None else timeout_seconds

        async def _runner() -> None:
            try:
                await self._sem.acquire()
            except asyncio.CancelledError:
                # Cancelled while still queued for a permit, so ``coro`` was
                # never started. Close it, or it is collected un-awaited and
                # Python warns "coroutine was never awaited" for every job
                # cancelled before its turn.
                coro.close()
                log.info("Background task %r cancelled before it started", name)
                raise
            try:
                await asyncio.wait_for(coro, timeout=timeout)
            except TimeoutError:
                log.error(
                    "Background task %r exceeded hard timeout of %.0fs; cancelled",
                    name,
                    timeout,
                )
            except asyncio.CancelledError:
                log.info("Background task %r cancelled", name)
                raise
            except Exception:
                log.exception("Background task %r failed", name)
            finally:
                self._sem.release()

        task = asyncio.create_task(_runner(), name=name)
        self._tasks.add(task)
        if name is not None:
            self._tasks_by_name[name] = task

        def _cleanup(t: asyncio.Task[Any]) -> None:
            self._tasks.discard(t)
            if name is not None and self._tasks_by_name.get(name) is t:
                del self._tasks_by_name[name]

        task.add_done_callback(_cleanup)
        return task

    def cancel(self, name: str) -> bool:
        """Cancel a running task by its submit-time ``name``.

        Returns ``True`` if a live task was found and cancellation was
        requested. ``False`` means the task either never existed or already
        finished — in both cases the caller's job state has already moved on.
        """
        task = self._tasks_by_name.get(name)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    @property
    def in_flight(self) -> int:
        """Tasks not yet finished, running or waiting, plus live reservations."""
        return len(self._tasks) + self._reserved

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    def reserve(self, capacity: int) -> Reservation:
        """Claim one of ``capacity`` slots now, until a task is submitted.

        Raises :class:`SchedulerFull` if ``in_flight`` is already at
        ``capacity``. The check and the claim are one synchronous step; use
        the returned :class:`Reservation` as a context manager around the
        work leading up to :meth:`submit`, whose task takes over the slot.
        Checking ``in_flight`` and submitting later instead leaves every
        ``await`` in between as a window in which concurrent callers all pass
        the same check and overshoot the cap.
        """
        if self.in_flight >= capacity:
            raise SchedulerFull(self.in_flight, capacity)
        self._reserved += 1
        return Reservation(self)

    def _release_reservation(self) -> None:
        self._reserved -= 1

    async def shutdown(self, *, timeout: float = 30.0) -> None:
        """Wait for all in-flight tasks to finish, then cancel any stragglers."""
        if not self._tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            log.warning(
                "Scheduler shutdown timed out with %d tasks still running; cancelling",
                len(self._tasks),
            )
            for t in list(self._tasks):
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)


class SingleFlight[T]:
    """At most one live task per key; later callers join it rather than duplicate it.

    For work a request starts but may stop waiting on — the model-catalogue
    fetches behind ``GET /v1/models``, where a slow provider is dropped from
    one listing while its fetch keeps running to warm that backend's cache.

    The entry is created *with* the task, not after the starter gives up.
    Registering later leaves a window — the whole time the starter is waiting —
    in which every concurrent caller sees an empty map, starts its own task,
    and then overwrites the previous entry on the way out. What made that
    subtle rather than obvious is that the survivors are fine: it is the
    *untracked* task that ends up holding the upstream request, so a drain
    awaits the wrong one and the connection it needs is closed underneath it.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[T]] = {}

    def join_or_start(
        self,
        key: str,
        factory: Callable[[], Coroutine[Any, Any, T]],
        *,
        name: str | None = None,
    ) -> asyncio.Task[T]:
        """The live task for ``key``, starting one from ``factory`` if there isn't one.

        A factory rather than a coroutine so nothing is constructed when an
        existing task is joined — an un-awaited coroutine would warn.
        """
        existing = self._tasks.get(key)
        if existing is not None:
            return existing
        task = asyncio.create_task(factory(), name=name)
        self._tasks[key] = task

        def _forget(finished: asyncio.Task[T]) -> None:
            # Identity, not key: by the time this fires the slot may hold a
            # newer task, and removing that one would untrack a live fetch.
            if self._tasks.get(key) is finished:
                del self._tasks[key]

        task.add_done_callback(_forget)
        return task

    def __len__(self) -> int:
        return len(self._tasks)

    def keys(self) -> list[str]:
        return list(self._tasks)

    async def cancel_all(self) -> None:
        """Cancel every live task and wait for the cancellations to land.

        Cancel rather than wait-then-cancel. Everything in here is by
        construction work some request already gave up on, and what it would
        finish for — a cache inside a backend the process is about to close —
        does not survive the shutdown either. Waiting first only adds its
        grace period to every restart.
        """
        tasks = list(self._tasks.values())
        if not tasks:
            return
        log.info("Cancelling %d in-flight background fetch(es)", len(tasks))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
