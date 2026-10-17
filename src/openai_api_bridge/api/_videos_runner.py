"""Background runner that drives a video job from queued → completed/failed.

Runs as an asyncio task scheduled by the ``TaskScheduler``. The task itself
holds the input_reference bytes (transient, in-memory). If the bridge restarts
mid-flight, the row is cleaned up by ``JobStore.mark_stale_failed`` at startup.
"""

from __future__ import annotations

import asyncio
import logging

from ..backends.base import InputImage
from ..dispatcher import BackendDispatcher
from ..errors import BridgeError
from ..infra.filestore import FileStore
from ..infra.jobstore import JobStore

log = logging.getLogger(__name__)


async def run_video_job(
    *,
    job_id: str,
    provider_id: str,
    model_slug: str,
    prompt: str,
    size: str | None,
    aspect_ratio: str | None,
    seconds: float | None,
    input_reference: InputImage | None,
    dispatcher: BackendDispatcher,
    jobstore: JobStore,
    filestore: FileStore,
    timeout_seconds: float | None = None,
) -> None:
    """Drive one video generation; persist state transitions to ``video_jobs``.

    ``timeout_seconds`` bounds the whole job, render and storage both, and
    fails the row with a message that says so. It is enforced here rather
    than left to the scheduler's hard cap, because a cap applied from outside
    arrives as a cancellation and the row would read "Job cancelled" — the
    same as a user's ``DELETE`` — for what was a timeout. ``None`` sets none.
    """
    deadline = asyncio.timeout(timeout_seconds)

    async def _persist_upstream_id(upstream_id: str) -> None:
        try:
            await jobstore.update(job_id, upstream_id=upstream_id)
        except Exception:
            log.exception("Failed to persist upstream_id for job %s", job_id)

    async def _render() -> None:
        await jobstore.update(job_id, status="in_progress")
        backend = dispatcher.for_provider(provider_id)
        asset = await backend.generate_video(
            model_slug=model_slug,
            prompt=prompt,
            size=size,
            aspect_ratio=aspect_ratio,
            seconds=seconds,
            input_reference=input_reference,
            on_upstream_id=_persist_upstream_id,
        )
        file_id = await filestore.put(
            asset.data,
            content_type=asset.content_type,
            kind=asset.kind,
            source_backend=provider_id,
            source_model=model_slug,
            prompt_excerpt=prompt,
            pinned=False,
        )
        await jobstore.update(
            job_id,
            status="completed",
            file_id=file_id,
            progress_pct=100,
            # What the backend actually rendered, which snapping means need not
            # be what the request named. None is written through as NULL rather
            # than skipped: a backend that deals in no ratios honoured nothing,
            # so the row must stop reporting the request as if it had.
            aspect_ratio=asset.aspect_ratio,
        )
        log.info("Video job %s completed; file_id=%s", job_id, file_id)

    try:
        async with deadline:
            await _render()
    except asyncio.CancelledError:
        log.info("Video job %s cancelled; marking failed", job_id)
        # Best-effort: mark the row failed so it doesn't sit in_progress
        # forever. We swallow errors here so the cancellation propagates
        # cleanly even if the DB write itself races with shutdown.
        # Conditional so we don't overwrite the more specific message a
        # DELETE /v1/videos/{id} caller already recorded — that caller marks
        # the row terminal before cancelling us.
        try:
            await jobstore.fail_if_active(job_id, "Job cancelled")
        except Exception:
            log.exception("Failed to mark cancelled job %s as failed", job_id)
        raise
    except TimeoutError:
        if not deadline.expired():
            # Not our deadline: something inside raised a bare TimeoutError.
            # Treat it like any other unexpected failure.
            log.exception("Video job %s failed (unexpected)", job_id)
            await jobstore.update(
                job_id, status="failed", error_message="Internal error: TimeoutError"
            )
            return
        log.warning("Video job %s exceeded its %ss limit", job_id, timeout_seconds)
        await jobstore.update(
            job_id,
            status="failed",
            error_message=(
                f"Video job timed out after {timeout_seconds:.0f}s (VIDEO_JOB_TIMEOUT_SECONDS)"
            ),
        )
    except BridgeError as e:
        log.warning("Video job %s failed (bridge error): %s", job_id, e.message)
        await jobstore.update(job_id, status="failed", error_message=e.message)
    except Exception as e:
        log.exception("Video job %s failed (unexpected)", job_id)
        # The type only. The row is served to clients by GET /v1/videos/{id},
        # and an unexpected exception's text is whatever the code that raised
        # it chose to put there — upstream URLs, file paths, response
        # fragments. The full exception and traceback are in the log above,
        # the same split the synchronous routes' 500 handler makes.
        await jobstore.update(
            job_id,
            status="failed",
            error_message=f"Internal error: {type(e).__name__}",
        )
