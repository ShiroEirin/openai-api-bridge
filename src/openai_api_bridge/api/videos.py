"""``POST /v1/videos`` (async submit), ``GET /v1/videos/{id}``, ``GET /v1/videos/{id}/content``."""

from __future__ import annotations

import logging
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Header, Request, Response, UploadFile

from ..config import parse_model_id
from ..errors import (
    InvalidRequest,
    JobNotFound,
    JobNotReady,
    UpstreamError,
    VideoQueueFull,
)
from ..infra.jobstore import VideoJob
from ..infra.tasks import SchedulerFull
from ..resources import resources
from ..util.sizes import check_size_bounds
from ._assets import asset_response
from ._videos_runner import run_video_job

log = logging.getLogger(__name__)

router = APIRouter()

# How far past a video job's own deadline the scheduler's hard cap sits: room
# for the runner to record the timeout before anything cancels it.
_BACKSTOP_GRACE_SECONDS = 60.0

# Ceiling on a requested clip length. ComfyUI turns seconds into a frame
# count, so an absurd value is an absurd allocation on the render box; the
# hosted video models top out well under this.
MAX_VIDEO_SECONDS = 120.0


def _video_to_dict(job: VideoJob) -> dict[str, Any]:
    """Render a VideoJob row in the OpenAI-compatible Sora video object shape."""
    completed_at = job.updated_at if job.status in ("completed", "failed") else None
    error = None
    if job.status == "failed" and job.error_message:
        error = {"message": job.error_message, "type": "api_error", "code": None}
    return {
        "id": job.id,
        "object": "video",
        "model": job.model,
        "status": job.status,
        "progress": job.progress_pct,
        "seconds": job.seconds,
        "size": job.size,
        "aspect_ratio": job.aspect_ratio,
        "created_at": job.created_at,
        "completed_at": completed_at,
        "error": error,
    }


@router.post("/v1/videos")
async def videos_create(
    request: Request,
    model: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
    size: Annotated[str | None, Form()] = None,
    aspect_ratio: Annotated[str | None, Form()] = None,
    seconds: Annotated[float | None, Form()] = None,
    input_reference: Annotated[
        UploadFile | None,
        File(description="Optional input image for image-to-video"),
    ] = None,
) -> dict[str, Any]:
    provider_id, model_slug = parse_model_id(model)
    res = resources(request)
    # Eagerly fail with 404 if the provider is unknown — friendlier than
    # accepting the job and immediately failing inside the runner.
    res.dispatcher.for_provider(provider_id)

    if seconds is not None and not 0 < seconds <= MAX_VIDEO_SECONDS:
        raise InvalidRequest(
            f"seconds must be greater than 0 and at most {MAX_VIDEO_SECONDS:g}", param="seconds"
        )
    check_size_bounds(size)

    # Claimed before the upload is read into memory — a queued job holds those
    # bytes until it starts, which is what this bound exists to limit — and
    # held across every await below until submit() takes it over, so
    # concurrent requests cannot all pass the check and overshoot it.
    capacity = res.scheduler.max_concurrent + res.settings.max_queued_video_jobs
    try:
        slot = res.scheduler.reserve(capacity)
    except SchedulerFull as e:
        raise VideoQueueFull(
            f"{e.in_flight} video jobs are already running or queued "
            f"(limit {capacity}); retry later."
        ) from None

    with slot:
        input_ref_bytes: bytes | None = None
        input_ref_ct: str | None = None
        if input_reference is not None:
            input_ref_bytes = await input_reference.read()
            if not input_ref_bytes:
                raise InvalidRequest("input_reference upload was empty", param="input_reference")
            input_ref_ct = input_reference.content_type or "image/png"

        job_id = secrets.token_hex(16)
        job = await res.jobstore.create(
            job_id=job_id,
            model=model,
            prompt=prompt,
            size=size,
            aspect_ratio=aspect_ratio,
            seconds=seconds,
        )

        res.scheduler.submit(
            run_video_job(
                job_id=job_id,
                provider_id=provider_id,
                model_slug=model_slug,
                prompt=prompt,
                size=size,
                aspect_ratio=aspect_ratio,
                seconds=seconds,
                input_reference=input_ref_bytes,
                input_reference_content_type=input_ref_ct,
                dispatcher=res.dispatcher,
                jobstore=res.jobstore,
                filestore=res.filestore,
                timeout_seconds=res.settings.video_job_timeout_seconds,
            ),
            name=f"video-job-{job_id}",
            # The runner enforces the job's deadline itself, so its row says it
            # timed out. The scheduler's cap is only the backstop past it, for a
            # runner that somehow does not stop.
            timeout_seconds=res.settings.video_job_timeout_seconds + _BACKSTOP_GRACE_SECONDS,
        )

    return _video_to_dict(job)


@router.get("/v1/videos/{video_id}")
async def videos_get(video_id: str, request: Request) -> dict[str, Any]:
    job = await resources(request).jobstore.get(video_id)
    if job is None:
        raise JobNotFound(f"Video job {video_id!r} not found")
    return _video_to_dict(job)


@router.delete("/v1/videos/{video_id}")
async def videos_cancel(video_id: str, request: Request) -> dict[str, Any]:
    """Cancel a queued or in-progress video job.

    Returns the job's current state (200) regardless of whether cancellation
    was actually possible. Already-terminal jobs are returned as-is. For
    in-flight jobs we ask the scheduler to cancel the asyncio.Task; the
    runner's CancelledError handler then marks the row failed and releases
    the semaphore permit. If the task can't be found (e.g. the bridge was
    restarted since the job was submitted), we still mark the DB row failed
    so callers see a consistent terminal state.
    """
    res = resources(request)

    job = await res.jobstore.get(video_id)
    if job is None:
        raise JobNotFound(f"Video job {video_id!r} not found")
    if job.status in ("completed", "failed"):
        return _video_to_dict(job)

    # Mark the row failed synchronously so the caller's poll sees a terminal
    # state immediately, but only while it's still active: the runner can
    # complete between the read above and this write, and an unconditional
    # update would flip a finished render to failed and orphan its file.
    await res.jobstore.fail_if_active(video_id, "Cancelled by user")
    res.scheduler.cancel(f"video-job-{video_id}")

    refreshed = await res.jobstore.get(video_id)
    assert refreshed is not None
    return _video_to_dict(refreshed)


@router.get("/v1/videos/{video_id}/content")
async def videos_get_content(
    video_id: str,
    request: Request,
    if_none_match: Annotated[str | None, Header()] = None,
) -> Response:
    res = resources(request)

    job = await res.jobstore.get(video_id)
    if job is None:
        raise JobNotFound(f"Video job {video_id!r} not found")
    if job.status != "completed":
        raise JobNotReady(f"Video job is {job.status}, not completed")
    if not job.file_id:
        raise UpstreamError("Job marked completed but has no file_id")

    opened = await res.filestore.open_for_read(job.file_id)
    if opened is None:
        # The file was evicted before the client fetched it. We could re-pin
        # at completion time to prevent this, but for v1 we accept it.
        raise JobNotFound("Video file no longer available (evicted from cache)")
    return asset_response(opened, if_none_match=if_none_match)
