"""``POST /v1/images/generations`` and ``POST /v1/images/edits``."""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, UploadFile

from ..backends.base import GeneratedAsset, InputImage
from ..config import BridgeSettings, parse_model_id
from ..errors import InvalidRequest
from ..infra.filestore import FileStore
from ..resources import resources
from ..schemas.openai import MAX_IMAGES_PER_REQUEST, ImagesGenerationRequest
from ..util.concurrency import run_all
from ..util.sizes import check_size_bounds

router = APIRouter()

# Cap on reference images per edit. Matches OpenAI's /images/edits limit and
# bounds memory on backends that buffer every image (OpenRouter base64-expands
# them); edits are authenticated, so this is a sanity guard, not a DoS control.
_MAX_EDIT_IMAGES = 16


def _build_url(settings: BridgeSettings, file_id: str) -> str:
    """Construct the URL we return to clients for a stored asset."""
    if settings.public_base_url:
        return f"{settings.public_base_url.rstrip('/')}/v1/files/{file_id}/content"
    return f"/v1/files/{file_id}/content"


async def _render_assets(
    assets: list[GeneratedAsset],
    *,
    response_format: str,
    filestore: FileStore,
    settings: BridgeSettings,
    provider_id: str,
    model_slug: str,
    prompt: str,
) -> list[dict[str, Any]]:

    def _extra(asset: GeneratedAsset) -> dict[str, Any]:
        # Non-standard, omitted when the backend didn't say. The ratio actually
        # rendered, which snapping means need not be the one requested.
        return {"aspect_ratio": asset.aspect_ratio} if asset.aspect_ratio else {}

    if response_format == "b64_json":
        return [
            {"b64_json": base64.b64encode(asset.data).decode("ascii"), **_extra(asset)}
            for asset in assets
        ]

    def _store(asset: GeneratedAsset) -> Callable[[], Coroutine[Any, Any, str]]:
        return lambda: filestore.put(
            asset.data,
            content_type=asset.content_type,
            kind=asset.kind,
            source_backend=provider_id,
            source_model=model_slug,
            prompt_excerpt=prompt,
        )

    # Concurrently: each put writes off the event loop, so n images' disk
    # writes overlap instead of the response waiting on their sum.
    file_ids = await run_all([_store(asset) for asset in assets])
    return [
        {"url": _build_url(settings, file_id), **_extra(asset)}
        for asset, file_id in zip(assets, file_ids, strict=True)
    ]


@router.post("/v1/images/generations")
async def images_generations(req: ImagesGenerationRequest, request: Request) -> dict[str, Any]:
    check_size_bounds(req.size)
    provider_id, model_slug = parse_model_id(req.model)
    res = resources(request)
    backend = res.dispatcher.for_provider(provider_id)

    assets = await backend.generate_image(
        model_slug=model_slug,
        prompt=req.prompt,
        size=req.size,
        aspect_ratio=req.aspect_ratio,
        n=req.n,
    )

    data = await _render_assets(
        assets,
        response_format=req.response_format,
        filestore=res.filestore,
        settings=res.settings,
        provider_id=provider_id,
        model_slug=model_slug,
        prompt=req.prompt,
    )
    return {"created": int(time.time()), "data": data}


@router.post("/v1/images/edits")
async def images_edits(
    request: Request,
    prompt: Annotated[str, Form()],
    model: Annotated[str, Form()],
    # Accept both the OpenAI single-field ``image`` (which some clients
    # repeat to send multiples) and the ``image[]`` array convention. A
    # single declared ``UploadFile`` silently keeps only the last of a
    # repeated field, which is what dropped all-but-one reference image.
    image: Annotated[
        list[UploadFile] | None, File(description="Source image(s) (multipart upload)")
    ] = None,
    image_array: Annotated[list[UploadFile] | None, File(alias="image[]")] = None,
    n: Annotated[int, Form()] = 1,
    size: Annotated[str | None, Form()] = None,
    aspect_ratio: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "url",
) -> dict[str, Any]:
    # The same ceiling ImagesGenerationRequest applies on the JSON path, which
    # this endpoint can't reuse: its fields are multipart Form bindings, not a
    # pydantic body model.
    if not 1 <= n <= MAX_IMAGES_PER_REQUEST:
        raise InvalidRequest(
            f"n must be between 1 and {MAX_IMAGES_PER_REQUEST} (got {n})", param="n"
        )
    check_size_bounds(size)
    if response_format not in ("url", "b64_json"):
        raise InvalidRequest(
            f"response_format must be 'url' or 'b64_json' (got {response_format!r})",
            param="response_format",
        )

    # Preserve client order: ``image`` entries first, then any ``image[]``.
    uploads = [*(image or []), *(image_array or [])]
    if not uploads:
        raise InvalidRequest("At least one input image is required", param="image")
    if len(uploads) > _MAX_EDIT_IMAGES:
        raise InvalidRequest(
            f"At most {_MAX_EDIT_IMAGES} input images are allowed (got {len(uploads)})",
            param="image",
        )

    # Validate each upload first so an empty file surfaces as a 400 rather
    # than getting masked by a downstream provider/model 404.
    images: list[InputImage] = []
    for upload in uploads:
        raw = await upload.read()
        if not raw:
            raise InvalidRequest("Uploaded image is empty", param="image")
        images.append(InputImage(data=raw, content_type=upload.content_type or "image/png"))

    provider_id, model_slug = parse_model_id(model)
    res = resources(request)
    backend = res.dispatcher.for_provider(provider_id)

    assets = await backend.edit_image(
        model_slug=model_slug,
        prompt=prompt,
        images=images,
        size=size,
        aspect_ratio=aspect_ratio,
        n=n,
    )

    data = await _render_assets(
        assets,
        response_format=response_format,
        filestore=res.filestore,
        settings=res.settings,
        provider_id=provider_id,
        model_slug=model_slug,
        prompt=prompt,
    )
    return {"created": int(time.time()), "data": data}
