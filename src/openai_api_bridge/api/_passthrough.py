"""Shared prologue for the two endpoints that forward a body unexamined.

``/v1/chat/completions`` and ``/v1/embeddings`` differ only in what they do
with the backend once they have it. Everything before that — decode the body,
insist it's an object, pull ``model`` out to dispatch on, rewrite it to the
upstream's native slug, check the backend implements the operation — was
written out twice, which is two places to keep an error envelope consistent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from fastapi import Request

from ..backends.base import Backend
from ..config import parse_model_id
from ..errors import InvalidRequest, UnsupportedOperation
from ..resources import resources
from ..util.json_body import JsonBody, parse_object


def _overrides(backend: Backend, base_method: Callable[..., Any]) -> bool:
    """Whether ``backend`` actually implements ``base_method``.

    The base ``Backend`` raises ``UnsupportedOperation`` from every optional
    method, so this is an optimisation on the error message rather than a
    correctness check: it lets the caller say which provider was asked and for
    what, instead of the generic text the ABC raises.

    Compares the underlying function against the ABC's, so a subclass that
    hasn't overridden it is still the base implementation.
    """
    return bool(getattr(backend.__class__, base_method.__name__) is not base_method)


class Passthrough(NamedTuple):
    """A resolved passthrough request: where it goes and what to send."""

    backend: Backend
    body: JsonBody
    # Kept for error messages, which name the provider the client actually
    # asked for rather than the rewritten slug.
    provider_id: str


async def prepare(
    request: Request,
    *,
    base_method: Callable[..., Any],
    operation: str,
) -> Passthrough:
    """Resolve the backend for a passthrough request and build its body.

    Body parsing is intentionally minimal: ``model`` is extracted to dispatch
    on and rewritten to the upstream's native slug, and everything else is
    forwarded unchanged — so tools, vision, JSON mode, and any future OpenAI
    request field work without a bridge change.

    ``operation`` is the human-readable name used when the provider doesn't
    support it ("chat completions", "embeddings").
    """
    try:
        body = parse_object(await request.body())
    except ValueError as e:
        raise InvalidRequest(f"Request body must be a JSON object: {e}") from e

    model = body.fields.get("model")
    if not isinstance(model, str) or not model:
        raise InvalidRequest("Missing or invalid 'model' field", param="model")

    provider_id, model_slug = parse_model_id(model)
    backend = resources(request).dispatcher.for_provider(provider_id)

    if not _overrides(backend, base_method):
        raise UnsupportedOperation(f"Provider {provider_id!r} does not support {operation}")

    # Rewrite the model id so the upstream sees its native slug, not our
    # prefixed form. Spliced into the caller's bytes rather than re-encoding
    # the whole body for one string; see util.json_body.
    forwarded = body.replace_top_level({"model": model_slug})
    return Passthrough(backend=backend, body=forwarded, provider_id=provider_id)
