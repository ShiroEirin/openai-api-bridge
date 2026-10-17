"""OpenAI-shaped request models (responses are built as plain dicts).

We intentionally accept-and-ignore extra fields the OpenAI API supports but the
bridge has no use for (``user``, ``quality``, ``style``, etc.) so well-behaved
clients don't error out. Validation happens at the field level — invalid
``size`` formats fail fast with a 400.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Permissive(BaseModel):
    """Accept and silently drop unknown fields."""

    model_config = ConfigDict(extra="ignore")


# --- /v1/images/generations -------------------------------------------------

# Ceiling on images per request. Lives here rather than beside the endpoint
# because this model is what enforces it on the JSON path — but the multipart
# edits path can't use a pydantic model (it binds Form fields), so it checks
# the same value by hand. Two enforcement points, one number: they disagreed
# silently when it was written out twice.
MAX_IMAGES_PER_REQUEST = 4


class ImagesGenerationRequest(_Permissive):
    model: str
    prompt: str
    n: int = Field(default=1, ge=1, le=MAX_IMAGES_PER_REQUEST)
    size: str | None = None
    # Non-standard, and declared rather than left to `extra` because this class
    # *ignores* extras — an undeclared field would vanish here rather than reach
    # the backend. The canonical ratio a client picked from the model's
    # advertised `aspect_ratios` ("16:9"). Unvalidated on purpose: only the
    # backend knows what a given model offers, and it snaps to the nearest
    # rather than rejecting.
    aspect_ratio: str | None = None
    response_format: Literal["url", "b64_json"] = "url"
