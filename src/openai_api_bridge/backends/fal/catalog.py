"""fal's model catalogue, translated into the bridge's model listing.

Pure functions over the raw catalogue the client fetches: which categories the
backend serves, how text-driven models pair with their reference-image halves,
and the ``ModelEntry`` list ``/v1/models`` shows. ``FalBackend`` owns the
caching and the network; nothing here does I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ...config import FalModelConfig
from ...kinds import AssetKind
from ..base import ModelEntry, disambiguate_display_names

log = logging.getLogger(__name__)

# fal categories this backend serves, mapped to the bridge's `kind` hint. The
# audio and 3d categories fal also publishes stay out: there's no code path for
# them, so listing them would advertise models every request would fail on.
CATEGORY_KINDS: dict[str, AssetKind] = {
    "text-to-image": "image",
    "image-to-image": "image",
    "text-to-video": "video",
    "image-to-video": "video",
}
SUPPORTED_CATEGORIES: tuple[str, ...] = tuple(CATEGORY_KINDS)

# fal publishes a model's text-driven and reference-image-driven halves as
# separate endpoints, which is an easy trap: a client sends an edit to
# `fal-ai/nano-banana-2` without knowing `/edit` exists, or a still to
# `fal-ai/veo3.1` without knowing `/image-to-video` does. Where we can pair the
# two *confidently*, the bridge lists one model and routes by request shape.
#
# "Confidently" means both ids exist in the catalogue and sit in the expected
# categories — never a guess from a name. Against the live catalogue this pairs
# ~81 of 194 text-to-image models and ~74 of 125 text-to-video ones; anything
# unpaired, and every reference-only model (inpainting, upscalers, background
# removal), is left exactly as it is.
#
# text-driven category -> (its reference-image counterpart, endpoint suffixes)
_VARIANT_PAIRS: dict[str, tuple[str, tuple[str, ...]]] = {
    "text-to-image": ("image-to-image", ("/edit", "/image-to-image", "/edit-image")),
    "text-to-video": ("image-to-video", ("/image-to-video", "/edit")),
}


def variant_candidates(text_model_id: str, category: str) -> list[str]:
    """Ids that would be the reference-image half of ``text_model_id``.

    Two shapes occur: a bare base gaining a suffix
    (``fal-ai/nano-banana-2`` -> ``fal-ai/nano-banana-2/edit``), and sibling
    suffixes under a shared stem
    (``bytedance/seedream/v5/pro/text-to-image`` -> ``.../pro/edit``).
    """
    pair = _VARIANT_PAIRS.get(category)
    if pair is None:
        return []
    _, suffixes = pair
    text_suffix = f"/{category}"
    stem = (
        text_model_id[: -len(text_suffix)] if text_model_id.endswith(text_suffix) else text_model_id
    )
    candidates = [stem + suffix for suffix in suffixes]
    if stem != text_model_id:
        # A stem-suffixed model can also pair with `<full id>/edit`.
        candidates += [text_model_id + suffix for suffix in suffixes]
    return candidates


def entries_from_catalog(
    raw: list[dict[str, Any]],
    *,
    models: Mapping[str, FalModelConfig],
    collapse_variants: bool,
) -> tuple[list[ModelEntry], dict[str, str]]:
    """Translate a raw catalogue into entries and the variant routing map.

    Returns the routes rather than assigning them, so the caller can
    publish both together only once the result is known good. Assigning
    here meant a fetch that succeeded at the HTTP level but yielded nothing
    usable — fal answering ``{"models": []}`` during an incident, a
    category rename — replaced a working routing map with an empty one on
    its way to reporting failure. Harmless while the catalogue was fetched
    once per process; not once it refreshes on a TTL.
    """
    by_id = {
        item["endpoint_id"]: item
        for item in raw
        if isinstance(item.get("endpoint_id"), str) and item["endpoint_id"]
    }
    variant_routes, collapsed = pair_variants(by_id, collapse_variants=collapse_variants)

    entries: list[ModelEntry] = []
    seen: set[str] = set()
    pinned: set[str] = set()
    for item in raw:
        model_id = item.get("endpoint_id")
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            continue
        if model_id in collapsed:
            # Folded into its text-driven half, which routes here when a
            # reference image is supplied. Listing both is what invites
            # picking the wrong one.
            continue
        seen.add(model_id)
        meta = item.get("metadata")
        meta = meta if isinstance(meta, dict) else {}
        override = models.get(model_id)
        display = None
        if override is not None and override.display_name:
            display = override.display_name
            pinned.add(model_id)
        elif isinstance(meta.get("display_name"), str):
            display = meta["display_name"]
        category = meta.get("category")
        kind = CATEGORY_KINDS.get(category) if isinstance(category, str) else None
        # What this entry actually accepts. A collapsed model covers both
        # halves; without this the merged id would be indistinguishable
        # from a text-only one, and a client would find out by failing.
        # Only categories the backend knows are published: they double as
        # the capability strings, so an operator pointing `categories` at
        # something outside CATEGORY_KINDS would otherwise emit a value
        # that isn't in the documented {input}-to-{output} vocabulary.
        capabilities: tuple[str, ...] | None = None
        if isinstance(category, str) and category in CATEGORY_KINDS:
            routed = variant_routes.get(model_id)
            partner = category_of(by_id, routed) if routed else None
            capabilities = (
                (category, partner) if partner in CATEGORY_KINDS and partner else (category,)
            )
        entries.append(
            ModelEntry(
                id=model_id,
                # The bridge picks the code path from the request shape
                # (POST /v1/images vs /v1/videos), so this is just a hint.
                kind=kind or "image",
                display_name=display or model_id,
                prompt_style=override.prompt_style if override else None,
                prompt_hint=override.prompt_hint if override else None,
                capabilities=capabilities,
            )
        )
    # fal titles an endpoint after its model family, so multi-endpoint
    # families arrive as a run of identically-named rows.
    return disambiguate_display_names(entries, pinned=frozenset(pinned)), variant_routes


def category_of(by_id: dict[str, dict[str, Any]], model_id: str) -> str | None:
    meta = by_id.get(model_id, {}).get("metadata")
    category = meta.get("category") if isinstance(meta, dict) else None
    return category if isinstance(category, str) else None


def pair_variants(
    by_id: dict[str, dict[str, Any]], *, collapse_variants: bool
) -> tuple[dict[str, str], set[str]]:
    """Match text-driven models to their reference-image half.

    Returns ``(routes, collapsed)`` — where to send a request carrying a
    reference image, and which ids to drop from the listing because they're
    now reachable through their sibling.

    A pair is only made when both ids are in this catalogue *and* their
    categories are the expected halves, so nothing is inferred from a name
    alone. Models without a partner — including every reference-only
    endpoint — are untouched.
    """
    if not collapse_variants:
        return {}, set()
    routes: dict[str, str] = {}
    collapsed: set[str] = set()
    for model_id in by_id:
        category = category_of(by_id, model_id)
        pair = _VARIANT_PAIRS.get(category) if category else None
        if category is None or pair is None:
            continue
        partner_category, _ = pair
        for candidate in variant_candidates(model_id, category):
            if candidate in by_id and category_of(by_id, candidate) == partner_category:
                routes[model_id] = candidate
                collapsed.add(candidate)
                break
    if routes:
        log.debug("fal: collapsed %d variant pairs into their base models", len(routes))
    return routes, collapsed
