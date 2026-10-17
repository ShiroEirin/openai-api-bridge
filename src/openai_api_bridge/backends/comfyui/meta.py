"""The ``.meta.json`` schema, checked at scan time in warn-only mode.

A workflow's meta file is read as a plain dict, and a mistake in it has always
been silent: a typo'd key is ignored, a field of the wrong type is skipped or
misread (``"multiple": "false"`` is truthy), and the only symptom is a render
that quietly ignores part of the request.

:func:`check_meta` validates the meta against the documented schema and logs
one warning per problem, naming the file and the field. It never rejects
anything: the workflow loads exactly as it did before. That gives operators a
release to fix their files before the schema is enforced.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

log = logging.getLogger(__name__)


def _node_id(value: Any) -> Any:
    # A graph keys its nodes by string, and an integer id is accepted (and
    # normalised at scan time). bool is an int subclass but never a node id.
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else value


NodeId = Annotated[str, BeforeValidator(_node_id)]


class _Strict(BaseModel):
    # strict: "false" is not a bool and "24" is not a number. forbid: an
    # unknown key is almost always a typo of a real one.
    model_config = ConfigDict(strict=True, extra="forbid")


class ImageInput(_Strict):
    node: NodeId
    field: str
    format: Literal["filename", "list"] = "filename"
    multiple: bool = False


class WorkflowMeta(_Strict):
    """The documented ``.meta.json`` fields (README, "Workflow meta.json schema")."""

    positive_prompt_node: NodeId
    positive_prompt_field: str = "text"
    display_name: str | None = None
    image_inputs: list[ImageInput] = Field(default_factory=list)
    image_required: bool = False
    dimensions_node: NodeId | None = None
    width_field: str = "width"
    height_field: str = "height"
    aspect_ratio_node: NodeId | None = None
    aspect_ratio_field: str = "aspect_ratio"
    aspect_ratios: list[str] | None = None
    length_node: NodeId | None = None
    length_field: str = "value"
    fps: float | None = Field(default=None, gt=0)
    seed_nodes: list[NodeId] | None = None
    output_type: Literal["image", "video"] | None = None
    prompt_style: str | None = None
    prompt_hint: str | None = None


def check_meta(meta: dict[str, Any], filename: str) -> list[str]:
    """Log each way ``meta`` departs from the schema; return the messages.

    Warn-only: the caller loads the workflow regardless.
    """
    try:
        WorkflowMeta.model_validate(meta)
    except ValidationError as e:
        problems = [_describe(err) for err in e.errors()]
    else:
        return []
    for problem in problems:
        log.warning(
            "%s: %s. Loaded anyway for now; a future release will reject this "
            "workflow until the meta is fixed.",
            filename,
            problem,
        )
    return problems


def _describe(err: Any) -> str:
    loc = ".".join(str(part) for part in err.get("loc", ())) or "(meta)"
    if err.get("type") == "extra_forbidden":
        return f"unknown meta field {loc!r} (a typo?)"
    if err.get("type") == "missing":
        return f"meta field {loc!r} is required"
    return f"meta field {loc!r}: {err.get('msg')} (got {err.get('input')!r})"
