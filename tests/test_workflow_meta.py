"""``.meta.json`` schema checks: warned about at scan time, never enforced (yet).

A meta mistake used to be silent — a typo'd key ignored, a wrong-typed value
misread — so these pin that each kind of mistake is named in the log, and that
the workflow still loads exactly as before.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from openai_api_bridge.backends.comfyui.meta import check_meta
from openai_api_bridge.backends.comfyui.workflows import scan_workflows

_FULL: dict[str, Any] = {
    "positive_prompt_node": "10",
    "positive_prompt_field": "text",
    "display_name": "Everything",
    "image_inputs": [{"node": 7, "field": "image", "format": "list", "multiple": True}],
    "image_required": True,
    "dimensions_node": "5",
    "width_field": "width",
    "height_field": "height",
    "aspect_ratio_node": "115",
    "aspect_ratio_field": "aspect_ratio",
    "aspect_ratios": ["1:1 (Square)", "16:9 (Widescreen)"],
    "length_node": 8,
    "length_field": "value",
    "fps": 24,
    "seed_nodes": ["3", 4],
    "output_type": "video",
    "prompt_style": "cinematic-prose",
    "prompt_hint": "describe camera motion",
}


def test_every_documented_field_passes() -> None:
    assert check_meta(_FULL, "full.meta.json") == []
    assert check_meta({"positive_prompt_node": 6}, "min.meta.json") == []


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        (
            {"positive_prompt_node": "1", "positve_prompt_field": "t"},
            "unknown meta field 'positve_prompt_field'",
        ),
        (
            {"positive_prompt_node": "1", "image_inputs": [{"node": "2"}]},
            "'image_inputs.0.field' is required",
        ),
        (
            {
                "positive_prompt_node": "1",
                "image_inputs": [{"node": "2", "field": "image", "multiple": "false"}],
            },
            "'image_inputs.0.multiple'",
        ),
        (
            {
                "positive_prompt_node": "1",
                "image_inputs": [{"node": "2", "field": "image", "format": "csv"}],
            },
            "'image_inputs.0.format'",
        ),
        ({"positive_prompt_node": "1", "fps": "24"}, "'fps'"),
        ({"positive_prompt_node": "1", "fps": 0}, "'fps'"),
        ({"positive_prompt_node": "1", "output_type": "audio"}, "'output_type'"),
        ({"positive_prompt_node": True}, "'positive_prompt_node'"),
        ({"positive_prompt_node": "1", "seed_nodes": "3"}, "'seed_nodes'"),
        ({"display_name": "x"}, "'positive_prompt_node' is required"),
    ],
)
def test_each_mistake_is_named(
    meta: dict[str, Any], expected: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        problems = check_meta(meta, "w.meta.json")
    assert any(expected in p for p in problems), problems
    assert any(
        r.message.startswith("w.meta.json: ") and "future release will reject" in r.message
        for r in caplog.records
    )


def test_a_flawed_meta_still_loads(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Warn-only: the workflow is served exactly as before the check existed."""
    (tmp_path / "w.json").write_text(
        json.dumps({"1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}})
    )
    (tmp_path / "w.meta.json").write_text(
        json.dumps({"positive_prompt_node": "1", "dispaly_name": "Typo"})
    )
    with caplog.at_level(logging.WARNING):
        records = scan_workflows(tmp_path)
    assert "w" in records
    assert any("unknown meta field 'dispaly_name'" in r.message for r in caplog.records)
