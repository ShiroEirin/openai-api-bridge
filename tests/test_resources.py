"""The resource graph's accessor, including what it says when it's not there."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from openai_api_bridge.resources import BridgeResources, install, resources


def _fake_request() -> Any:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


def test_install_then_read_round_trips() -> None:
    request = _fake_request()
    graph = cast(BridgeResources, SimpleNamespace(dispatcher="d", settings="s"))
    install(request.app, graph)
    assert resources(request) is graph


def test_missing_resources_names_the_lifespan() -> None:
    """The whole point of routing every read through one accessor: when the
    graph isn't installed, the failure can say why once instead of surfacing
    as an AttributeError naming whichever attribute the route reached for."""
    with pytest.raises(RuntimeError, match="lifespan has not run"):
        resources(_fake_request())


def test_resources_are_frozen() -> None:
    """Nothing should swap a live dispatcher out from under an in-flight
    request, so the container refuses assignment rather than relying on
    convention."""
    fields = BridgeResources.__dataclass_fields__
    graph = BridgeResources(**dict.fromkeys(fields, None))  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        graph.dispatcher = graph.dispatcher  # type: ignore[misc]
