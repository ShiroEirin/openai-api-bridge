"""Backends' upstream clients keep idle connections long enough to reuse them.

httpx2's default keepalive is 5s, shorter than the gap between almost any two
requests a backend makes, so at the default each request paid a fresh TCP and
TLS handshake. Every backend builds its clients through ``upstream_client``.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx2

import openai_api_bridge
from openai_api_bridge.util.http import upstream_client

_SRC = Path(openai_api_bridge.__file__).parent


def test_no_backend_builds_its_own_client() -> None:
    offenders = [
        str(path.relative_to(_SRC))
        for path in (_SRC / "backends").rglob("*.py")
        if re.search(r"httpx2\.AsyncClient\(", path.read_text())
    ]
    assert offenders == [], "build backend clients with util.http.upstream_client"


async def test_upstream_client_outlives_the_default_keepalive() -> None:
    client = upstream_client(timeout=httpx2.Timeout(5.0), headers={"x": "y"})
    try:
        # httpcore2's pool keeps the setting privately; there is no public view.
        pool = client._transport._pool  # type: ignore[attr-defined]
        assert pool._keepalive_expiry == 60.0
        assert client.headers["x"] == "y"
    finally:
        await client.aclose()
