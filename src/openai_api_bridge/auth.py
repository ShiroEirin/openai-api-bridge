"""Bearer-token auth, enforced as the outermost ASGI middleware.

Single shared API key model: clients send ``Authorization: Bearer <BRIDGE_API_KEY>``.
Constant-time comparison to prevent timing attacks on key recovery.

A middleware rather than a FastAPI dependency, because FastAPI reads and
validates a route's body parameters (pydantic models, ``Form``, ``File``)
*before* it solves dependencies. As a dependency, the check ran only after
``/v1/images/edits`` and ``/v1/videos`` had parsed a multipart upload — and
``/v1/images/generations`` its JSON body — for a caller holding no key at all.
Out here it runs before a single body byte is pulled.

It is the bridge's only auth gate: every route requires the key except the
interactive docs, which exist only when ``BRIDGE_ENABLE_DOCS`` opts them in.
"""

from __future__ import annotations

import hmac
from collections.abc import Collection

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .errors import Unauthorized, error_payload

_BEARER = "Bearer "


def check_bearer(authorization: str | None, api_key: str) -> str | None:
    """Return why ``authorization`` fails to carry ``api_key``, or None if it does."""
    if not authorization or not authorization.startswith(_BEARER):
        return "Missing or malformed Authorization header"
    presented = authorization[len(_BEARER) :]
    # Compare as bytes: hmac.compare_digest raises TypeError on a str holding
    # non-ASCII, which would surface a bad key as a 500 (plus a traceback in
    # the log) instead of a 401 — and that path is reachable pre-auth by
    # anyone who can reach the port.
    if not hmac.compare_digest(presented.encode("utf-8"), api_key.encode("utf-8")):
        return "Invalid API key"
    return None


class ApiKeyMiddleware:
    """Reject any HTTP request without the bridge's bearer token.

    ``exempt_paths`` are matched exactly; they are the docs routes when those
    are enabled, and nothing otherwise.
    """

    def __init__(self, app: ASGIApp, *, api_key: str, exempt_paths: Collection[str] = ()) -> None:
        self.app = app
        self.api_key = api_key
        self.exempt_paths = frozenset(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self.exempt_paths:
            await self.app(scope, receive, send)
            return
        # Headers decodes as latin-1, so a non-ASCII token arrives here as a
        # str that check_bearer compares as bytes rather than raising on.
        reason = check_bearer(Headers(scope=scope).get("authorization"), self.api_key)
        if reason is None:
            await self.app(scope, receive, send)
            return
        response = JSONResponse(
            status_code=Unauthorized.status_code,
            content=error_payload(
                message=reason,
                type_=Unauthorized.error_type,
                code=Unauthorized.code,
            ),
            headers={"www-authenticate": "Bearer"},
        )
        await response(scope, receive, send)
