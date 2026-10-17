"""End-to-end endpoint tests via TestClient.

Empty-providers config — covers auth, validation, parse_model_id error mapping,
and 404 paths. Backend-success paths are covered separately with respx stubs.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_auth_required(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_auth_rejects_non_ascii_key_as_401(client: TestClient) -> None:
    """A non-ASCII bearer token is a bad key, not a server fault.

    hmac.compare_digest raises TypeError on a str carrying non-ASCII, so
    comparing without encoding turned this into a 500 (and a logged
    traceback) on a path any unauthenticated caller can reach.
    """
    r = client.get("/v1/models", headers={b"Authorization": "Bearer é".encode()})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_auth_rejects_wrong_key(client: TestClient) -> None:
    r = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_routes_absent_by_default(
    client: TestClient, auth_headers: dict[str, str], path: str
) -> None:
    """FastAPI mounts these outside any route dependency, so they once
    answered unauthenticated and published the whole API surface to anyone
    who could reach the port. Off unless BRIDGE_ENABLE_DOCS says otherwise,
    and behind the key like everything else when off."""
    assert client.get(path).status_code == 401
    assert client.get(path, headers=auth_headers).status_code == 404


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_routes_served_when_opted_in(docs_client: TestClient, path: str) -> None:
    assert docs_client.get(path).status_code == 200


def test_every_route_requires_the_key(client: TestClient) -> None:
    """The middleware is the only auth gate, so a route added later is
    covered without anyone remembering to opt it in."""
    from fastapi.routing import APIRoute

    from openai_api_bridge.api import chat, embeddings, files, images, models, videos

    routers = (chat, embeddings, files, images, models, videos)
    routes = [r for m in routers for r in m.router.routes if isinstance(r, APIRoute)]
    assert len(routes) >= 10
    for route in routes:
        path = route.path.replace("{video_id}", "video_x").replace("{file_id}", "file_x")
        for method in route.methods or ():
            r = client.request(method, path)
            assert r.status_code == 401, (method, path)
            assert r.json()["error"]["code"] == "invalid_api_key"
            assert r.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    ("path", "kwargs"),
    [
        ("/v1/images/generations", {"json": {"model": "x/y", "prompt": "p"}}),
        (
            "/v1/images/edits",
            {"data": {"model": "x/y", "prompt": "p"}, "files": {"image": ("a.png", b"x")}},
        ),
        (
            "/v1/videos",
            {
                "data": {"model": "x/y", "prompt": "p"},
                "files": {"input_reference": ("a.png", b"x")},
            },
        ),
        ("/v1/chat/completions", {"json": {"model": "x/y", "messages": []}}),
    ],
)
def test_unauthenticated_body_is_never_read(
    client: TestClient, path: str, kwargs: dict[str, object]
) -> None:
    """FastAPI parses body parameters before it solves dependencies, so when
    auth was a dependency these routes decoded JSON and multipart uploads for
    callers with no key. The gate now runs before the body is pulled."""
    from unittest.mock import patch

    from starlette.requests import Request

    with (
        patch.object(Request, "body", side_effect=AssertionError("body read")),
        patch.object(Request, "form", side_effect=AssertionError("form read")),
    ):
        r = client.post(path, **kwargs)  # type: ignore[arg-type]
    assert r.status_code == 401


def test_models_lists_empty(client: TestClient, auth_headers: dict[str, str]) -> None:
    r = client.get("/v1/models", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == {"object": "list", "data": []}


def test_images_generations_malformed_model(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    r = client.post(
        "/v1/images/generations",
        headers=auth_headers,
        json={"model": "no-slash", "prompt": "x"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"
    assert r.json()["error"]["param"] == "model"


def test_images_generations_unknown_provider(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    r = client.post(
        "/v1/images/generations",
        headers=auth_headers,
        json={"model": "ghost/x", "prompt": "x"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "provider_not_found"


def test_images_generations_n_too_large_via_pydantic(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    r = client.post(
        "/v1/images/generations",
        headers=auth_headers,
        json={"model": "ghost/x", "prompt": "x", "n": 10},
    )
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["type"] == "invalid_request_error"
    assert "n" in body.get("param", "")


def test_images_edits_empty_image(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    r = client.post(
        "/v1/images/edits",
        headers=auth_headers,
        files={"image": ("x.png", b"", "image/png")},
        data={"model": "ghost/x", "prompt": "x"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_images_edits_too_many_images(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    r = client.post(
        "/v1/images/edits",
        headers=auth_headers,
        files=[("image", (f"{i}.png", b"x", "image/png")) for i in range(17)],
        data={"model": "ghost/x", "prompt": "x"},
    )
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["code"] == "invalid_request"
    assert "image" in body.get("param", "")


def test_videos_get_nonexistent(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    r = client.get("/v1/videos/abcdef", headers=auth_headers)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_videos_content_nonexistent(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    r = client.get("/v1/videos/abcdef/content", headers=auth_headers)
    assert r.status_code == 404


def test_files_content_nonexistent(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    r = client.get("/v1/files/abcdef/content", headers=auth_headers)
    assert r.status_code == 404


def test_images_generations_rejects_an_oversized_size(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    r = client.post(
        "/v1/images/generations",
        headers=auth_headers,
        json={"model": "ghost/x", "prompt": "x", "size": "100000x100000"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "size"


def test_images_edits_rejects_an_oversized_size(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    r = client.post(
        "/v1/images/edits",
        headers=auth_headers,
        data={"model": "ghost/x", "prompt": "x", "size": "512x100000"},
        files={"image": ("a.png", b"x")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "size"
