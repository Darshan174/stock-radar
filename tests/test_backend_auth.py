from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend.auth import LOCAL_DEV_BACKEND_API_KEY, verify_backend_auth


def _request(api_key: str, client_host: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/test",
            "headers": [(b"x-backend-api-key", api_key.encode("utf-8"))],
            "client": (client_host, 12345),
        }
    )


def test_verify_backend_auth_accepts_local_dev_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BACKEND_API_KEY", raising=False)

    verify_backend_auth(_request(LOCAL_DEV_BACKEND_API_KEY, "127.0.0.1"))


def test_verify_backend_auth_requires_explicit_key_for_non_local_requests(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("BACKEND_API_KEY", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        verify_backend_auth(_request(LOCAL_DEV_BACKEND_API_KEY, "10.0.0.5"))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Backend API key is not configured"
