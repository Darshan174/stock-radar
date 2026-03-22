from __future__ import annotations

import hmac
import os
from fastapi import HTTPException, Request, status

LOCAL_DEV_BACKEND_API_KEY = "stock-radar-local-dev-key"
LOCAL_DEV_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _is_local_request(request: Request) -> bool:
    client_host = request.client.host if request.client else None
    return bool(client_host and client_host in LOCAL_DEV_HOSTS)


def verify_backend_auth(request: Request) -> None:
    expected = os.getenv("BACKEND_API_KEY")
    if not expected and _is_local_request(request):
        # Allow localhost development to work without extra env wiring.
        expected = LOCAL_DEV_BACKEND_API_KEY

    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Backend API key is not configured",
        )

    provided = request.headers.get("X-Backend-Api-Key", "")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid backend API key",
        )
