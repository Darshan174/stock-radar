from __future__ import annotations

import hmac
import os
from fastapi import HTTPException, Request, status


def verify_backend_auth(request: Request) -> None:
    expected = os.getenv("BACKEND_API_KEY")
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
