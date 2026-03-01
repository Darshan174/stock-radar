from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeJobCreateRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    mode: Literal["intraday", "longterm"] = "intraday"
    period: str = "max"


class AnalyzeJobCreated(BaseModel):
    jobId: str
    statusUrl: str
    status: Literal["queued", "running", "succeeded", "failed"]


class AnalyzeJobStatus(BaseModel):
    jobId: str
    status: Literal["queued", "running", "succeeded", "failed"]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    symbol: Optional[str] = Field(default=None, max_length=20)
    sessionId: Optional[str] = Field(default=None, max_length=128)


class AskResponse(BaseModel):
    answer: str
    stockSymbols: list[str]
    sourcesUsed: list[dict[str, Any]]
    modelUsed: str
    tokensUsed: int
    processingTimeMs: int
    sessionId: str
    contextRetrieved: dict[str, Any]
