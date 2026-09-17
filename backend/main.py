"""FastAPI entrypoint for both local summarization strategies."""

from __future__ import annotations

import time
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.summarizers.abstractive import model_status, summarize_abstractive
from backend.summarizers.extractive import summarize_extractive


class SummarizationMethod(str, Enum):
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    method: SummarizationMethod = SummarizationMethod.EXTRACTIVE
    length: int = Field(default=90, ge=20, le=250)

    @field_validator("text")
    @classmethod
    def text_must_contain_words(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be blank")
        return cleaned


class SummaryStats(BaseModel):
    original_words: int
    summary_words: int
    compression_ratio: float
    processing_ms: int


class SummarizeResponse(BaseModel):
    summary: str
    method: SummarizationMethod
    stats: SummaryStats


app = FastAPI(
    title="SummarizeX2",
    version="1.0.0",
    description="Local extractive and abstractive text summarization.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str | dict[str, str | bool]]:
    return {"status": "ok", "abstractive": model_status()}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    """Summarize text and report basic, human-readable compression stats."""

    started = time.perf_counter()
    try:
        if request.method is SummarizationMethod.EXTRACTIVE:
            summary = summarize_extractive(request.text, request.length)
        else:
            summary = summarize_abstractive(request.text, request.length)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    original_words = len(request.text.split())
    summary_words = len(summary.split())
    ratio = summary_words / original_words if original_words else 0.0
    return SummarizeResponse(
        summary=summary,
        method=request.method,
        stats=SummaryStats(
            original_words=original_words,
            summary_words=summary_words,
            compression_ratio=round(ratio, 3),
            processing_ms=round((time.perf_counter() - started) * 1_000),
        ),
    )
