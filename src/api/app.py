"""FastAPI application exposing the Bedrock-powered RAG pipeline."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.rag.pipeline import RAGPipeline

app = FastAPI(title="Bedrock RAG Service", version="1.0.0")


class HistoryTurn(BaseModel):
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str


class QueryRequest(BaseModel):
    question: str
    history: Optional[List[HistoryTurn]] = Field(
        default=None,
        description="Conversation history alternating user/assistant turns",
    )


class ContextChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    question: str
    rewritten_question: str
    answer: str
    contexts: List[ContextChunk]
    query_variants: List[str]
    hyde_document: Optional[str]


pipeline: RAGPipeline | None = None
settings: Settings | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global pipeline, settings
    settings = get_settings()
    pipeline = RAGPipeline(settings=settings)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    history_payload = [turn.dict() for turn in request.history] if request.history else None

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pipeline.run(request.question, history_payload),
    )
    return QueryResponse(**result)


__all__ = ["app"]

