"""FastAPI service: /healthz, /search and /answer over a documentation index."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import corpus
from .answer import answer
from .index import RETRIEVERS, Index


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    retriever: str = "hybrid"


def create_app(index: Index | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.index = index or Index(corpus.load_docs(Path(os.getenv("DOCS_DIR", str(corpus.DEFAULT_DIR)))))
        yield

    app = FastAPI(title="rag-eval-kit", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        idx: Index = app.state.index
        return {"status": "ok", "documents": idx.n_docs, "chunks": len(idx.chunks)}

    @app.post("/search")
    def search(req: SearchRequest):
        if req.retriever not in RETRIEVERS:
            raise HTTPException(422, f"retriever must be one of {RETRIEVERS}")
        hits = app.state.index.search_docs(req.query, retriever=req.retriever, k=req.k)
        return {
            "results": [
                {"doc_id": h.doc_id, "rank": i, "heading": h.chunk.heading, "snippet": h.chunk.text[:300]}
                for i, h in enumerate(hits, start=1)
            ]
        }

    @app.post("/answer")
    def answer_endpoint(req: SearchRequest):
        hits = app.state.index.search_docs(req.query, retriever=req.retriever, k=req.k)
        return answer(req.query, hits)

    return app


def __getattr__(name: str):  # `uvicorn rag_eval.api:app` builds the app lazily so importing the module stays cheap
    if name == "app":
        return create_app()
    raise AttributeError(name)
