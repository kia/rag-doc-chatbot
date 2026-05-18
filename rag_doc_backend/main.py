import logging

logger = logging.getLogger("uvicorn.error")
from typing import List

from rag_doc_backend.estimation.tocken_estimate import estimate_query_cost
from rag_doc_backend.model.open_ai import get_open_ai_models
from rag_doc_backend.model.ollama import get_local_ollama_models
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .rag_engine import RAGEngine

app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

class Source(BaseModel):
    source: str
    chunk_preview: str
    page: int
    search_text: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    total_sources: int
    token_usage: dict
    quality_metrics: dict

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model_name: str
    model_family: str = "ollama"
    k: int = 5

class EstimateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model_name: str
    model_family: str = "ollama"
    context: str = ""
    k: int = 5

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/v1/model_names/ollama", response_model=list[str])
def get_available_ollama_models():
    return get_local_ollama_models()

@app.get("/api/v1/model_names/openai", response_model=list[str])
def get_available_openai_models():
    return get_open_ai_models()

@app.get("/api/v1/init", response_model=dict)
def init(model_name: str, model_family: str = "ollama", k: int = 5):
    logger.debug("set model " + model_name)

    def status_callback(msg):
        logger.info("Status = {}".format(msg))

    rag_engine = RAGEngine(model_name=model_name, model_family=model_family, status_callback=status_callback, k=k)
    logger.debug("init " + rag_engine.model_name)
    rag_engine.init()
    return {"status": "success"}

@app.get("/api/v1/vectorstore/reset", response_model=dict)
def reset_vectorstore(model_name: str, model_family: str, k: int = 5):
    logger.debug("set model " + model_name)
    rag_engine = RAGEngine(model_name=model_name, model_family=model_family, k=k)
    logger.debug("reset vectorstore " + rag_engine.model_name)
    rag_engine.reset_vectorstore()
    return {"status": "success"}

@app.post("/api/v1/estimate", response_model=dict)
def estimate(request: EstimateRequest):
    rag_engine = RAGEngine(model_name=request.model_name, model_family=request.model_family, k=request.k)
    return estimate_query_cost(request.query, request.model_name, rag_engine.vectorstore, context=request.context, estimated_output=100)

@app.post("/api/v1/query", response_model=dict)
def query(request: QueryRequest):
    try:
        rag_engine = RAGEngine(model_name=request.model_name, model_family=request.model_family, k = request.k)
        response = rag_engine.query(request.query)
        return response
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
