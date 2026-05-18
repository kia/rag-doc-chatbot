import time
import logging
logger = logging.getLogger("uvicorn.error")

from rag_doc_backend.generation.build_prompt import build_query_prompt
from rag_doc_backend.metrics.quality_metrics import build_quality_metrics
from rag_doc_backend.retrieval.embed_query import run_embedded_query


def generate_answer(qa_chain, question: str, use_openai_api: bool, context: str = "") -> dict:
    start_time = time.time()
    full_query = build_query_prompt(question, context)
    result, token_usage = run_embedded_query(qa_chain, full_query, use_openai_api)
    source_documents = result["source_documents"]

    #filtered_docs = [(doc, score) for doc, score in source_documents if score >= 0.5]
    logger.debug(source_documents)

    sources = []
    for doc in source_documents:
        source_path = doc.metadata.get("source", "Unknown")
        chunk_text = (doc.page_content or "").strip().replace("\n", " ")
        page_number = doc.metadata.get("page")
        page_one_based = int(page_number) + 1 if isinstance(page_number, int) else None
        search_text = " ".join(chunk_text.split())[:120]
        if len(chunk_text) > 220:
            chunk_text = chunk_text[:220].rstrip() + "..."
        sources.append({
            "source": source_path,
            "chunk_preview": chunk_text,
            "page": page_one_based,
            "search_text": search_text,
        })

    latency_ms = int((time.time() - start_time) * 1000)
    print(result)
    quality_metrics = build_quality_metrics(
        source_documents=source_documents,
        answer=result["result"],
        latency_ms=latency_ms,
        token_usage=token_usage,
    )

    return {
        "answer": result["result"],
        "sources": sources,
        "total_sources": len(sources),
        "token_usage": token_usage,
        "quality_metrics": quality_metrics,
    }