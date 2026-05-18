import datetime
import logging
logger = logging.getLogger("uvicorn.error")

def build_quality_metrics(source_documents: list, answer: str, latency_ms: int, token_usage: dict | None) -> dict:
    scores = []
    for doc in source_documents:
        score = doc.metadata.get("score")
        if score is None:
            score = doc.metadata.get("relevance_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        logger.debug(doc.json())
    if scores:
        # Chroma/L2: Lower is better (0 is perfect match). 
        # Convert to a 0-1 scale where 1 is perfect match.
        # This is an approximation.
        avg_score = sum(scores) / len(scores)
        retrieval_score = max(0.0, 1.0 - (avg_score / 2.0))
    else:
        retrieval_score = None

    return {
        "similarity_score": round(retrieval_score, 4) if retrieval_score is not None else None,
        "num_chunks": len(source_documents),
        "latency_ms": latency_ms,
        "token_count": token_usage["total_tokens"] if token_usage else len(answer.split()),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }

