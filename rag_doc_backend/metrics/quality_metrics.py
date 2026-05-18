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
    return {
        "similarity_score": round(sum(scores) / len(scores), 4) if scores else None,
        "num_chunks": len(source_documents),
        "latency_ms": latency_ms,
        "token_count": token_usage["total_tokens"] if token_usage else len(answer.split()),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }

