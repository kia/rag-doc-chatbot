import logging
import tiktoken
from typing import List, Dict
from rag_doc_backend import build_cost_estimation_prompt
from rag_doc_backend import PricingManager
from tokencost import calculate_all_costs_and_tokens

logger = logging.getLogger("uvicorn.error")

def _count_rag_tokens(
        query: str,
        retrieved_chunks: List[str],
        model: str,
        system_prompt: str = "",
        include_metadata: bool = True
) -> Dict[str, int]:
    """
    Calculate total tokens for a RAG request before sending.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Base tokens
    query_tokens = len(encoding.encode(query))
    system_tokens = len(encoding.encode(system_prompt)) if system_prompt else 0

    # Chunk tokens
    chunk_tokens = [len(encoding.encode(chunk)) for chunk in retrieved_chunks]
    total_chunk_tokens = sum(chunk_tokens)

    # Formatting overhead (approximate)
    # Each chunk typically gets wrapped like: "Context [n]: {chunk_text}"
    formatting_per_chunk = 10  # "Context 1: " + newlines
    formatting_overhead = len(retrieved_chunks) * formatting_per_chunk

    # Metadata overhead (source, page numbers, etc.)
    metadata_overhead = len(retrieved_chunks) * 15 if include_metadata else 0

    # Message structure overhead (OpenAI adds ~3-5 tokens per message)
    message_overhead = 5

    total_input = (
            system_tokens +
            query_tokens +
            total_chunk_tokens +
            formatting_overhead +
            metadata_overhead +
            message_overhead
    )

    return {
        "query_tokens": query_tokens,
        "system_tokens": system_tokens,
        "chunk_tokens": total_chunk_tokens,
        "chunk_count": len(retrieved_chunks),
        "avg_chunk_size": int(total_chunk_tokens / len(retrieved_chunks)) if retrieved_chunks else 0,
        "formatting_overhead": formatting_overhead + metadata_overhead,
        "total_input_tokens": total_input
    }

def estimate_rag_cost(
        query: str,
        retrieved_chunks: List[str],
        model: str = "gpt-5-mini",
        max_output_tokens: int = 250,
        system_prompt: str = ""
) -> Dict:
    """
    Estimate cost for a RAG request.
    """
    # Pricing per 1M tokens (update for current rates)
    pricing_manager = PricingManager()
    pricing = pricing_manager.get_pricing(model)
    logger.debug("get pricing for model :" + model)
    logger.debug("pricing: " + str(pricing))
    token_counts = _count_rag_tokens(query, retrieved_chunks, model, system_prompt)
    logger.debug("rag tokens count: " + str(token_counts))


    input_tokens = token_counts["total_input_tokens"]

    input_cost = input_tokens * pricing["input_cost_per_token"]
    output_cost = max_output_tokens * pricing["output_cost_per_token"]
    total_cost = input_cost + output_cost
    return {
        **token_counts,
        "max_output_tokens": max_output_tokens,
        "estimated_input_cost": round(input_cost, 6),
        "estimated_output_cost": round(output_cost, 6),
        "total_estimated_cost": round(total_cost, 6)
    }


def estimate_query_cost(question, model_name, vectorstore, context="", estimated_output=50):
    """Estimate cost BEFORE making API call"""
    logger.debug(question)
    # 1. If context is empty, retrieve relevant chunks from the vector store
    if not context and vectorstore:
        # Perform retrieval to get the context that would be used in a real query
        docs = vectorstore.similarity_search(question, k=5)
        retrieved_chunks = [doc.page_content for doc in docs]
        context = "\n".join(retrieved_chunks)
    else:
        # If context was provided (e.g., from frontend), split it into chunks for specialized estimation
        retrieved_chunks = [c for c in context.split("\n") if c.strip()] if context else []

    system_prompt = "Answer based on the context provided."
    logger.debug(system_prompt)
    # 2. Use specialized estimate_rag_cost for detailed breakdown
    try:
        rag_estimate = estimate_rag_cost(
            query=question,
            retrieved_chunks=retrieved_chunks,
            model=model_name,
            max_output_tokens=estimated_output,
            system_prompt=system_prompt
        )
    except Exception:
        rag_estimate = {}

    # 3. Use tokencost as a broader fallback or for pricing calculation
    messages = build_cost_estimation_prompt(question, context)
    estimated_completion = "A" * max(1, int(estimated_output))

    try:
        result = calculate_all_costs_and_tokens(
            prompt=messages,
            completion=estimated_completion,
            model=model_name,
        )
    except Exception:
        # Fallback if model is not supported by tokencost library
        result = {
            "prompt_tokens": rag_estimate.get("total_input_tokens", 0),
            "completion_tokens": estimated_output,
            "prompt_cost": 0.0,
            "completion_cost": 0.0
        }

    return {
        "input_tokens": rag_estimate.get("total_input_tokens", result["prompt_tokens"]),
        "estimated_output_tokens": result["completion_tokens"],
        "estimated_total_cost": rag_estimate.get("total_estimated_cost",
                                                 result["prompt_cost"] + result["completion_cost"]),
        "chunk_count": rag_estimate.get("chunk_count", len(retrieved_chunks)),
        "details": rag_estimate  # Optional: provides the full breakdown to the client
    }