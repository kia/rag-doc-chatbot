from generation.build_prompt import build_query_prompt
from retrieval.embed_query import run_embedded_query


def generate_answer(qa_chain, question: str, use_openai_api: bool, context: str = "") -> dict:
    full_query = build_query_prompt(question, context)
    result, token_usage = run_embedded_query(qa_chain, full_query, use_openai_api)

    sources = []
    for doc in result["source_documents"]:
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

    return {
        "answer": result["result"],
        "sources": sources,
        "total_sources": len(sources),
        "token_usage": token_usage,
    }