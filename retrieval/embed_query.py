from langchain_community.callbacks.manager import get_openai_callback


def run_embedded_query(qa_chain, query_text: str, use_openai_api: bool):
    token_usage = None
    if use_openai_api:
        with get_openai_callback() as cb:
            result = qa_chain.invoke({"query": query_text})
            token_usage = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost,
            }
    else:
        result = qa_chain.invoke({"query": query_text})

    return result, token_usage