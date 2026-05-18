def build_query_prompt(question: str, context: str = "") -> str:
    return f"Context: {context}\n\nQuestion: {question}" if context else question


def build_cost_estimation_prompt(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Answer based on the context provided."},
        {"role": "user", "content": build_query_prompt(question, context)},
    ]