import os
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def create_embeddings(use_openai_api: bool, model_name: str):
    if use_openai_api:
        return OpenAIEmbeddings()

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaEmbeddings(model=model_name, base_url=ollama_base_url)