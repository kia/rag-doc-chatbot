from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def create_embeddings(use_openai_api: bool, model_name: str):
    if use_openai_api:
        return OpenAIEmbeddings()

    return OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")