import logging
logger = logging.getLogger("uvicorn.error")
import requests
from rag_doc_frontend.config import Config


class RagClient:
    def __init__(self, 
                 model_name, 
                 model_family, 
                 status_callback,
                 k=5):
        self.model_name = model_name
        self.model_family = model_family
        self.status_callback = status_callback

        config = Config()
        response = requests.get(config.query_init, params={"model_family": self.model_family, "model_name": self.model_name, "k": k})
        logger.debug(response.status_code)
        if response.status_code == 200:
            self.status_callback(f"{model_family} model {model_name} initialized ")
        else:
            detail = self._extract_error_detail(response)
            raise Exception(f"Failed to query RAG server: {detail}")

    @staticmethod
    def _extract_error_detail(response):
        try:
            return response.json().get("detail", response.text)
        except Exception:
            return response.text

    @staticmethod
    def get_available_ollama_models():
        return requests.get(Config().model_names(provider="ollama")).json()

    @staticmethod
    def get_open_ai_models():
        return requests.get(Config().model_names(provider="openai")).json()

    @staticmethod
    def query_rag_server(query, model_family, model_name, k):
        json_data = {"query": query, "model_name": model_name, "model_family": model_family, "k" : k}
        response = requests.post(Config().query_endpoint, json=json_data)
        if response.status_code == 200:
            return response.json()
        else:
            detail = RagClient._extract_error_detail(response)
            raise Exception(f"Failed to query RAG server: {detail}")

    @staticmethod
    def estimate_query_cost(query, model_family, model_name, context, k):
        response = requests.post(Config().estimate_endpoint, json={"query": query, "model_family": model_family, "model_name": model_name, "context":context, "k":k})
        if response.status_code == 200:
            return response.json()
        else:
            detail = RagClient._extract_error_detail(response)
            raise Exception(f"Failed to estimate query cost: {detail}")

    @staticmethod
    def reset_vectorstore(model_name, model_family, k):
        params = {"model_family": model_family, "model_name": model_name, "k": k}
        response = requests.get(Config().query_reset_vectorstore(), params=params)
        if response.status_code == 200:
            return response.json()
        else:
            detail = RagClient._extract_error_detail(response)
            raise Exception(f"Failed to reset vectorstore: {detail}")
