import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Mock both the LLM and Embeddings to avoid real API/local calls
with patch('langchain_ollama.ChatOllama'), \
     patch('langchain_ollama.OllamaEmbeddings'), \
     patch('langchain_chroma.Chroma'):
    from rag_doc_backend.main import app


@pytest.fixture
def client():
    """Test client for FastAPI"""
    return TestClient(app)


@pytest.fixture
def mock_llm():
    """Mock LLM with predefined responses"""
    llm = Mock()
    llm.invoke.return_value = Mock(content="The company revenue is €8.7M (mocked).")
    return llm


@pytest.fixture(autouse=True)
def mock_rag_engine():
    """Automatically mock RAGEngine for all tests in this file to avoid document loading"""
    with patch('rag_doc_backend.main.RAGEngine') as mock:
        yield mock

class TestRAGAPI:
    """Test suite for RAG API"""

    def test_health_endpoint(self, client):
        """Test health check – no mocking needed"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_query_endpoint(self, client, mock_llm, mock_rag_engine):
        """Test query with mocked LLM"""
        # Patch RAGEngine's llm and vectorstore to avoid real initialization/calls

        mock_engine = mock_rag_engine.return_value
        mock_engine.query.return_value = {
            "answer": "The company revenue is €8.7M (mocked).",
            "sources": [
                {
                    "source": "docs/report.pdf",
                    "chunk_preview": "Revenue was €8.7M in 2023.",
                    "page": 1,
                    "search_text": "Revenue was €8.7M"
                }
            ],
            "total_sources": 1,
            "token_usage": {"total_tokens": 100},
            "quality_metrics": {"faithfulness": 1.0}
        }

        response = client.post(
            "/api/v1/query",
            json={"query": "What is the revenue? (any string here)", "model_name": "test_model", "model_family": "test_family"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The company revenue is €8.7M (mocked)."
        assert "€8.7M" in data["answer"]

    def test_query_empty_query(self, client):
        """Test validation – empty query"""
        response = client.post(
            "/api/v1/query",
            json={"query": "", "model_name": "test_model", "model_family": "test_family"}
        )
        assert response.status_code == 422

    def test_query_error_handling(self, client, mock_rag_engine):
        """Test error handling when LLM fails"""

        mock_engine = mock_rag_engine.return_value
        mock_engine.query.side_effect = Exception("LLM unavailable")

        response = client.post(
            "/api/v1/query",
            json={"query": "Test?", "model_name": "m", "model_family": "f"}
        )

        assert response.status_code == 500
        assert "LLM unavailable" in response.json()["detail"]

    def test_init(self, client, mock_rag_engine):
        mock_engine = mock_rag_engine.return_value
        mock_engine.init.return_value = {"status": "success"}
        response = client.get("/api/v1/init", params={"model_family": "openai", "model_name": "test_model"})
        assert response.status_code == 200
        assert response.json() == {"status": "success"}

    def test_estimate_query_cost(self, client, mock_rag_engine):
        mock_engine = mock_rag_engine.return_value
        mock_engine.estimate_query_cost.return_value = {"cost": 0.01}
        response = client.post("/api/v1/estimate", json={"query": "Test?", "model_family": "openai", "model_name": "test_model", "context": "test_context"})
        assert response.status_code == 200



