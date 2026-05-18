import pytest
from unittest.mock import Mock, patch
from tests.fake_llm import FakeLLM


@pytest.fixture
def mock_llm():
    """Fixture: provides a mock LLM for tests"""
    return FakeLLM()


@pytest.fixture
def mock_ollama():
    """Fixture: patches Ollama class globally"""
    mock = Mock()
    mock.invoke.return_value = "Mocked response"

    with patch('langchain_community.llms.Ollama', return_value=mock):
        yield mock


@pytest.fixture
def client():
    """Fixture: FastAPI test client"""
    from rag_doc_backend.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client