# frontend/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

class Config:
    BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://localhost:8000")
    API_VERSION = os.getenv("RAG_API_VERSION", "v1")
    API_BASE = f"{BACKEND_URL}/api/{API_VERSION}"


    def model_names(self, provider):
        return f"{self.API_BASE}/model_names/{provider}"

    def query_reset_vectorstore(self):
        return f"{self.API_BASE}/vectorstore/reset"

    @property
    def estimate_endpoint(self):
        return f"{self.API_BASE}/estimate"

    @property
    def query_init(self):
        return f"{self.API_BASE}/init"


    @property
    def query_endpoint(self):
        return f"{self.API_BASE}/query"

    @property
    def status_endpoint(self):
        return f"{self.API_BASE}/status"

    @property
    def health_endpoint(self):
        return f"{self.API_BASE}/health"

    def __repr__(self):
        """String representation of Config object for debugging."""
        return f"Config(BACKEND_URL='{self.BACKEND_URL}', API_VERSION='{self.API_VERSION}')"

    def __str__(self):
        """Human-readable string representation of Config object."""
        return f"Config: BACKEND_URL={self.BACKEND_URL}, API_VERSION={self.API_VERSION}"

    def __eq__(self, other):
        """Equality comparison between Config objects."""
        if not isinstance(other, Config):
            return False
        return self.BACKEND_URL == other.BACKEND_URL and self.API_VERSION == other.API_VERSION

    def __hash__(self):
        """Hash function for Config objects."""
        return hash((self.BACKEND_URL, self.API_VERSION))
