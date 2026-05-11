import os
import requests
from langchain_openai import OpenAI


def get_open_ai_models():
    """Fetch models from Ollama API"""
    try:
        response = requests.get("https://api.openai.com/v1/models",
                                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        # Check if request succeeded
        if response.status_code == 200:
            # ✅ .json() already returns Python object - NO json.loads() needed!
            data = response.json()
            models = data.get("data", [])

            # Extract "id" from each model object
            model_ids = [model["id"] for model in models]
            filtered = [item for item in model_ids if item.startswith("gpt-")]
            return filtered

    except Exception as e:
        print(f"Error fetching models: {e}")

    # Fallback
    return ["gpt-5.5", "gpt-5.5 pro", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.3-codex", "gpt-4.1",
            "gpt-4.1-mini", "o3 / o3-pro"]

def validate_openai_key(api_key=os.getenv("OPENAI_API_KEY")):
    """Validate OpenAI API key by listing models"""
    try:
        OpenAI(api_key=api_key)

        return True
    except Exception as e:
        print(e)
        return False

