# tests/fake_llm.py

class FakeLLM:
    """Fake LLM for testing – no API calls, instant responses"""

    def __init__(self, responses: dict = None):
        self.responses = responses or {
            "default": "This is a test response.",
            "revenue": "The revenue is €8.7M.",
            "employees": "The company has 450 employees.",
            "founded": "Founded in 2015.",
        }
        self.call_count = 0
        self.call_history = []

    def invoke(self, prompt: str) -> str:
        """Mock invoke method (LangChain compatible)"""
        self.call_count += 1
        self.call_history.append(prompt)

        # Find matching response
        prompt_lower = prompt.lower()
        for key, response in self.responses.items():
            if key in prompt_lower:
                return response

        return self.responses["default"]

    async def ainvoke(self, prompt: str) -> str:
        """Async version (for async code)"""
        return self.invoke(prompt)