"""Multi-provider embedding adapter (Local CPU, Ollama, NVIDIA NIM, OpenRouter, Gemini, OpenAI)."""

from typing import List


class BaseEmbeddingProvider:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
