"""Multi-provider embedding adapters supporting local ONNX, Ollama, NVIDIA NIM, OpenRouter, Gemini, OpenAI, and custom endpoints."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import httpx

from axonpg.config import AxonConfig

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity safely between two equal-length vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingProvider(ABC):
    """Abstract base class for dense text embedding backends."""

    model_name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of stored documents/memories."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query (override for asymmetric retrieval models)."""
        res = self.embed_documents([query])
        return res[0] if res else []

    def healthcheck(self) -> tuple[bool, str]:
        """Verify provider connectivity and vector dimensionality."""
        try:
            vec = self.embed_query("healthcheck")
            if len(vec) == 0:
                return False, "Provider returned empty vector."
            if len(vec) != self.dim:
                return False, f"Dimension mismatch: expected {self.dim}, got {len(vec)}."
            return True, f"ok ({self.model_name}, dim={len(vec)})"
        except Exception as exc:
            return False, str(exc)


class NullEmbeddingProvider(EmbeddingProvider):
    """Graceful keyword/graph-only mode when embeddings are disabled."""

    def __init__(self, dim: int = 384):
        self.model_name = "none"
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return []

    def healthcheck(self) -> tuple[bool, str]:
        return True, "disabled (keyword + graph mode)"


class FastEmbedProvider(EmbeddingProvider):
    """Zero-API local ONNX CPU embedding provider via fastembed."""

    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _prepare(self, texts: list[str], is_query: bool) -> list[str]:
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            return [f"{prefix}{t}" for t in texts]
        return texts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        prepared = self._prepare(texts, is_query=False)
        return [vec.tolist() for vec in model.embed(prepared)]

    def embed_query(self, query: str) -> list[float]:
        model = self._get_model()
        prepared = self._prepare([query], is_query=True)
        vecs = list(model.embed(prepared))
        return vecs[0].tolist() if vecs else []


class OpenAICompatEmbeddingProvider(EmbeddingProvider):
    """
    Unified HTTP client for OpenAI-compatible `/embeddings` APIs:
    OpenAI, NVIDIA NIM, OpenRouter, vLLM, LM Studio, and custom endpoints.
    """

    def __init__(
        self,
        model_name: str,
        dim: int,
        base_url: str,
        api_key: str = "",
        provider_label: str = "openai",
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider_label = provider_label
        self.timeout = timeout

    def _request(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider_label == "openrouter":
            headers.setdefault("HTTP-Referer", "https://github.com/ricardo2591/axonpg")
            headers.setdefault("X-Title", "AxonPG")

        payload: dict[str, object] = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        # NVIDIA NIM asymmetric models (e.g. nv-embedqa, llama-3.2-nv-embedqa) require input_type
        if self.provider_label == "nvidia" and "nv-embed" in self.model_name.lower():
            payload["input_type"] = input_type
            payload["truncate"] = "END"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return [list(map(float, item["embedding"])) for item in items]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._request(texts, input_type="passage")

    def embed_query(self, query: str) -> list[float]:
        vecs = self._request([query], input_type="query")
        return vecs[0] if vecs else []


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local/remote Ollama `/api/embed` client."""

    def __init__(
        self,
        model_name: str,
        dim: int,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
        return [list(map(float, vec)) for vec in data.get("embeddings", [])]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini Generative Language API REST embedding provider."""

    def __init__(
        self,
        model_name: str,
        dim: int,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 30.0,
    ):
        self.model_name = model_name.removeprefix("models/")
        self.dim = dim
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        requests_payload = [
            {
                "model": f"models/{self.model_name}",
                "content": {"parts": [{"text": txt}]},
                "taskType": task_type,
                "outputDimensionality": self.dim,
            }
            for txt in texts
        ]
        url = f"{self.base_url}/models/{self.model_name}:batchEmbedContents"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                url,
                params={"key": self.api_key},
                json={"requests": requests_payload},
            )
            resp.raise_for_status()
            data = resp.json()
        return [list(map(float, item["values"])) for item in data.get("embeddings", [])]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> list[float]:
        vecs = self._embed_batch([query], task_type="RETRIEVAL_QUERY")
        return vecs[0] if vecs else []


def build_embedding_provider(config: AxonConfig) -> EmbeddingProvider:
    """Instantiate the configured embedding backend without external coupling."""
    prov = config.embed_provider.lower()
    if prov == "none":
        return NullEmbeddingProvider(dim=config.embed_dim)
    if prov == "fastembed":
        return FastEmbedProvider(model_name=config.embed_model, dim=config.embed_dim)
    if prov == "ollama":
        return OllamaEmbeddingProvider(
            model_name=config.embed_model,
            dim=config.embed_dim,
            base_url=config.embed_base_url or "http://127.0.0.1:11434",
        )
    if prov == "gemini":
        if not config.embed_api_key:
            logger.warning("AXONPG_EMBED_API_KEY missing for gemini; degrading to keyword+graph mode.")
            return NullEmbeddingProvider(dim=config.embed_dim)
        return GeminiEmbeddingProvider(
            model_name=config.embed_model,
            dim=config.embed_dim,
            api_key=config.embed_api_key,
            base_url=config.embed_base_url or "https://generativelanguage.googleapis.com/v1beta",
        )
    if prov in ("openai", "nvidia", "openrouter", "custom"):
        if prov in ("openai", "nvidia", "openrouter") and not config.embed_api_key:
            logger.warning("AXONPG_EMBED_API_KEY missing for %s; degrading to keyword+graph mode.", prov)
            return NullEmbeddingProvider(dim=config.embed_dim)
        return OpenAICompatEmbeddingProvider(
            model_name=config.embed_model,
            dim=config.embed_dim,
            base_url=config.embed_base_url,
            api_key=config.embed_api_key,
            provider_label=prov,
        )
    raise ValueError(f"Unsupported embedding provider: {config.embed_provider}")
