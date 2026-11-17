"""Tests for config presets, secret redaction, and multi-provider clients."""

from __future__ import annotations

import json

import httpx
import pytest

from axonpg.config import AxonConfig
from axonpg.providers.embeddings import (
    GeminiEmbeddingProvider,
    NullEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAICompatEmbeddingProvider,
    build_embedding_provider,
    cosine_similarity,
)
from axonpg.providers.llm import (
    GeminiLLMProvider,
    OllamaLLMProvider,
    OpenAICompatLLMProvider,
    _extract_json_object,
    build_llm_provider,
)


def test_config_redaction_and_env_roundtrip():
    cfg = AxonConfig(
        mode="lite",
        namespace="alpha",
        db_password="super-secret-db-pass",
        embed_provider="nvidia",
        embed_api_key="nvapi-secret",
        llm_provider="openrouter",
        llm_api_key="sk-or-secret",
    )
    redacted = cfg.redacted_dict()
    assert redacted["db_password"] == "***"
    assert redacted["embed_api_key"] == "***"
    assert redacted["llm_api_key"] == "***"
    assert redacted["namespace"] == "alpha"

    env_text = cfg.to_env_text()
    assert "AXONPG_MODE=lite" in env_text
    assert "AXONPG_EMBED_PROVIDER=nvidia" in env_text


def test_cosine_similarity_and_null_embedder():
    null_emb = NullEmbeddingProvider(dim=384)
    assert null_emb.embed_documents(["hello"]) == [[]]
    assert null_emb.embed_query("hello") == []
    assert null_emb.healthcheck()[0] is True

    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([], [1.0]) == 0.0


def test_openai_compat_embedding_and_llm_providers(monkeypatch):
    seen_requests: list[tuple[str, dict, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_requests.append((str(request.url), dict(request.headers), body))
        if str(request.url).endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '```json\n{"scenes": [{"summary": "ok"}]}\n```'}}]},
        )

    transport = httpx.MockTransport(handler)
    orig_client = httpx.Client

    class PatchedClient(orig_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("axonpg.providers.embeddings.httpx.Client", PatchedClient)
    monkeypatch.setattr("axonpg.providers.llm.httpx.Client", PatchedClient)

    emb = OpenAICompatEmbeddingProvider(
        model_name="baai/bge-m3",
        dim=3,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-test",
    )
    vecs = emb.embed_documents(["hola"])
    assert vecs == [[0.1, 0.2, 0.3]]

    llm = OpenAICompatLLMProvider(
        model_name="meta-llama/llama-3.3-70b-instruct:free",
        base_url="https://openrouter.ai/api/v1",
        api_key="or-test",
        provider_label="openrouter",
    )
    parsed = llm.complete_json("sys", "usr")
    assert parsed == {"scenes": [{"summary": "ok"}]}
    assert len(seen_requests) == 2
    assert seen_requests[1][1]["x-title"] == "AxonPG"


def test_ollama_and_gemini_providers(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/api/embed"):
            return httpx.Response(200, json={"embeddings": [[0.5, 0.5]]})
        if url.endswith("/api/chat"):
            return httpx.Response(200, json={"message": {"content": '{"scenes": []}'}})
        if "batchEmbedContents" in url:
            return httpx.Response(200, json={"embeddings": [{"values": [0.9, 0.1]}]})
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": '{"scenes": [{"summary": "ok"}]}'}]}}]},
        )

    transport = httpx.MockTransport(handler)
    orig_client = httpx.Client

    class PatchedClient(orig_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("axonpg.providers.embeddings.httpx.Client", PatchedClient)
    monkeypatch.setattr("axonpg.providers.llm.httpx.Client", PatchedClient)

    ol_emb = OllamaEmbeddingProvider(model_name="bge-m3", dim=2)
    assert ol_emb.embed_documents(["teste"]) == [[0.5, 0.5]]

    ol_llm = OllamaLLMProvider(model_name="qwen2.5:7b")
    assert ol_llm.complete_json("s", "u") == {"scenes": []}

    gem_emb = GeminiEmbeddingProvider(model_name="gemini-embedding-001", dim=2, api_key="gem-key")
    assert gem_emb.embed_documents(["bonjour"]) == [[0.9, 0.1]]

    gem_llm = GeminiLLMProvider(model_name="gemini-2.5-flash", api_key="gem-key")
    assert gem_llm.complete_json("s", "u") == {"scenes": [{"summary": "ok"}]}


def test_factory_builds_expected_types():
    cfg = AxonConfig(embed_provider="none", llm_provider="none")
    assert isinstance(build_embedding_provider(cfg), NullEmbeddingProvider)
    assert build_llm_provider(cfg) is None
    assert _extract_json_object('{"a": 1}') == {"a": 1}
