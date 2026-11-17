"""Optional lightweight LLM clients for sleep-time semantic consolidation."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from axonpg.config import AxonConfig


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Parse JSON safely even if wrapped in markdown code fences."""
    cleaned = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1)
    return json.loads(cleaned)


class LLMProvider(ABC):
    """Minimal chat-completion interface for background memory consolidation."""

    model_name: str

    @abstractmethod
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Execute a prompt expecting a structured JSON object response."""


class OpenAICompatLLMProvider(LLMProvider):
    """Covers OpenRouter, NVIDIA NIM, OpenAI, vLLM, and LM Studio."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "",
        provider_label: str = "openai",
        timeout: float = 45.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider_label = provider_label
        self.timeout = timeout

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider_label == "openrouter":
            headers.setdefault("HTTP-Referer", "https://github.com/ricardo2591/axonpg")
            headers.setdefault("X-Title", "AxonPG")

        payload = {
            "model": self.model_name,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return _extract_json_object(content)


class OllamaLLMProvider(LLMProvider):
    """Local Ollama `/api/chat` JSON client."""

    def __init__(self, model_name: str, base_url: str = "http://127.0.0.1:11434", timeout: float = 60.0):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return _extract_json_object(data["message"]["content"])


class GeminiLLMProvider(LLMProvider):
    """Google Gemini REST JSON client."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 45.0,
    ):
        self.model_name = model_name.removeprefix("models/")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, params={"key": self.api_key}, json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _extract_json_object(text)


def build_llm_provider(config: AxonConfig) -> LLMProvider | None:
    """Return configured LLM client, or None for zero-cost deterministic consolidation."""
    prov = config.llm_provider.lower()
    if prov == "none":
        return None
    if prov == "ollama":
        return OllamaLLMProvider(
            model_name=config.llm_model,
            base_url=config.llm_base_url or "http://127.0.0.1:11434",
        )
    if prov == "gemini":
        if not config.llm_api_key:
            return None
        return GeminiLLMProvider(
            model_name=config.llm_model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url or "https://generativelanguage.googleapis.com/v1beta",
        )
    if prov in ("openrouter", "nvidia", "openai", "custom"):
        if prov in ("openrouter", "nvidia", "openai") and not config.llm_api_key:
            return None
        return OpenAICompatLLMProvider(
            model_name=config.llm_model,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            provider_label=prov,
        )
    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
