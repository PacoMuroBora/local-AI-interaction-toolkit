"""Async Ollama chat client."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from luzifer.config import LlmSettings

ChatMessage = dict[str, str]


class OllamaEngine:
    """Talk to a local Ollama server over HTTP."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url: str | None = None

    async def _get_client(self, base_url: str) -> httpx.AsyncClient:
        if self._client is None or self._base_url != base_url:
            if self._client is not None:
                await self._client.aclose()
            self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=120.0)
            self._base_url = base_url
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        user_text: str,
        system_prompt: str,
        settings: LlmSettings,
        history: list[ChatMessage] | None = None,
    ) -> str:
        client = await self._get_client(settings.base_url)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": settings.model,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "num_predict": settings.max_tokens,
            },
        }
        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            return str(content).strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Ollama chat failed: {}", exc)
            return "I am momentarily silent. Try again in a moment."
