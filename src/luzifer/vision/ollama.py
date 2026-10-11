"""Ollama VLM caption client (separate from the chat model)."""

from __future__ import annotations

import base64

import httpx
from loguru import logger

from luzifer.config import VisionSettings


class VisionOllama:
    """POST /api/chat with images[] for a small local VLM."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url: str | None = None

    async def _get_client(self, base_url: str) -> httpx.AsyncClient:
        if self._client is None or self._base_url != base_url:
            if self._client is not None:
                await self._client.aclose()
            self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=90.0)
            self._base_url = base_url
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def describe(
        self,
        jpeg: bytes,
        settings: VisionSettings,
        *,
        base_url: str,
    ) -> str:
        client = await self._get_client(base_url)
        b64 = base64.b64encode(jpeg).decode("ascii")
        payload = {
            "model": settings.model,
            "stream": False,
            "keep_alive": "30m",
            "messages": [
                {
                    "role": "user",
                    "content": settings.prompt,
                    "images": [b64],
                }
            ],
            "options": {
                "temperature": 0.2,
                "num_predict": settings.num_predict,
            },
        }
        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            return str(content).strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Vision caption failed: {}", exc)
            return ""
