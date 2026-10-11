"""Ollama VLM caption client (separate from the chat model)."""

from __future__ import annotations

import base64
import re

import httpx
from loguru import logger

from luzifer.config import VisionSettings

# Moondream often emits punctuation-only / emoji junk on vague prompts.
_JUNK = re.compile(r"^[\s.!?;:,\-_/\\|@#$%^&*()+=\[\]{}'\"`~<>]+$")
_WORD = re.compile(r"[A-Za-zÀ-ÿ]{2,}")


def is_usable_caption(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 8:
        return False
    if _JUNK.match(t):
        return False
    if len(_WORD.findall(t)) < 2:
        return False
    return True


class VisionOllama:
    """POST /api/chat (fallback /api/generate) with images for a small local VLM."""

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
        prompt = (settings.prompt or "").strip() or (
            "Describe the scene in one or two short factual English sentences. "
            "Name people, pose, clothing, objects, and background. No poetry."
        )
        options = {
            "temperature": 0.1,
            "num_predict": max(32, int(settings.num_predict)),
        }

        # 1) chat API
        try:
            response = await client.post(
                "/api/chat",
                json={
                    "model": settings.model,
                    "stream": False,
                    "keep_alive": "30m",
                    "messages": [
                        {"role": "user", "content": prompt, "images": [b64]},
                    ],
                    "options": options,
                },
            )
            response.raise_for_status()
            data = response.json()
            content = str(data.get("message", {}).get("content", "") or "").strip()
            if is_usable_caption(content):
                return content
            logger.warning("Vision chat junk/empty: {!r}", content[:80])
        except Exception as exc:  # noqa: BLE001
            logger.error("Vision /api/chat failed: {}", exc)

        # 2) generate API (moondream is often happier here)
        try:
            response = await client.post(
                "/api/generate",
                json={
                    "model": settings.model,
                    "stream": False,
                    "keep_alive": "30m",
                    "prompt": prompt,
                    "images": [b64],
                    "options": options,
                },
            )
            response.raise_for_status()
            data = response.json()
            content = str(data.get("response", "") or "").strip()
            if is_usable_caption(content):
                return content
            logger.warning("Vision generate junk/empty: {!r}", content[:80])
        except Exception as exc:  # noqa: BLE001
            logger.error("Vision /api/generate failed: {}", exc)

        return ""
